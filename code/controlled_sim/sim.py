"""Controlled simulation core (protocol v1.0 §2-7). One code path for evaluation and engineering IDs.

Public entry: run_replicate(rid, seedseq, cal) -> dict of row lists (datasets, fits, starts, losses, information).
"""
import json, math, time, warnings
import numpy as np
from scipy import optimize, special, stats

CFG = json.load(open('config.json'))
STRATA = CFG['strata']
N = CFG['n_obs']
P_COL = np.concatenate([np.full(s['n'], s['P'], float) for s in STRATA])
D_COL = np.concatenate([np.full(s['n'], s['d'], float) for s in STRATA])
S_IDX = np.concatenate([np.full(s['n'], j) for j, s in enumerate(STRATA)])
X = np.column_stack([np.ones(N), P_COL, D_COL])
BETA_TRUE = np.array([math.log(20.0), 0.20, 1.0])
LAM_S = np.array([math.exp(BETA_TRUE @ [1, s['P'], s['d']]) for s in STRATA])
DEPTHS = np.array([0.05, 0.15, 0.30])
L_TRUE = np.expm1(0.20 + DEPTHS)
CONDS = [(e, q) for e in CFG['conditions']['eta'] for q in CFG['conditions']['q']]
TAIL_SWITCH = 1e-250
GTOL = 1e-6


# ---------------------------------------------------------------- tails (strict: Pr(Y > k))
def _series_logsf(k, mu, tol=1e-17):
    """log Pr(Y>k) = logpmf(k+1) + log(1 + sum_{m>=1} mu^m / prod_{j=2}^{m+1}(k+j)); remainder < last term."""
    lp = -mu + (k + 1) * math.log(mu) - math.lgamma(k + 2)
    t = s = 1.0
    j = 2
    while True:
        r = mu / (k + j)
        t *= r
        s += t
        j += 1
        if r < 0.5 and t < tol * s:
            return lp + math.log(s)


def logsf(k, mu):
    """Vectorized strict log-survival log Pr(Y > k), Y ~ Pois(mu). No floor."""
    k = np.asarray(k, float)
    mu = np.asarray(mu, float)
    shape = np.broadcast(k, mu).shape
    k, mu = np.broadcast_to(k, shape).ravel(), np.broadcast_to(mu, shape).ravel()
    with np.errstate(divide='ignore'):
        S = special.gammainc(k + 1.0, mu)
        out = np.log(S)            # 1-d writable array (0-d input previously lost series values)
    for i in np.flatnonzero(~(S > TAIL_SWITCH) & np.isfinite(mu) & (mu > 0)):
        out[i] = _series_logsf(k[i], mu[i])
    return out.reshape(shape)


def logpmf(k, mu):
    return k * np.log(mu) - mu - special.gammaln(k + 1.0)


def _curv_bracket_series(k, mu, tol=1e-17):
    """mu + h - k - 1 without cancellation, valid for k + 1 > 2 mu.
    S = pmf(k+1) * T, T = 1 + sum_{m>=1} mu^m / prod_{j=2}^{m+1}(k+j)  =>  h = (k+1)/T,
    mu + h - k - 1 = mu - (k+1) (T-1)/T, and T-1 is a positive-term sum."""
    t, tm1, j = 1.0, 0.0, 2
    while True:
        t *= mu / (k + j)
        tm1 += t
        j += 1
        if t < tol * (1.0 + tm1):
            T = 1.0 + tm1
            return mu - (k + 1.0) * (tm1 / T)


# ---------------------------------------------------------------- objective / gradient / Hessian
def _parts(b, y, I):
    t = X @ b
    mu = np.exp(t)
    if I is None or not I.any():
        return mu, None
    logh = t[I] + logpmf(y[I], mu[I]) - logsf(y[I], mu[I])
    return mu, np.exp(logh)


def nll(b, y, I=None):
    """Average negative log likelihood; I=None -> ordinary Poisson."""
    t = X @ b
    mu = np.exp(t)
    if I is None:
        return -np.mean(y * t - mu - special.gammaln(y + 1.0))
    ll = np.where(I, 0.0, y * t - mu - special.gammaln(y + 1.0))
    if I.any():
        ll[I] = logsf(y[I], mu[I])
    return -np.mean(ll)


def grad(b, y, I=None):
    mu, h = _parts(b, y, I)
    s = y - mu
    if h is not None:
        s = s.copy()
        s[I] = h
    return -(X.T @ s) / N


def curv_bracket(k, mu, h):
    """Vectorized (mu + h - k - 1); series form where k+1 > 2 mu (far tail, cancellation-prone)."""
    out = mu + h - k - 1.0
    for i in np.flatnonzero(k + 1.0 > 2.0 * mu):
        out[i] = _curv_bracket_series(k[i], mu[i])
    return out


def hess(b, y, I=None):
    mu, h = _parts(b, y, I)
    w = mu.copy()
    if h is not None:
        w[I] = h * curv_bracket(y[I], mu[I], h)
    return (X.T * w) @ X / N


# ---------------------------------------------------------------- data generation (§2-3)
def _qpois(z, mean, counter):
    u = stats.norm.cdf(z)
    lo, hi = u == 0.0, u == 1.0
    counter[0] += int(lo.sum() + hi.sum())
    u = np.where(lo, np.nextafter(0.0, 1.0), np.where(hi, np.nextafter(1.0, 0.0), u))
    return stats.poisson.ppf(u, mean).astype(np.int64)


def draw_latent(seedseq):
    rng = np.random.Generator(np.random.PCG64(seedseq))
    zd = rng.standard_normal(N)
    zc = rng.standard_normal(N)
    return zd, zc


def make_demand(zd):
    cnt = [0]
    D = _qpois(zd, LAM_S[S_IDX], cnt)
    return D, cnt[0]


def make_condition(zd, zc, D, eta, q, cal):
    kap = np.array([cal[(s['name'], eta, q)] for s in STRATA])[S_IDX]
    cnt = [0]
    C = _qpois(eta * zd + math.sqrt(1.0 - eta * eta) * zc, kap, cnt)
    K = np.minimum(D, C)
    I = D > C
    return K, I, cnt[0]


def svd_rank(A):
    if A.shape[0] == 0:
        return 0
    sv = np.linalg.svd(A, compute_uv=False)
    tol = max(A.shape[0], 3) * np.finfo(float).eps * sv[0]
    return int((sv > tol).sum())


# ---------------------------------------------------------------- finite-maximum screen (§5)
def screen(y, I):
    """Returns (status, J*) with status in {'pass','nonexistent','unresolved'}."""
    I = np.zeros(N, bool) if I is None else I
    Up = np.unique(X[(~I) & (y > 0)], axis=0)
    U0 = np.unique(X[(~I) & (y == 0)], axis=0)
    C1 = np.unique(X[I], axis=0)
    if svd_rank(Up) == 3:
        return 'pass', 0.0
    c = -(C1.sum(0) - U0.sum(0))          # linprog minimizes
    A_ub = np.vstack([U0, -C1]) if (len(U0) + len(C1)) else None
    b_ub = np.zeros(len(U0) + len(C1)) if A_ub is not None else None
    A_eq = Up if len(Up) else None
    b_eq = np.zeros(len(Up)) if len(Up) else None
    r = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(-1, 1)] * 3, method='highs')
    if r.status != 0:
        return 'unresolved', float('nan')
    v = r.x
    res = max([0.0] + list(np.abs(Up @ v)) + list(np.maximum(U0 @ v, 0)) + list(np.maximum(-(C1 @ v), 0))
              + list(np.maximum(np.abs(v) - 1, 0)))
    if res > 1e-9:
        return 'unresolved', float('nan')
    J = -r.fun
    if J > 1e-8:
        return 'nonexistent', J
    if J >= -1e-8:
        return 'pass', J
    return 'unresolved', J


# ---------------------------------------------------------------- fitting (§5)
def _one_start(b0, y, I):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(all='ignore'):
            r = optimize.minimize(nll, b0, args=(y, I), jac=grad, method='L-BFGS-B',
                                  options=dict(maxiter=2000, maxfun=20000, maxls=50, maxcor=10,
                                               gtol=GTOL, ftol=0.0))
            g = grad(r.x, y, I)
            f = nll(r.x, y, I)
    gn = float(np.max(np.abs(g))) if np.all(np.isfinite(g)) else float('inf')
    finite = bool(np.all(np.isfinite(r.x)) and np.isfinite(f))
    conv = bool(r.success) and finite and gn <= GTOL
    return dict(b=r.x, obj=float(f), gnorm=gn, nit=int(r.nit), nfev=int(r.nfev), success=bool(r.success),
                msg=str(r.message), finite=finite, converged=conv)


def fit_model(model, y, I, b_g=None):
    """Returns (fit_record, start_records). model in G/M/O."""
    scr, J = screen(y, I if model == 'M' else None)
    base = np.array([math.log(max(float(np.mean(y)), 1e-8)), 0.0, 0.0])
    if model == 'M':
        b = b_g if b_g is not None else base
        inits = [b, b + [0, 0.1, 0], b + [0, 0, 0.1]]
        init_src = 'G_valid' if b_g is not None else 'logmeanK'
    else:
        inits, init_src = [base], 'logmean'
    starts, best = [], None
    for j, b0 in enumerate(inits):
        rec = dict(start=j, init=json.dumps([float(v) for v in b0]), init_source=init_src)
        if scr != 'pass':
            rec.update(attempted=False, skip_reason=f'screen_{scr}')
            starts.append(rec)
            continue
        o = _one_start(np.array(b0, float), y, I if model == 'M' else None)
        rec.update(attempted=True, skip_reason='', solver_success=o['success'], solver_msg=o['msg'], objective=o['obj'],
                   gnorm=o['gnorm'], nit=o['nit'], nfev=o['nfev'], converged=o['converged'],
                   b0_hat=o['b'][0], bP_hat=o['b'][1], bd_hat=o['b'][2])
        starts.append(rec)
        if o['converged'] and (best is None or o['obj'] < best[1]['obj'] - 1e-12):
            best = (j, o)
    flags = dict(f_struct_nonexist=scr == 'nonexistent', f_struct_unresolved=scr == 'unresolved',
                 f_nonconv=False, f_nonfinite_obj_coef=False, f_nonfinite_loss=False)
    fit = dict(model=model, screen=scr, screen_J=J, selected_start=None, b0_hat=np.nan, bP_hat=np.nan, bd_hat=np.nan,
               objective=np.nan, gnorm=np.nan, nit=np.nan)
    if scr == 'pass':
        if best is None:
            flags['f_nonconv'] = True
            att = [s for s in starts if s.get('attempted')]
            flags['f_nonfinite_obj_coef'] = all(not np.isfinite(s['objective']) or not np.all(np.isfinite(
                [s['b0_hat'], s['bP_hat'], s['bd_hat']])) for s in att)
        else:
            j, o = best
            fit.update(selected_start=j, b0_hat=o['b'][0], bP_hat=o['b'][1], bd_hat=o['b'][2], objective=o['obj'],
                       gnorm=o['gnorm'], nit=o['nit'])
    loss = losses(fit)
    if fit['selected_start'] is not None and not loss['finite']:
        flags['f_nonfinite_loss'] = True
    for name, st in [('f_struct_nonexist', 'structural_nonexistence'), ('f_struct_unresolved', 'unresolved_screen'),
                     ('f_nonconv', 'nonconvergence'), ('f_nonfinite_obj_coef', 'nonfinite_objective_coefficient'),
                     ('f_nonfinite_loss', 'nonfinite_lift_loss')]:
        if flags[name]:
            fit['status'] = st
            break
    else:
        fit['status'] = 'valid'
    fit.update(flags, valid=fit['status'] == 'valid')
    if not fit['valid']:   # §6: failed fits receive missing losses, never zeros/caps
        loss = {k: (np.nan if k != 'finite' else False) for k in loss}
    return fit, starts, loss


def losses(fit):
    bP, bd = fit['bP_hat'], fit['bd_hat']
    with np.errstate(all='ignore'):
        Lh = np.expm1(bP + DEPTHS * bd)
        ae = 100 * np.abs(Lh - L_TRUE)
        A = float(np.mean(ae))
        EP, Ed = abs(bP - 0.20) / 0.20, abs(bd - 1.0)
    vals = list(Lh) + list(ae) + [A, EP, Ed]
    out = dict(L_05=Lh[0], L_15=Lh[1], L_30=Lh[2], ae_05_pp=ae[0], ae_15_pp=ae[1], ae_30_pp=ae[2],
               A_pp=A, E_P=EP, E_d=Ed)
    out['finite'] = bool(np.all(np.isfinite(vals)))
    return out


def information(fit, y, I):
    rec = dict(finite_candidate=fit['selected_start'] is not None, normalized_by_n=True)
    if not rec['finite_candidate']:
        rec.update(diag_status='no_candidate')
        return rec
    b = np.array([fit['b0_hat'], fit['bP_hat'], fit['bd_hat']])
    with np.errstate(all='ignore'):
        H = hess(b, y, I)
    for i in range(3):
        for j in range(i, 3):
            rec[f'H{i}{j}'] = H[i, j]
    if not np.all(np.isfinite(H)):
        rec.update(diag_status='missing', cond=np.nan)
        return rec
    ev = np.linalg.eigvalsh(H)
    tau = 1e-10 * max(1.0, abs(ev[-1]))
    rec.update(eig_min=ev[0], eig_mid=ev[1], eig_max=ev[-1])
    if ev[0] > tau:
        rec.update(diag_status='ok', cond=ev[-1] / ev[0])
    elif ev[0] >= -tau:
        rec.update(diag_status='singular_or_nearly', cond=np.inf)
    else:
        rec.update(diag_status='indefinite', cond=np.nan)
    return rec


# ---------------------------------------------------------------- one replicate (all six conditions + oracle)
def load_calibration(path='calibration.csv'):
    import csv
    return {(r['stratum'], float(r['eta']), float(r['q'])): float(r['kappa']) for r in csv.DictReader(open(path))}


def run_replicate(rid, seedseq, cal, stream='evaluation'):
    zd, zc = draw_latent(seedseq)
    D, adjD = make_demand(zd)
    out = dict(datasets=[], fits=[], starts=[], losses=[], information=[])
    seed_id = f'{stream}:{rid}:' + ','.join(map(str, seedseq.spawn_key))
    # oracle once per replicate
    fo, so, lo = fit_model('O', D.astype(float), None)
    key = dict(rid=rid, condition='')
    out['fits'].append({**key, **fo})
    out['starts'] += [{**key, 'model': 'O', **s} for s in so]
    out['losses'].append({**key, 'model': 'O', **lo})
    for eta, q in CONDS:
        cond = f'eta{eta}_q{q}'
        K, I, adjC = make_condition(zd, zc, D, eta, q, cal)
        unc = ~I
        ds = dict(rid=rid, condition=cond, eta=eta, q=q, seed_id=seed_id, cens_share=float(I.mean()),
                  uncens_rank=svd_rank(X[unc]), boundary_adj_D=adjD, boundary_adj_C=adjC,
                  demand_sha=hash_arr(D))
        for j, s in enumerate(STRATA):
            m = S_IDX == j
            ds[f'cens_share_{s["name"]}'] = float(I[m].mean())
            ds[f'n_uncens_{s["name"]}'] = int(unc[m].sum())
        ds['min_uncens_stratum'] = min(ds[f'n_uncens_{s["name"]}'] for s in STRATA)
        out['datasets'].append(ds)
        key = dict(rid=rid, condition=cond)
        Kf = K.astype(float)
        _t = time.perf_counter()
        fg, sg, lg = fit_model('G', Kf, None)
        bg = np.array([fg['b0_hat'], fg['bP_hat'], fg['bd_hat']]) if fg['valid'] else None
        fm, sm, lm = fit_model('M', Kf, I, b_g=bg)
        for f, s, l, mname in [(fg, sg, lg, 'G'), (fm, sm, lm, 'M')]:
            out['fits'].append({**key, **f})
            out['starts'] += [{**key, 'model': mname, **x} for x in s]
            out['losses'].append({**key, 'model': mname, **l, 'reason': '' if f['valid'] else f['status']})
        out['information'].append({**key, **information(fm, Kf, I)})
        ds['fit_seconds_GM'] = time.perf_counter() - _t
    return out


def hash_arr(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
