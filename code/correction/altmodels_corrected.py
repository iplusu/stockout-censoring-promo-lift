"""Corrected versions of the empirical alternative-model likelihoods (07_pending_3 NB2/ZIP, 08 run_pA ZIP2 /
mixture-censoring Poisson / mixture-censoring NB2) and their recovery functions. Same model definitions and
parameter bounds as legacy; changes: strict stable tails (mfit_corrected), no 1e-300 floors or lambda clipping,
log-domain mixtures (logaddexp), average-NLL objective, frozen solver contract (mf._minimize: L-BFGS-B maxiter 5000,
validity = stationarity <= 1e-6 on original and standardized scale). The M recession screen gates NB2/ZIP/ZIP2 as in
mf.fit; mixture-censoring models are not gated by it (their flagged-row likelihood stays bounded below as lambda grows,
so the M recession certificate does not apply) -- divergence shows up as nonconvergence.
"""
import math
import numpy as np
import mfit_corrected as mf

lgt = lambda t: (-np.logaddexp(0.0, -t), -np.logaddexp(0.0, t))     # (log pi, log 1-pi)


# ------------------------------------------------------------------ ZIP with zero part pi = logistic(Z gamma)
def zip2_nll(th, P):
    kz = P.Z.shape[1]; b, g = th[:-kz], th[-kz:]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lpi, l1pi = lgt(P.Z @ g); mu = np.exp(P.eta(b)); k = P.K
        z0 = (~P.I) & (k == 0); up = (~P.I) & (k > 0); ll = np.empty(P.n)
        ll[z0] = np.logaddexp(lpi[z0], l1pi[z0] - mu[z0])
        ll[up] = l1pi[up] + mf.pois_logpmf(k[up], mu[up])
        if P.I.any():
            ll[P.I] = l1pi[P.I] + mf.pois_logsf(k[P.I], mu[P.I])
    return -np.mean(ll)


def zip2_grad(th, P):
    kz = P.Z.shape[1]; b, g = th[:-kz], th[-kz:]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lpi, l1pi = lgt(P.Z @ g); pi = np.exp(lpi); tt = P.eta(b); mu = np.exp(tt); k = P.K
        z0 = (~P.I) & (k == 0); up = (~P.I) & (k > 0); sb = np.empty(P.n); st = -pi.copy()
        w0 = np.exp(l1pi[z0] - mu[z0] - np.logaddexp(lpi[z0], l1pi[z0] - mu[z0]))
        sb[z0] = -mu[z0] * w0; st[z0] = (1 - w0) - pi[z0]
        sb[up] = k[up] - mu[up]
        if P.I.any():
            sb[P.I] = np.exp(tt[P.I] + mf.pois_logpmf(k[P.I], mu[P.I]) - mf.pois_logsf(k[P.I], mu[P.I]))
    return -np.concatenate([P.X.T @ sb, P.Z.T @ st]) / P.n


# ------------------------------------------------------------------ mixture censoring (flagged: pi S(K) + (1-pi) f(K))
def _comp(b, lth, P, base):
    mu = np.exp(P.eta(b)); k = P.K
    if base == 'pois':
        lf = mf.pois_logpmf(k, mu); dlf = k - mu
        ls = np.zeros(P.n); dls = np.zeros(P.n)
        if P.I.any():
            ls[P.I] = mf.pois_logsf(k[P.I], mu[P.I]); dls[P.I] = np.exp(P.eta(b)[P.I] + lf[P.I] - ls[P.I])
    else:
        th = math.exp(lth); lf = mf._nb_logpmf(k, mu, th); dlf = k - mu * (k + th) / (th + mu)
        ls = np.zeros(P.n); dls = np.zeros(P.n)
        if P.I.any():
            kc, muc = k[P.I], mu[P.I]; ls[P.I] = mf.nb_logsf(kc, muc, th)
            dls[P.I] = muc * (kc + th) / (th + muc) * np.exp(lf[P.I] - ls[P.I])
    return lf, dlf, ls, dls


def _split(th, P):
    """params: b (p) [, log theta if NB] [, logit pi if P.pi_fix is None]"""
    p = P.p; b = th[:p]; i = p
    lth = None
    if P.base == 'nb':
        lth = th[i]; i += 1
    t = th[i] if P.pi_fix is None else None
    return b, lth, t


def mix_nll(th, P):
    b, lth, t = _split(th, P)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lpi, l1pi = lgt(t) if t is not None else ((0.0, -np.inf) if P.pi_fix == 1.0 else (math.log(P.pi_fix), math.log1p(-P.pi_fix)))
        lf, _, ls, _ = _comp(b, lth, P, P.base)
        ll = lf.copy()
        if P.I.any():
            ll[P.I] = np.logaddexp(lpi + ls[P.I], l1pi + lf[P.I])
    return -np.mean(ll)


def mix_grad(th, P, h=1e-6):
    b, lth, t = _split(th, P)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lpi, l1pi = lgt(t) if t is not None else ((0.0, -np.inf) if P.pi_fix == 1.0 else (math.log(P.pi_fix), math.log1p(-P.pi_fix)))
        lf, dlf, ls, dls = _comp(b, lth, P, P.base)
        s = dlf.copy()
        if P.I.any():
            lh = np.logaddexp(lpi + ls[P.I], l1pi + lf[P.I]); wS = np.exp(lpi + ls[P.I] - lh)
            s[P.I] = wS * dls[P.I] + (1 - wS) * dlf[P.I]
    out = [-(P.X.T @ s) / P.n]
    if P.base == 'nb':
        e = np.zeros_like(th); e[P.p] = h
        out.append([(mix_nll(th + e, P) - mix_nll(th - e, P)) / (2 * h)])
    if t is not None:
        pi = math.exp(lgt(t)[0])
        out.append([-np.sum(wS - pi) / P.n if P.I.any() else 0.0])
    return np.concatenate(out)


# ------------------------------------------------------------------ fitting
def _fit(fun, jac, P, starts, bounds, gscale, gate=True):
    if gate:
        scr, J = mf.screen(P.X, P.K, P.I)
        if scr != 'pass':
            return dict(status={'nonexistent': 'structural_nonexistence', 'unresolved': 'unresolved_screen',
                                'unidentified': 'unidentified'}[scr], valid=False, x=None, nit=0)
    best, recs = None, []
    for x0 in starts:
        if x0 is None or not np.all(np.isfinite(x0)):
            continue
        o = mf._minimize(fun, jac, np.asarray(x0, float), bounds, 5000, P, gscale); recs.append(o)
        if o['converged'] and (best is None or o['obj'] < best['obj'] - 1e-12):
            best = o
    if best is None:
        st = 'nonfinite' if recs and all(not np.isfinite(o['obj']) for o in recs) else 'nonconvergence'
        return dict(status=st, valid=False, x=None, nit=max([o['nit'] for o in recs], default=0))
    return dict(status='valid', valid=True, x=best['x'], obj=best['obj'], nit=best['nit'], g_orig=best['g_orig'], g_std=best['g_std'])


def fit_zip2(X, K, C, Z, starts):
    P = mf.Problem(X, K, np.asarray(C).astype(bool)); P.Z = np.asarray(Z, float)
    sdz = P.Z.std(0); sdz[sdz == 0] = 1.0
    return _fit(zip2_nll, zip2_grad, P, starts, [(None, None)] * P.p + [mf.PI_LOGIT_BOUNDS] * P.Z.shape[1], np.append(P.sd, sdz))


def fit_mix(X, K, C, base, pi_fix, starts):
    P = mf.Problem(X, K, np.asarray(C).astype(bool)); P.base = base; P.pi_fix = pi_fix
    bounds = [(None, None)] * P.p + ([(None, None)] if base == 'nb' else []) + ([mf.PI_LOGIT_BOUNDS] if pi_fix is None else [])
    gs = np.append(P.sd, np.ones(len(bounds) - P.p))
    return _fit(mix_nll, lambda th, P: mix_grad(th, P), P, starts, bounds, gs, gate=False)


# ------------------------------------------------------------------ recovery E[D | D > K] on flagged rows
def recov_pois(b, X, K, C):
    I = np.asarray(C).astype(bool); K = np.asarray(K, float); D = K.copy()
    if b is None or not np.all(np.isfinite(b)):
        return np.full(len(K), np.nan)
    eta = X @ b; mu = np.exp(eta)
    if I.any():
        D[I] = mu[I] + np.exp(eta[I] + mf.pois_logpmf(K[I], mu[I]) - mf.pois_logsf(K[I], mu[I]))
    return D


def recov_nb2(b, th, X, K, C):
    """E[D | D > K] = A/S, A = E[D 1{D>K}] = mu * Pr(NB(th+1, same p) > K-1) (legacy identity), strict log tails."""
    I = np.asarray(C).astype(bool); K = np.asarray(K, float); D = K.copy()
    if b is None or not np.all(np.isfinite(b)):
        return np.full(len(K), np.nan)
    mu = np.exp(X @ b)
    if I.any():
        kc, muc = K[I], mu[I]
        lA = np.log(muc).copy(); pos = kc >= 1
        lA[pos] += mf.nb_logsf(kc[pos] - 1.0, muc[pos] * (th + 1.0) / th, th + 1.0)
        D[I] = np.exp(lA - mf.nb_logsf(kc, muc, th))
    return D


def selfcheck():
    from scipy import stats
    rng = np.random.default_rng(3); n = 300
    X = np.column_stack([np.ones(n), rng.integers(0, 2, n), rng.uniform(0, .5, n)]); K = rng.poisson(3, n).astype(float)
    I = rng.random(n) < .3; Z = np.column_stack([np.ones(n), X[:, 1]])
    res = {}

    def fd(f, g, th, P, h=1e-6):
        num = np.array([(f(th + e, P) - f(th - e, P)) / (2 * h) for e in np.eye(len(th)) * h])
        return float(np.max(np.abs(num - g(th, P))))
    P = mf.Problem(X, K, I); P.Z = Z
    th = np.array([1.0, .2, .5, -1.0, .3]); res['zip2 grad FD'] = fd(zip2_nll, zip2_grad, th, P)
    P1 = mf.Problem(X, K, I); P1.Z = Z[:, :1]
    res['zip2(Z=1) == zip'] = abs(zip2_nll(th[:4], P1) - mf.zip_nll(th[:4], P1))
    for base, pif, tht in [('pois', .85, [1.0, .2, .5]), ('pois', None, [1.0, .2, .5, 1.5]), ('nb', .85, [1.0, .2, .5, 1.2]),
                           ('nb', None, [1.0, .2, .5, 1.2, 1.5])]:
        Q = mf.Problem(X, K, I); Q.base = base; Q.pi_fix = pif
        res[f'mix {base} pi={pif} grad FD'] = fd(mix_nll, lambda t, P: mix_grad(t, P), np.array(tht), Q, 1e-5)
    Q = mf.Problem(X, K, I); Q.base = 'pois'; Q.pi_fix = 1.0
    res['mix pois pi=1 == M'] = abs(mix_nll(np.array([1.0, .2, .5]), Q) - mf.m_nll(np.array([1.0, .2, .5]), Q))
    Q = mf.Problem(X, K, I); Q.base = 'nb'; Q.pi_fix = 1.0
    res['mix nb pi=1 == NB2'] = abs(mix_nll(np.array([1.0, .2, .5, 1.2]), Q) - mf.nb_nll(np.array([1.0, .2, .5, 1.2]), Q))
    Q = mf.Problem(np.ones((1, 1)), np.array([200.0]), np.array([True])); Q.base = 'pois'; Q.pi_fix = 1.0
    res['no floor K=200 mu=1 (exact -869.530)'] = -mix_nll(np.array([0.0]), Q)
    # recovery vs brute force
    mu, th, k = 2.5, 3.0, 4
    d = np.arange(5, 4000); pN = stats.nbinom.pmf(d, th, th / (th + mu)); bf = (d * pN).sum() / pN.sum()
    res['recov_nb2 vs brute'] = abs(recov_nb2(np.array([math.log(mu)]), th, np.ones((1, 1)), np.array([k]), np.array([1]))[0] - bf)
    pP = stats.poisson.pmf(d, mu); bf = (d * pP).sum() / pP.sum()
    res['recov_pois vs brute'] = abs(recov_pois(np.array([math.log(mu)]), np.ones((1, 1)), np.array([k]), np.array([1]))[0] - bf)
    res['recov_nb2 K=0'] = abs(recov_nb2(np.array([math.log(mu)]), th, np.ones((1, 1)), np.array([0]), np.array([1]))[0]
                               - (np.arange(1, 4000) * stats.nbinom.pmf(np.arange(1, 4000), th, th / (th + mu))).sum() / stats.nbinom.sf(0, th, th / (th + mu)))
    ok = (res['zip2 grad FD'] < 1e-7 and res['zip2(Z=1) == zip'] < 1e-12 and all(v < 1e-6 for kk, v in res.items() if 'mix' in kk and 'FD' in kk)
          and res['mix pois pi=1 == M'] < 1e-12 and res['mix nb pi=1 == NB2'] < 1e-12 and abs(res['no floor K=200 mu=1 (exact -869.530)'] + 869.530) < 1e-2
          and res['recov_nb2 vs brute'] < 1e-8 and res['recov_pois vs brute'] < 1e-8 and res['recov_nb2 K=0'] < 1e-8)
    return ok, res


if __name__ == '__main__':
    ok, res = selfcheck()
    for k, v in res.items():
        print(f'{k}: {v:.3e}' if abs(v) < 1 else f'{k}: {v:.4f}')
    print('ALL PASS' if ok else 'FAIL')
