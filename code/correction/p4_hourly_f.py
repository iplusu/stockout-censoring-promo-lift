"""P4: hourly allocation (mask fix, two steps) for the 254 retained cells (pilot = cell 0,4) and the auxiliary
store-day F model for the original 20 cells (main_topF.csv), legacy vs corrected on identical inputs.

Hourly (instructions sec. 4, plan Q3-B):
  H0 legacy replay  : profile k_h[~C] (integer-index bug), legacy daily recovery (legacy M, floored tail)
  HA mask only      : profile k_h[C == 0], same legacy daily recovery          -> isolates the mask effect
  HB mask + M       : profile k_h[C == 0], corrected daily recovery lambda + h from the corrected M (valid fits only)
  No uncensored row -> hourly result 'unavailable' (no fallback). Zero-profile fallback (equal split over flagged
  hours) is kept as in legacy and counted.
F (instructions sec. 5): joint MLE of slopes and store-day offsets with the corrected stable likelihood.
  Identification convention: intercept and every column constant within all store-days are absorbed by the offsets
  (reported 'absorbed', not estimated). Groups whose offset has no finite MLE (all rows uncensored with K = 0 ->
  -inf; all rows censored -> +inf) carry no slope information and are separated out (Correia, Guimaraes & Zylkin 2019
  convention for separated FE groups); their counts are reported. A sparse recession LP on the remaining design
  checks that a finite joint MLE exists. Optimizer: damped joint Newton on (slopes, offsets) via the Schur complement
  of the diagonal offset block, Armijo backtracking on the current joint objective; backtracking exhaustion = failed.
  Valid = joint stationarity (slopes: average-scale sup-norm <= 1e-6 original and standardized; offsets: per-group
  mean score <= 1e-6) and objective stability (last relative change <= 1e-12).
  python p4_hourly_f.py --cells 12:29 --test      # quick
  python p4_hourly_f.py --all --workers 8         # production (resume by out/p4/<city>_<cat>.npz)
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import optimize, sparse

H = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.dirname(H)
GTOL = 1e-6


def legacy():
    sys.path.insert(0, os.path.join(H, 'legacy', 'main')); sys.path.insert(0, os.path.join(H, 'legacy', 'pilot'))
    import importlib
    return importlib.import_module('tobit'), importlib.import_module('05_decomp'), importlib.import_module('03_main')


# ---------------------------------------------------------------- hourly
def apportion_fixed(k_h, ST, C, K, Dhat):
    """Corrected allocation. Returns (hourly matrix or None if no uncensored row, diagnostics)."""
    unc = (C == 0)
    if not unc.any():
        return None, dict(unavailable=True)
    P = k_h[unc].mean(axis=0)
    L = np.maximum(Dhat - K, 0.0)
    den = (ST * P[None, :]).sum(axis=1)
    out = k_h.astype(np.float64).copy()
    good = (C == 1) & (den > 1e-9)
    out[good] += L[good, None] * (ST[good] * P[None, :] / den[good, None])
    bad = (C == 1) & (den <= 1e-9)
    if bad.any():
        out[bad] += L[bad, None] * (ST[bad] / np.maximum(ST[bad].sum(axis=1)[:, None], 1))
    return out, dict(unavailable=False, n_zero_profile_fallback=int(bad.sum()), n_dhat_below_K=int(((C == 1) & (Dhat < K)).sum()))


def check_allocation(out, k_h, ST, C, K, Dhat):
    """Instruction sec. 4 checks on a full cell. Returns dict of booleans/max errors."""
    add = out - k_h
    tot = out.sum(axis=1); target = np.where(C == 1, np.maximum(Dhat, K), K)
    return dict(nonneg=bool((out >= -1e-12).all()), only_flagged_hours=bool((np.abs(add[~ST]) <= 1e-12).all()),
                uncensored_unchanged=bool((add[C == 0] == 0).all()),
                max_daily_total_err=float(np.max(np.abs(tot - target) / np.maximum(1.0, target))))


def profiles(Y, pm):
    return Y[pm].mean(axis=0) - Y[~pm].mean(axis=0), float(Y[pm].mean() - Y[~pm].mean())


# ---------------------------------------------------------------- corrected recovery
def corrected_dhat(b, X, K, I, off=None):
    import mfit_corrected as mf
    P = mf.Problem(X, K, I, off)
    D = K.astype(float).copy()
    if I.any():
        h, mu, _ = mf.m_hazard(b, P)
        D[I] = mu + h            # E[D | D > K] = lambda + lambda f(K)/Pr(D > K)
    return D


def obs_terms(eta, K, I):
    """Per-row NLL score ds/deta and curvature for the corrected strict-tail censored Poisson."""
    import mfit_corrected as mf
    n = len(K); P = mf.Problem(np.zeros((n, 1)), K, I, eta)
    mu = np.exp(eta); s = mu - K; w = mu.copy()
    if I.any():
        h, muc, k = mf.m_hazard(np.zeros(1), P)
        br = muc + h - k - 1.0
        for i in np.flatnonzero(k + 1.0 > 2.0 * muc):
            br[i] = mf._curv_series(k[i], muc[i])
        s[I] = -h; w[I] = h * br       # NLL: d/deta = -h ; d2/deta2 = -h (K+1-mu-h)
    nll = np.where(I, 0.0, mu - K * eta)
    if I.any():
        nll[I] = -mf.pois_logsf(K[I], mu[I])
    return nll, s, w      # nll without the constant lgamma(K+1) term on uncensored rows


# ---------------------------------------------------------------- F model
def sparse_screen(Xw, g, G, K, I):
    """Recession LP on [Xw, group dummies]: J* > 1e-8 -> no finite joint MLE."""
    n = len(K); D = sparse.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, G))
    A = sparse.hstack([sparse.csr_matrix(Xw), D]).tocsr()
    up, u0, ce = (~I) & (K > 0), (~I) & (K == 0), I
    c = -(np.asarray(A[ce].sum(axis=0)).ravel() - np.asarray(A[u0].sum(axis=0)).ravel())
    Aub = sparse.vstack([A[u0], -A[ce]]).tocsr()
    r = optimize.linprog(c, A_ub=Aub if Aub.shape[0] else None, b_ub=np.zeros(Aub.shape[0]) if Aub.shape[0] else None,
                         A_eq=A[up] if up.any() else None, b_eq=np.zeros(int(up.sum())) if up.any() else None,
                         bounds=[(-1, 1)] * A.shape[1], method='highs')
    if r.status != 0:
        return 'unresolved', float('nan')
    J = -r.fun
    return ('nonexistent' if J > 1e-8 else 'pass'), float(J)


def fit_f(X, K, C, sd_inv, b_start, return_params=False):
    """Corrected F. Returns record (no arrays except slopes)."""
    I = C.astype(bool); Kf = K.astype(float)
    G0 = int(sd_inv.max()) + 1
    bfull = np.full(X.shape[1], np.nan); afull = np.full(G0, np.nan)
    ret = (lambda r, D: (r, D, bfull, afull)) if return_params else (lambda r, D: (r, D))
    # within-group variation per column
    var = []
    for j in range(X.shape[1]):
        mn = np.full(G0, np.inf); mx = np.full(G0, -np.inf)
        np.minimum.at(mn, sd_inv, X[:, j]); np.maximum.at(mx, sd_inv, X[:, j])
        var.append(int(((mx - mn) > 1e-12).sum()))
    wcols = [j for j in range(X.shape[1]) if var[j] > 0]
    rec = dict(n=len(K), n_groups=G0, within_varying_groups=var, identified_cols=wcols,
               absorbed_cols=[j for j in range(X.shape[1]) if var[j] == 0])
    ng = np.bincount(sd_inv, minlength=G0); nc = np.bincount(sd_inv, weights=I, minlength=G0)
    sK = np.bincount(sd_inv, weights=Kf, minlength=G0)
    neg = (nc == 0) & (sK == 0); pos = nc == ng
    rec.update(sep_groups_all_zero=int(neg.sum()), sep_groups_all_censored=int(pos.sum()),
               sep_rows=int((neg | pos)[sd_inv].sum()))
    keepg = ~(neg | pos); keep = keepg[sd_inv]
    newid = -np.ones(G0, int); newid[keepg] = np.arange(int(keepg.sum()))
    g = newid[sd_inv[keep]]; G = int(keepg.sum())
    Xw = X[keep][:, wcols]; Kk = Kf[keep]; Ik = I[keep]; n = len(Kk); ngk = np.bincount(g, minlength=G).astype(float)
    rec['n_groups_kept'] = G; rec['n_rows_kept'] = n
    # rank of within-demeaned slope design
    Xd = Xw - (np.stack([np.bincount(g, weights=Xw[:, j], minlength=G) for j in range(len(wcols))], 1) / ngk[:, None])[g]
    rec['within_rank'] = int(np.linalg.matrix_rank(Xd)); rec['p_w'] = len(wcols)
    if rec['within_rank'] < len(wcols):
        rec['status'] = 'unidentified'; return ret(rec, None)
    t = time.time(); scr, J = sparse_screen(Xw, g, G, Kk, Ik); rec.update(screen=scr, screen_J=J, screen_seconds=time.time() - t)
    if scr != 'pass':
        rec['status'] = {'nonexistent': 'structural_nonexistence', 'unresolved': 'unresolved_screen'}[scr]; return ret(rec, None)
    sdw = Xw.std(axis=0)
    beta = np.asarray(b_start, float)[wcols].copy()
    alpha = np.log((np.bincount(g, weights=Kk, minlength=G) + 0.5) / ngk) - np.bincount(g, weights=Xw @ beta, minlength=G) / ngk

    def evaluate(beta, alpha):
        eta = Xw @ beta + alpha[g]
        nll, s, w = obs_terms(eta, Kk, Ik)
        return float(nll.sum() / n), s, w

    f, s, w = evaluate(beta, alpha)
    hist, status, last_rel = [], None, np.inf
    for it in range(200):
        gb = Xw.T @ s / n; ga = np.bincount(g, weights=s, minlength=G) / n
        gb_std = gb * sdw; ga_grp = ga * n / ngk
        stat = max(np.max(np.abs(gb)), np.max(np.abs(gb_std)), np.max(np.abs(ga_grp)))
        hist.append(dict(it=it, obj=f, g_beta=float(np.max(np.abs(gb))), g_beta_std=float(np.max(np.abs(gb_std))),
                         g_alpha_group=float(np.max(np.abs(ga_grp)))))
        if stat <= GTOL and last_rel <= 1e-12:
            status = 'valid'; break
        # w == 0 is legitimate underflow (censored row with mu >> K: h ~ 0); negative or nonfinite is a failure
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            status = 'nonfinite_or_nonconvex_curvature'; break
        # Newton via Schur complement: H = [[Hbb, Hba],[Hab, diag(d)]] (average scale)
        d = np.bincount(g, weights=w, minlength=G) / n
        if np.any(d <= 0):
            status = 'zero_group_curvature'; break
        Hba = np.stack([np.bincount(g, weights=w * Xw[:, j], minlength=G) for j in range(Xw.shape[1])], 0) / n
        Hbb = (Xw.T * w) @ Xw / n
        S = Hbb - (Hba / d) @ Hba.T
        try:
            db = np.linalg.solve(S, -(gb - (Hba / d) @ ga))
        except np.linalg.LinAlgError:
            status = 'singular_newton'; break
        da = -(ga + Hba.T @ db) / d
        slope = gb @ db + ga @ da
        if not slope < 0:
            db, da, slope = -gb, -ga, -(gb @ gb + ga @ ga)
        step, ok = 1.0, False
        for _ in range(40):
            f2, s2, w2 = evaluate(beta + step * db, alpha + step * da)
            if np.isfinite(f2) and f2 <= f + 1e-4 * step * slope + 1e-15 * abs(f):
                ok = True; break
            step *= 0.5
        if not ok:
            status = 'line_search_exhausted'; break
        last_rel = abs(f - f2) / max(1.0, abs(f))
        beta, alpha, f, s, w = beta + step * db, alpha + step * da, f2, s2, w2
    else:
        status = 'nonconvergence'
    rec.update(status=status, nit=len(hist), objective=f, final=hist[-1], last_rel_change=last_rel,
               beta=dict(zip([str(j) for j in wcols], beta.tolist())))
    if status != 'valid':
        return ret(rec, None)
    bfull[wcols] = beta; afull[keepg] = alpha; afull[neg] = -np.inf; afull[pos] = np.inf
    # recovered daily demand on all rows: kept rows use fitted offsets; all-zero groups uncensored -> K;
    # all-censored groups have no finite offset -> recovery unavailable (NaN), excluded and counted downstream
    D = Kf.copy(); D[(pos)[sd_inv]] = np.nan
    eta = Xw @ beta + alpha[g]
    D[keep] = _dhat_eta(eta, Kk, Ik)
    return ret(rec, D)


def _dhat_eta(eta, K, I):
    import mfit_corrected as mf
    P = mf.Problem(np.zeros((len(K), 1)), K, I, eta)
    D = K.copy()
    if I.any():
        h, mu, _ = mf.m_hazard(np.zeros(1), P); D[I] = mu + h
    return D


def within_B(Y, pm, sdi, G, rows=None):
    """Legacy store-day within estimator (03_main) on hourly-apportioned values; rows mask optional."""
    if rows is not None:
        Y, pm, sdi = Y[rows], pm[rows], sdi[rows]
    n_p = np.bincount(sdi, weights=pm, minlength=G).astype(float); n_np = np.bincount(sdi, weights=1 - pm, minlength=G).astype(float)
    sp = np.bincount(sdi, weights=np.where(pm[:, None], Y, 0).sum(axis=1), minlength=G)
    snp = np.bincount(sdi, weights=np.where(pm[:, None], 0, Y).sum(axis=1), minlength=G)
    n_all = n_p + n_np; p_bar = n_p / np.maximum(n_all, 1); ybar = (sp + snp) / np.maximum(n_all, 1)
    num = (1 - p_bar) * (sp - n_p * ybar) - p_bar * (snp - n_np * ybar)
    den = (1 - p_bar) ** 2 * n_p + p_bar ** 2 * n_np
    both = (n_p > 0) & (n_np > 0)
    return float(np.sum(num[both]) / max(np.sum(den[both]), 1e-9))


# ---------------------------------------------------------------- cell driver
def run_batch(cells, topF, test=False):
    tobit, m05, m3 = legacy()
    groups = m05.build_groups(set(cells)); done = []
    for (city, cat) in cells:
        t0 = time.time()
        d = m3.build_cell(groups.pop((city, cat)))
        X, K, C, k_h, ST = d['X'], d['K'], d['C'], d['k_h'], d['ST']
        I = C.astype(bool); pm = d['promo'] == 1; Yf = k_h.astype(np.float64)
        o = json.load(open(os.path.join(H, 'out', 'p3emp', f'{city}_{cat}.json')))
        bL = np.array(o['legacy_M']['b']); cM = o['corrected_M']
        rec = dict(city=city, cat=cat, n=int(len(K)), n_uncensored=int((C == 0).sum()), corrected_M_status=cM['status'])
        with np.errstate(all='ignore'):
            DL, _ = tobit.latent_day(bL, X, K, C)
        arrs = {}
        naive_h, naiveA = profiles(Yf, pm); arrs['naive_h'] = naive_h; rec['naiveA'] = naiveA
        H0 = m3.apportion(k_h, ST, C, K, DL)
        arrs['rec_h_H0'], rec['recA_H0'] = profiles(H0, pm)
        rec['H0_profile_rows'] = int(len(k_h[~C]))
        HA, dA = apportion_fixed(k_h, ST, C, K, DL); rec.update({f'HA_{k}': v for k, v in dA.items()})
        if HA is not None:
            arrs['rec_h_HA'], rec['recA_HA'] = profiles(HA, pm); rec.update({f'HA_chk_{k}': v for k, v in check_allocation(HA, k_h, ST, C, K, DL).items()})
            rec['daily_total_maxdiff_H0_HA'] = float(np.max(np.abs(H0.sum(1) - HA.sum(1))))
        if cM['status'] == 'valid':
            DB = corrected_dhat(np.array(cM['x']), X, K.astype(float), I)
            HB, dB = apportion_fixed(k_h, ST, C, K, DB); rec.update({f'HB_{k}': v for k, v in dB.items()})
            if HB is not None:
                arrs['rec_h_HB'], rec['recA_HB'] = profiles(HB, pm); rec.update({f'HB_chk_{k}': v for k, v in check_allocation(HB, k_h, ST, C, K, DB).items()})
            rec['daily_recovered_mean_legacy'] = float(DL.mean()); rec['daily_recovered_mean_corrected'] = float(DB.mean())
        else:
            rec['HB_unavailable_reason'] = f"corrected M {cM['status']}"
        if (city, cat) in topF:
            G = len(d['sd_uniq']); sdi = d['sd_inv']
            rec['F_naiveB'] = within_B(Yf, pm, sdi, G)
            # legacy replay of the original F path
            t = time.time()
            with np.errstate(all='ignore'):
                fF = tobit.fit_fe(X, K, C, sdi, max_iter=25)
                DFL = tobit.latent_day(fF['b'], X, K, C, offset=fF['alpha'][sdi])[0]
            HFL = m3.apportion(k_h, ST, C, K, DFL)
            rec['F_legacy'] = dict(b=fF['b'].tolist(), nit=fF['nit'], converged=bool(fF['converged']), n_groups=fF['n_groups'],
                                   recF=within_B(HFL, pm, sdi, G), seconds=time.time() - t)
            t = time.time()
            fr, DF = fit_f(X, K, C, sdi, np.array(cM['x']) if cM['status'] == 'valid' else bL)
            fr['seconds'] = time.time() - t
            if DF is not None:
                ok = np.isfinite(DF)
                HF, _ = apportion_fixed(k_h, ST, C, K, np.where(ok, DF, K))
                fr['recF'] = within_B(HF, pm, sdi, G, rows=ok) if HF is not None else None
                fr['naiveB_same_rows'] = within_B(Yf, pm, sdi, G, rows=ok)
                fr['rows_without_finite_recovery'] = int((~ok).sum())
            rec['F_corrected'] = fr
        rec['seconds'] = time.time() - t0
        os.makedirs(os.path.join(H, 'out', 'p4'), exist_ok=True)
        tag = f'{city}_{cat}' + ('_test' if test else '')
        np.savez(os.path.join(H, 'out', 'p4', f'.{tag}.tmp.npz'), **arrs)
        os.replace(os.path.join(H, 'out', 'p4', f'.{tag}.tmp.npz'), os.path.join(H, 'out', 'p4', f'{tag}_hourly.npz'))
        json.dump(rec, open(os.path.join(H, 'out', 'p4', f'{tag}.json'), 'w'), default=lambda v: v.item() if hasattr(v, 'item') else str(v))
        done.append((city, cat, rec['n'], round(rec['seconds'], 1), rec.get('F_corrected', {}).get('status')))
    return done


def unit_checks():
    """Fixed examples (instruction sec. 4/5)."""
    import mfit_corrected as mf
    k_h = np.array([[1, 2, 0, 0], [0, 1, 1, 0], [2, 0, 0, 0], [0, 0, 0, 0]]); C = np.array([0, 1, 0, 1], np.int8)
    ST = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], bool); K = k_h.sum(1)
    D = np.array([3.0, 5.0, 2.0, 4.0])
    out, dg = apportion_fixed(k_h, ST, C, K, D)
    assert np.allclose(k_h[C == 0].mean(0), [1.5, 1.0, 0, 0])
    assert list(~C) == [-1, -2, -1, -2] and k_h[~C].shape == (4, 4)   # legacy bug: ~int gives row indices, not a mask
    chk = check_allocation(out, k_h, ST, C, K, D)
    assert chk['nonneg'] and chk['only_flagged_hours'] and chk['uncensored_unchanged'] and chk['max_daily_total_err'] < 1e-12
    assert dg['n_zero_profile_fallback'] == 1 and np.allclose(out[1], [0, 1, 2.5, 1.5])   # profile 0 on flagged hours 2,3 -> equal split of L=3
    assert np.allclose(out[3], [2.4, 1.6, 0, 0])                                       # L=4 by profile over flagged hours
    assert apportion_fixed(k_h[[1, 3]], ST[[1, 3]], C[[1, 3]], K[[1, 3]], D[[1, 3]])[0] is None
    _, s, w = obs_terms(np.array([0.0]), np.array([0.0]), np.array([True]))
    assert abs(-w[0] - (-0.3386968873)) < 1e-9, w
    return 'unit checks pass'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cells'); ap.add_argument('--all', action='store_true')
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--test', action='store_true'); a = ap.parse_args()
    import pandas as pd
    print(unit_checks(), flush=True)
    topF = {(int(r.city), int(r.cat)) for r in pd.read_csv(os.path.join(LIVE, 'main', 'main_topF.csv')).itertuples()}
    if a.cells:
        cells = [tuple(map(int, c.split(':'))) for c in a.cells.split(',')]
    else:
        e = pd.read_csv(os.path.join(LIVE, 'main', 'elast_cells.csv')); e = e[e.status == 'ok']
        cells = sorted(zip(e.city.astype(int), e['cat'].astype(int)))
    cells = [c for c in cells if a.test or not os.path.exists(os.path.join(H, 'out', 'p4', f'{c[0]}_{c[1]}.json'))]
    print(f'{len(cells)} cells', flush=True)
    if a.workers == 1:
        print(run_batch(cells, topF, a.test), flush=True); return
    e = pd.read_csv(os.path.join(LIVE, 'main', 'elast_cells.csv')).set_index(['city', 'cat'])['n_rows']
    F = [c for c in cells if c in topF]; R = sorted([c for c in cells if c not in topF], key=lambda c: -e.get(c, 0))
    batches = [[c] for c in F] + [R[i::16] for i in range(16)]   # each F cell alone (heavy), rest in 16 batches
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        for f in as_completed([ex.submit(run_batch, b, topF) for b in batches]):
            print(f.result(), time.strftime('%H:%M:%S'), flush=True)


if __name__ == '__main__':
    main()
