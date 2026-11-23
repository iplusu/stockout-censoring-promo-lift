"""P2 numerical validation of mfit_corrected (instructions §2-3 (a)-(d) + recovery + legacy contrast).
  python validate_corrected.py   -> numeric_validation.log ; exit 1 on any failure."""
import math, sys, os
from decimal import Decimal, getcontext
import numpy as np
from scipy import special
import mfit_corrected as mf

getcontext().prec = 80
LOG, FAIL = [], []


def check(name, ok, detail=''):
    LOG.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); print(LOG[-1], flush=True)
    if not ok: FAIL.append(name)


def dec_pois_sf(k, mu):
    mu = Decimal(mu); t = (-mu).exp()
    for j in range(1, k + 2): t = t * mu / j
    s, j = Decimal(0), k + 1
    while s == 0 or t > s * Decimal(10) ** -60:
        s += t; j += 1; t = t * mu / j
    return s


def dec_nb_sf(k, mu, th):
    mu, th = Decimal(mu), Decimal(th); p = th / (th + mu); q = mu / (th + mu)
    f = (th * p.ln()).exp()                                    # f(0) = p^th
    for m in range(0, k + 1): f = f * (m + th) / (m + 1) * q   # -> f(k+1)
    s, m = Decimal(0), k + 1
    while s == 0 or f > s * Decimal(10) ** -60:
        s += f; f = f * (m + th) / (m + 1) * q; m += 1
    return s


def se_fd(fun, x, P, h=1e-4):
    k = len(x); H = np.zeros((k, k)); E = np.eye(k) * h
    for i in range(k):
        for j in range(k):
            H[i, j] = (fun(x + E[i] + E[j], P) - fun(x + E[i] - E[j], P) - fun(x - E[i] + E[j], P) + fun(x - E[i] - E[j], P)) / (4 * h * h)
    return np.sqrt(np.diag(np.linalg.inv(H * P.n)))


def fd(fun, x, P, h):
    return np.array([(fun(x + e, P) - fun(x - e, P)) / (2 * h) for e in np.eye(len(x)) * h])


def main():
    rng = np.random.default_rng(20261122)
    n, p = 3000, 6
    X = np.column_stack([np.ones(n), rng.integers(0, 2, n), rng.uniform(0, .4, n), rng.normal(0, 1, (n, 3))])
    off = rng.normal(0, .1, n)
    b_true = np.array([1.6, 0.25, 1.0, 0.1, -0.15, 0.05])
    mu = np.exp(X @ b_true + off)
    # ---- (a) no censoring => M equals ordinary Poisson
    D = rng.poisson(mu).astype(float); Z = np.zeros(n, bool)
    P0 = mf.Problem(X, D, Z, off); b = b_true + 0.03
    pois = -np.mean(D * P0.eta(b) - np.exp(P0.eta(b)) - special.gammaln(D + 1))
    check('(a) I=0: M objective == Poisson', abs(mf.m_nll(b, P0) - pois) < 1e-14)
    check('(a) I=0: M gradient == Poisson', np.max(np.abs(mf.m_grad(b, P0) + P0.X.T @ (D - np.exp(P0.eta(b))) / n)) < 1e-14)
    # ---- (b) strict tails vs Decimal(80) direct summation, incl. far tails
    w = 0.0
    for k, m in [(0, .5), (3, 2.), (25, 20.), (5, 30.), (150, 1.), (200, 1.), (1000, 1.)]:
        w = max(w, abs(float(mf.pois_logsf(k, m)) - float(dec_pois_sf(k, m).ln())))
    check('(b) Poisson strict log-SF vs Decimal (k up to 1000, mu=1)', w < 1e-11, f'max abs={w:.2e}')
    w = 0.0
    for k, m, th in [(0, .5, 2.), (5, 3., .7), (40, 10., 3.), (200, 1., 2.), (400, 2., 5.), (30, 50., .5), (12, 4., 1e4)]:
        w = max(w, abs(float(mf.nb_logsf(k, m, th)) - float(dec_nb_sf(k, m, th).ln())))
    check('(b) NB2 strict log-SF vs Decimal (incl. far tail, theta 0.5..1e4)', w < 1e-10, f'max abs={w:.2e}')
    kk = np.arange(0, 60, dtype=float); mm = np.full(60, 7.3)
    dnb = np.max(np.abs(mf._nb_logpmf(kk, mm, 1e14) - mf.pois_logpmf(kk, mm)))
    import scipy.stats as sst
    dsc = np.max(np.abs(mf._nb_logpmf(kk, mm, 2.5) - sst.nbinom.logpmf(kk, 2.5, 2.5 / (2.5 + 7.3))))
    check('(b) NB2 log-pmf: stable form == scipy at theta=2.5, -> Poisson at theta=1e14 (theory gap ~(k^2+mu^2)/theta ~ 4e-11)', dsc < 1e-12 and dnb < 1e-9, f'vs scipy {dsc:.1e}; vs Poisson {dnb:.1e}')
    leg = math.log(max(special.gammainc(201.0, 1.0), 1e-300))
    check('(b) legacy 1e-300 floor differs from strict tail (documented defect)',
          abs(leg - float(mf.pois_logsf(200, 1.0))) > 100, f'legacy={leg:.3f} strict={float(mf.pois_logsf(200, 1.0)):.3f}')
    # ---- (c) derivatives on a censored general-X fixture with offset
    C = rng.poisson(0.9 * mu); I = D > C; K = np.minimum(D, C).astype(float)
    P = mf.Problem(X, K, I, off)
    for tag, bb in [('near truth', b_true + .05), ('far', np.array([0.5, 1.0, -1.0, .5, .5, -.5]))]:
        g, gf = mf.m_grad(bb, P), fd(mf.m_nll, bb, P, 1e-6)
        check(f'(c) M gradient vs centered FD ({tag})', np.max(np.abs(g - gf)) / (1 + np.max(np.abs(g))) < 1e-6,
              f'{np.max(np.abs(g - gf)):.1e}')
        H = mf.m_hess(bb, P); Hf = np.column_stack([(mf.m_grad(bb + e, P) - mf.m_grad(bb - e, P)) / 2e-5 for e in np.eye(p) * 1e-5])
        check(f'(c) M Hessian vs gradient differences ({tag})', np.max(np.abs(H - Hf)) / (1 + np.max(np.abs(H))) < 1e-5,
              f'{np.max(np.abs(H - Hf)):.1e}')
    # K=0, lambda=1 log-likelihood curvature (instructions value -0.3386968873)
    P1 = mf.Problem(np.ones((1, 1)), np.array([0.0]), np.array([True]))
    curv = -float(mf.m_hess(np.array([0.0]), P1)[0, 0])
    S0 = dec_pois_sf(0, 1.0); h = Decimal(1) * (Decimal(-1).exp()) / S0; ref = -h * (1 + h - 0 - 1)
    check('(c) censored curvature K=0, lambda=1 = -0.3386968873 (legacy F gave +0.3387)', abs(curv - float(ref)) < 1e-12,
          f'corrected={curv:.10f} ref={float(ref):.10f}')
    # far-tail curvature K=1000 mu=1
    P2 = mf.Problem(np.ones((1, 1)), np.array([1000.0]), np.array([True]))
    S = dec_pois_sf(1000, 1.0); f1000 = (Decimal(-1).exp())
    for j in range(1, 1001): f1000 /= j
    hh = f1000 / S; cref = -hh * (1 + hh - 1001)
    c2 = -float(mf.m_hess(np.array([0.0]), P2)[0, 0])
    check('(c) far-tail curvature K=1000, mu=1 vs Decimal', abs(c2 - float(cref)) / abs(float(cref)) < 1e-9, f'rel={abs(c2 - float(cref)) / abs(float(cref)):.1e}')
    th0 = np.append(b_true + .05, math.log(3.0))
    g, gf = mf.nb_grad(th0, P), fd(mf.nb_nll, th0, P, 1e-6)
    check('(c) NB2 gradient vs centered FD (beta analytic, log-theta FD)', np.max(np.abs(g - gf)) / (1 + np.max(np.abs(g))) < 1e-6, f'{np.max(np.abs(g - gf)):.1e}')
    z0 = np.append(b_true + .05, -1.5)
    g, gf = mf.zip_grad(z0, P), fd(mf.zip_nll, z0, P, 1e-6)
    check('(c) MIX gradient vs centered FD', np.max(np.abs(g - gf)) / (1 + np.max(np.abs(g))) < 1e-6, f'{np.max(np.abs(g - gf)):.1e}')
    # ---- (d) boundary / structural cases
    check('(d) all censored -> structural_nonexistence', mf.screen(X, np.full(n, 3.), np.ones(n, bool))[0] == 'nonexistent')
    check('(d) all-zero uncensored -> structural_nonexistence', mf.screen(X, np.zeros(n), np.zeros(n, bool))[0] == 'nonexistent')
    Xr = np.column_stack([X, X[:, 1]])
    check('(d) rank-deficient design -> unidentified (separate from recession)', mf.screen(Xr, K, I)[0] == 'unidentified')
    check('(d) ordinary censored data -> pass', mf.screen(X, K, I)[0] == 'pass')
    Kp = K.copy(); Kp[X[:, 1] == 1] = 0; Ip = I.copy(); Ip[X[:, 1] == 1] = False
    st, J = mf.screen(X, Kp, Ip)
    fr, _ = mf.fit('M', X, Kp, Ip, [b_true])
    check('(d) promo rows all zero & uncensored -> nonexistent, fit not attempted', st == 'nonexistent' and fr['status'] == 'structural_nonexistence', f'J*={J:.3f}')
    # ---- recovery on known DGPs (n=3000)
    fm, _ = mf.fit('M', X, K, I, [np.zeros(p), b_true + .1], offset=off)
    check('recovery M (censored Poisson DGP)', fm['valid'] and np.max(np.abs(fm['x'] - b_true)) < 0.1,
          f"max|b-b*|={np.max(np.abs(fm['x'] - b_true)):.3f} g_orig={fm['g_orig']:.1e} g_std={fm['g_std']:.1e} info={fm.get('info')}")
    th = 2.0; Dn = rng.negative_binomial(th, th / (th + mu)); Cn = rng.poisson(0.9 * mu); In = Dn > Cn; Kn = np.minimum(Dn, Cn).astype(float)
    fn, _ = mf.fit('NB2', X, Kn, In, [np.append(b_true + .1, math.log(5.0))], offset=off)
    zn = (fn['x'] - np.append(b_true, math.log(th))) / se_fd(mf.nb_nll, fn['x'], mf.Problem(X, Kn, In, off))
    check('recovery NB2 (censored NB DGP, theta=2): valid and all |z| < 3.5 (SE from FD Hessian)', fn['valid'] and np.max(np.abs(zn)) < 3.5,
          f"z={np.round(zn, 2).tolist()} theta={math.exp(fn['x'][-1]):.2f} g_orig={fn['g_orig']:.1e}")
    pi = 0.15; Dz = np.where(rng.random(n) < pi, 0, rng.poisson(mu)); Iz = Dz > C; Kz = np.minimum(Dz, C).astype(float)
    fz, _ = mf.fit('MIX', X, Kz, Iz, [np.append(b_true + .1, math.log(.1 / .9))], offset=off)
    zz = (fz['x'] - np.append(b_true, math.log(pi / (1 - pi)))) / se_fd(mf.zip_nll, fz['x'], mf.Problem(X, Kz, Iz, off))
    check('recovery MIX (censored ZIP DGP, pi=0.15): valid and all |z| < 3.5', fz['valid'] and np.max(np.abs(zz)) < 3.5,
          f"z={np.round(zz, 2).tolist()} pi={1 / (1 + math.exp(-fz['x'][-1])):.3f} g_orig={fz['g_orig']:.1e}")
    # regression: on data without extreme tails the corrected and legacy NB2/MIX optima coincide
    sys.path.insert(0, 'legacy/main'); sys.path.insert(0, 'legacy/pilot')
    import importlib; m07 = importlib.import_module('07_pending_3_altmodels')
    for nm, Kk, Ii, lf, aux in [('NB2', Kn, In, m07.fit_nb2, math.log(5.0)), ('MIX', Kz, Iz, m07.fit_zip, math.log(.1 / .9))]:
        fc, _ = mf.fit(nm, X, Kk, Ii, [np.append(b_true + .1, aux)]); fl = lf(X, Kk, Ii.astype(int), b_true + .1)
        dmax = float(np.max(np.abs(fc['x'][:p] - fl['b'])))
        check(f'legacy agreement {nm} on non-extreme data (no offset): |b_corr - b_legacy| < 1e-5', dmax < 1e-5, f'{dmax:.1e}')
    # ---- multi-start agreement
    fm2, recs = mf.fit('M', X, K, I, [np.zeros(p), b_true + .1, b_true - .2], offset=off)
    objs = [r['objective'] for r in recs if r['converged']]
    check('multi-start: all converged starts reach the same objective', len(objs) == 3 and max(objs) - min(objs) < 1e-10, f'spread={max(objs) - min(objs):.1e}')
    open('numeric_validation.log', 'w').write('\n'.join(LOG) + f"\n\nRESULT: {'ALL PASS' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}\n")
    print('ALL PASS' if not FAIL else f'FAIL {FAIL}'); sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
