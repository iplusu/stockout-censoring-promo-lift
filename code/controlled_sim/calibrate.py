"""S3: calibrate the 24 capacity log-multipliers a_{s,eta,q} (protocol §2).

p(a) = sum_{k=0}^{kmax} int_{b_{k-1}}^{b_k} phi(z) * Phibar((c_k - eta z)/sqrt(1-eta^2)) dz
  b_k = Phi^{-1}(F_C(k)),  c_k = Phi^{-1}(F_D(k)),  C ~ Pois(kappa = lambda_s * exp(a)).
Derivation: W = eta Z_D + sqrt(1-eta^2) Z_C ~ N(0,1); C = k <=> W in (b_{k-1}, b_k];
D > k <=> Z_D > c_k; Z_D | W=z ~ N(eta z, 1-eta^2).
Run:  python calibrate.py   -> calibration.csv (frozen before any evaluation fit)
"""
import json, math, sys, warnings
import numpy as np
from scipy import integrate, optimize, stats

cfg = json.load(open('config.json'))
CAL = cfg['calibration']
STRATA = cfg['strata']
B0, BP, BD = math.log(20.0), cfg['beta_true']['bP'], cfg['beta_true']['bd']
norm, pois = stats.norm, stats.poisson


def lam(s):
    return math.exp(B0 + BP * s['P'] + BD * s['d'])


def kmax_for(kappa):
    # smallest kmax with Pr(C > kmax) < 1e-10
    k = int(pois.isf(1e-10, kappa))
    while pois.sf(k, kappa) >= 1e-10:
        k += 1
    while k > 0 and pois.sf(k - 1, kappa) < 1e-10:
        k -= 1
    return k


def p_of_a(a, lam_s, eta):
    kappa = lam_s * math.exp(a)
    km = kmax_for(kappa)
    ks = np.arange(km + 1)
    # stable boundaries: use inverse-survival on the upper side
    b = norm.isf(pois.sf(ks, kappa))          # Phi^{-1}(F_C(k)) = isf(Pr(C>k))
    c = norm.isf(pois.sf(ks, lam_s))          # Phi^{-1}(F_D(k)) = isf(Pr(D>k))
    s = math.sqrt(1.0 - eta * eta)
    epsabs = 1e-10 / (km + 1)   # protocol §2 item 2
    tot, err_tot, nwarn = 0.0, 0.0, 0
    lo = -np.inf
    for k in range(km + 1):
        hi = b[k]
        if not np.isfinite(c[k]) and c[k] > 0:   # Pr(D>k)=0 in double -> term 0
            lo = hi
            continue
        if eta == 0.0:
            f = lambda z, ck=c[k]: norm.pdf(z) * norm.sf(ck)
        else:
            f = lambda z, ck=c[k]: norm.pdf(z) * norm.sf((ck - eta * z) / s)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            v, e = integrate.quad(f, lo, hi, epsabs=epsabs, epsrel=CAL['quadrature']['epsrel'],
                                  limit=CAL['quadrature']['limit'])
            nwarn += len(w)
        tot += v
        err_tot += e
        lo = hi
    return tot, err_tot, nwarn, km


def eta0_reference(a, lam_s):
    kappa = lam_s * math.exp(a)
    km = kmax_for(kappa)
    ks = np.arange(km + 1)
    return float(np.sum(pois.pmf(ks, kappa) * pois.sf(ks, lam_s)))


def calibrate_cell(s, eta, q):
    lam_s = lam(s)
    g = lambda a: p_of_a(a, lam_s, eta)[0] - q
    lo, hi = CAL['root']['bracket']
    nexp = 0
    glo, ghi = g(lo), g(hi)
    while glo * ghi > 0 and nexp < CAL['root']['expand_max']:
        lo -= CAL['root']['expand_step']; hi += CAL['root']['expand_step']; nexp += 1
        glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:
        return dict(status='NO_BRACKET', n_expand=nexp)
    a, r = optimize.brentq(g, lo, hi, xtol=CAL['root']['xtol'], rtol=CAL['root']['rtol'],
                           maxiter=CAL['root']['maxiter'], full_output=True, disp=False)
    p, err, nwarn, km = p_of_a(a, lam_s, eta)
    out = dict(stratum=s['name'], P=s['P'], d=s['d'], lambda_s=lam_s, eta=eta, q=q, a=a,
               kappa=lam_s * math.exp(a), p_achieved=p, abs_dev=abs(p - q), quad_err_total=err,
               trunc_bound=pois.sf(km, lam_s * math.exp(a)), kmax=km, quad_warnings=nwarn,
               bracket_lo=lo, bracket_hi=hi, n_expand=nexp, brent_iter=r.iterations,
               brent_converged=r.converged)
    if eta == 0.0:
        out['eta0_ref'] = eta0_reference(a, lam_s)
        out['eta0_diff'] = abs(out['eta0_ref'] - p)
    ok = (r.converged and out['abs_dev'] <= CAL['accept_abs_err'] and err <= CAL['quadrature']['total_abs_err_max']
          and (eta != 0.0 or out['eta0_diff'] <= 1e-9))
    out['status'] = 'OK' if ok else 'FAIL'
    return out


if __name__ == '__main__':
    import pandas as pd
    rows = []
    for s in STRATA:
        for eta in cfg['conditions']['eta']:
            for q in cfg['conditions']['q']:
                rows.append(calibrate_cell(s, eta, q))
                rr = rows[-1]
                print(f"{rr.get('stratum', s['name']):15s} eta={eta} q={q}: a={rr.get('a', float('nan')):+.10f} "
                      f"p={rr.get('p_achieved', float('nan')):.12f} dev={rr.get('abs_dev', float('nan')):.2e} "
                      f"qerr={rr.get('quad_err_total', float('nan')):.1e} {rr['status']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('calibration.csv', index=False, float_format='%.17g')
    nfail = int((df['status'] != 'OK').sum())
    print(f'cells={len(df)} fail={nfail} max_abs_dev={df["abs_dev"].max():.3e} '
          f'max_quad_err={df["quad_err_total"].max():.3e} warnings={int(df["quad_warnings"].sum())}')
    sys.exit(1 if nfail else 0)
