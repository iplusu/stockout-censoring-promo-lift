"""Final pending analyses for JAMS-50K main.tex (Prism fixed-spec, pre-registered decision rules).

P-A  tex S3.9  — recovery-model alternatives vs M (same 254 cells, same X, same log link):
    NB2 (theta per-cell MLE, reuse 07_pending_3 fit), ZIP Z=1 (reuse) + ZIP Z=[1,promo] (new),
    mixture-censoring label noise (Poisson base): pi fixed 0.95/0.85/0.75 + pi-hat per cell,
    + NB2 x mixture variants (if runtime allows). Per alternative x {depth, promo}:
    median |bG - b_alt|/|b_alt| (bG from decomp_cells.csv, no refit), frac >= 20%, sign-flip rate,
    Rademacher wild-cluster 95% CI (S3.8 machinery, 2000 reps, seed 42), AIC/BIC + LRT-vs-M
    preference (converged cells). Non-converged cells listed, not dropped.
    Decision rule (tex S3.9): survives iff median gap >= 20% AND frac >= 20% >= 50% AND wild CI
    lower bound > 20%; if it fails under a data-preferred alternative the claim is sensitive.
P-B  tex S4.3.6 — Study 1 (Yelp CHI/UDIS), conventions = scripts/04_main_analysis.py:
    (1) activity-matched comparison per size band x seq: CEM on log median n_rec (0.25 bins)
        x window new-review count (dn_rec = n_rec(w+1)-n_rec(w), pooled quartile bins),
        1:1 with-replacement within stratum (seed 42); matched median |dR| event vs control,
        matched rank-biserial r, unmatched event windows.
    (2) 95% percentile CI for large-band event/pseudo median |dR| ratio (CHI 0.92 / UDIS 2.18)
        by business-level resampling (events nested in businesses), 2000 reps, raw + matched.
    Decision rule (tex): "not detectable in large" withdrawn if CI excludes 1; inconclusive if
    includes 1 but wide.
P-C  tex S3.8 — wild-cluster refinement: Rademacher (+-1) weights on the 18 cities (replaces
    resampling-with-replacement), 2000 reps, seed 42, for: P1a median|bias| + frac>=20,
    P1b depth/promo median + frac, pure-censoring depth/promo median + frac. City contributions
    explicit per statistic: frac = centered per-city exceedance sums (exact additive statistic);
    median = Bahadur linearization at the median (per-city sums of (1{|v|<=m}-1/2)/fhat(m),
    fhat = local empirical density, delta = 0.1*IQR). CI = 2.5/97.5 percentiles of T(w).
    Compared against existing city-block CIs (pending_joint_ci.csv, 2000 reps).

Outputs: pending_final_pA_cells.csv (per-cell fits, checkpointed), pending_final_pA_summary.csv,
         pending_final_pA_nonconv.csv, pending_final_pB_matching.csv, pending_final_pB_ratio_ci.csv,
         pending_final_wildci.csv, pending_final_report.md (section-flushed, interruption-tolerant).
Selftest (first!): --selftest -> P-A 10 cells (fits reproduce pending_altmodels.csv + numeric grad
checks + timing), P-B 20-business subsample, P-C 10 cities n_boot=200.
seed=42 everywhere. draft/ and latex/ are read-only inputs (spec + existing CIs).
"""
import importlib.util
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gammaln, gammainc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import tobit  # noqa: E402

_spec = importlib.util.spec_from_file_location('m05', os.path.join(HERE, '05_decomp.py'))
m05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m05)
_spec = importlib.util.spec_from_file_location('m07p3', os.path.join(HERE, '07_pending_3_altmodels.py'))
m07p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m07p3)

YB = '/home/ubuntu/papers/yelp-reputation/data/'
MAIN_CSV = os.path.join(HERE, 'main_cells.csv')
DECOMP_CSV = os.path.join(HERE, 'decomp_cells.csv')
ALT_CSV = os.path.join(HERE, 'pending_altmodels.csv')
JOINT_CSV = os.path.join(HERE, 'pending_joint_ci.csv')

PA_CELLS = os.path.join(HERE, 'pending_final_pA_cells.csv')
PA_SUM = os.path.join(HERE, 'pending_final_pA_summary.csv')
PA_NONCONV = os.path.join(HERE, 'pending_final_pA_nonconv.csv')
PB_MATCH = os.path.join(HERE, 'pending_final_pB_matching.csv')
PB_RATIO = os.path.join(HERE, 'pending_final_pB_ratio_ci.csv')
PC_WILD = os.path.join(HERE, 'pending_final_wildci.csv')
REPORT = os.path.join(HERE, 'pending_final_report.md')
SEED = 42
N_BOOT = 2000
_GROUPS = None  # set in run_pA before fork (workers inherit)
_BG = None
NPROMO = 1  # X col index: 0=intercept, 1=promo, 2=depth (03_main.XNAMES)
NDEPTH = 2

PA_COLS = ['city', 'cat', 'n_rows',
           'm_conv', 'm_nit', 'm_ll', 'm_bp', 'm_bd',
           'n2_conv', 'n2_nit', 'n2_ll', 'n2_th', 'n2_bp', 'n2_bd',
           'z1_conv', 'z1_nit', 'z1_ll', 'z1_pi', 'z1_bp', 'z1_bd',
           'z2_conv', 'z2_nit', 'z2_ll', 'z2_g0', 'z2_g1', 'z2_bp', 'z2_bd',
           'm95_conv', 'm95_ll', 'm95_bp', 'm95_bd',
           'm85_conv', 'm85_ll', 'm85_bp', 'm85_bd',
           'm75_conv', 'm75_ll', 'm75_bp', 'm75_bd',
           'mp_conv', 'mp_ll', 'mp_pi', 'mp_bp', 'mp_bd',
           'nm95_conv', 'nm95_ll', 'nm95_th', 'nm95_bp', 'nm95_bd',
           'nm85_conv', 'nm85_ll', 'nm85_th', 'nm85_bp', 'nm85_bd',
           'nm75_conv', 'nm75_ll', 'nm75_th', 'nm75_bp', 'nm75_bd',
           'nmp_conv', 'nmp_ll', 'nmp_pi', 'nmp_th', 'nmp_bp', 'nmp_bd',
           'bG_p', 'bG_d', 'pearson_df', 'recA_M', 'recA_n2', 'recA_z1']
ALT_KEYS = [('n2', 'NB2'), ('z1', 'ZIP_Z=1'), ('z2', 'ZIP_Z=[1,promo]'),
            ('m95', 'MIX_pi=0.95'), ('m85', 'MIX_pi=0.85'), ('m75', 'MIX_pi=0.75'), ('mp', 'MIX_pi_hat'),
            ('nm95', 'NB2MIX_pi=0.95'), ('nm85', 'NB2MIX_pi=0.85'), ('nm75', 'NB2MIX_pi=0.75'), ('nmp', 'NB2MIX_pi_hat')]

# ---------------- P-A: new likelihoods (ZIP zero-part with Z=[1,promo]; mixture censoring) ----

def zip2_ll_grad(thet, X, Z, K, C):
    """Censored ZIP with zero-part pi = logit^{-1}(Z gamma); gamma = last Z.shape[1] params."""
    k_z = Z.shape[1]
    b, gam = thet[:-k_z], thet[-k_z:]
    pi = 1.0 / (1.0 + np.exp(-Z @ gam))
    lam = np.clip(np.exp(X @ b), 1e-8, 1e5)
    s = K.astype(float)
    logpmf = -lam + s * np.log(lam) - gammaln(s + 1.0)
    up = gammainc(s + 1.0, lam)                    # P(D > K)
    logsf = np.log(np.clip(up, 1e-300, 1.0))
    unc = C == 0
    z0 = unc & (K == 0)
    upos = unc & (K > 0)
    p0 = pi + (1 - pi) * np.exp(-lam)
    ll = float(np.sum(np.where(z0, np.log(np.clip(p0, 1e-300, 1.0)),
                               np.where(upos, np.log1p(-pi) + logpmf, np.log1p(-pi) + logsf))))
    sf = np.clip(up, 1e-300, 1.0)
    fP = np.exp(logpmf - logsf)
    dlam = np.where(upos, -1.0 + s / lam,
                    np.where(z0, -(1 - pi) * np.exp(-lam) / np.clip(p0, 1e-300, 1.0), fP))
    sc_b = X.T @ (lam * dlam)
    dtpi = np.where(z0, (1 - np.exp(-lam)) / np.clip(p0, 1e-300, 1.0), -1.0 / (1 - pi))
    sc_gam = Z.T @ (pi * (1 - pi) * dtpi)
    return -ll, -np.concatenate([sc_b, sc_gam])


def _lbg(f, th0, bounds, maxiter=100):
    res = optimize.minimize(f, th0, method='L-BFGS-B', jac=True, bounds=bounds,
                            options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8))
    ll, g = f(res.x)
    return dict(ll=float(-ll), nit=int(res.nit), success=bool(res.success),
                converged=bool(res.success and np.isfinite(-ll)), grad_norm=float(np.linalg.norm(g)), x=res.x)


def _is_true(v):
    return v is True or (isinstance(v, str) and v.lower() == 'true')


def mix_poisson_ll_grad(thet, X, K, C, pi_fix=None, h_pi=1e-5):
    """Flagged day: pi*P(D>K) + (1-pi)*f(K;lambda), Poisson base; uncensored: f(K;lambda).
    pi fixed (pi_fix) or estimated (last param, logit). d P(D>K)/dlam = f(K;lam) (Poisson identity)."""
    fit_pi = pi_fix is None
    b = thet[:-1] if fit_pi else thet
    pi = 1.0 / (1.0 + np.exp(-thet[-1])) if fit_pi else pi_fix
    lam = np.clip(np.exp(X @ b), 1e-8, 1e5)
    s = K.astype(float)
    logpmf = -lam + s * np.log(lam) - gammaln(s + 1.0)
    f = np.exp(np.clip(logpmf, -700, 0.0))
    sf = gammainc(s + 1.0, lam)                    # P(D > K)
    unc = C == 0
    h = pi * sf + (1 - pi) * f
    ll = float(np.sum(np.where(unc, logpmf, np.log(np.clip(h, 1e-300, 1.0)))))
    # d logh/d eta: pi*(lam d sf/dlam) + (1-pi)*(lam d f/dlam) = pi*lam*f + (1-pi)*f*(K-lam)
    # (Poisson identities: d P(D>K)/dlam = f(K;lam); lam d f/dlam = f*(K-lam))
    dlam_c = (pi * lam * f + (1 - pi) * f * (s - lam)) / np.clip(h, 1e-300, 1.0)
    sc = np.where(unc, s - lam, dlam_c)
    gb = X.T @ sc
    if fit_pi:
        dlogh_dpi = (sf - f) / np.clip(h, 1e-300, 1.0)
        gp = float(np.sum(np.where(unc, 0.0, pi * (1 - pi) * dlogh_dpi)))
        return -ll, -np.concatenate([gb, [gp]])
    return -ll, -gb.copy()


def nb2mix_ll_grad(thet, X, K, C, pi_fix=None, h_logth=1e-5):
    """Censored NB2 with label-noise mixture on flagged days: pi*sf_NB2(K) + (1-pi)*f_NB2(K)."""
    fit_pi = pi_fix is None
    n_par = 1 + (1 if fit_pi else 0)
    b = thet[:-n_par]
    logth = thet[-n_par]
    pi = (1.0 / (1.0 + np.exp(-thet[-1]))) if fit_pi else pi_fix
    th = np.exp(logth)
    lam = np.clip(np.exp(X @ b), 1e-8, 1e5)
    sf = m07p3.nb2_sf(K, lam, th)                  # P(D > K)
    logpmf = m07p3.nb2_logpmf(K, lam, th)
    f = np.exp(np.clip(logpmf, -700, 0.0))
    unc = C == 0
    h = pi * sf + (1 - pi) * f

    def _ll(bv, thv, piv):
        l2 = np.clip(np.exp(X @ bv), 1e-8, 1e5)
        sf2 = m07p3.nb2_sf(K, l2, thv)
        lp2 = m07p3.nb2_logpmf(K, l2, thv)
        f2 = np.exp(np.clip(lp2, -700, 0.0))
        h2 = piv * sf2 + (1 - piv) * f2
        return float(np.sum(np.where(unc, lp2, np.log(np.clip(h2, 1e-300, 1.0)))))

    ll = _ll(b, th, pi)
    A = m07p3.nb2_A(K, lam, th)                    # E[D*1{D>K}]
    s = K.astype(float)
    # d sf/dlambda identity (same as 07_pending_3 nb2 score, verified there):
    dfsf_dlam = A / lam - (th * sf + A) / (th + lam)
    dlogf_dlam = (s - lam * (s + th) / (th + lam)) / lam
    # d logh / d eta = [pi*(lam d sf/dlam) + (1-pi)*(lam d f/dlam)] / h
    dlam_c = lam * (pi * dfsf_dlam + (1 - pi) * f * dlogf_dlam) / np.clip(h, 1e-300, 1.0)
    sc = np.where(unc, s - lam * (s + th) / (th + lam), dlam_c)
    gb = X.T @ sc
    out = [gb]
    tp, tm = thet.copy(), thet.copy()
    tp[-n_par] += h_logth
    tm[-n_par] -= h_logth
    # _ll returns +ll (unlike nb2_negll): (ll(tp)-ll(tm))/2h = d ll/dlogth, outer '-' gives d(-ll)
    out.append([(_ll(tp[:-n_par], np.exp(tp[-n_par]), 1.0 / (1.0 + np.exp(-tp[-1])) if fit_pi else pi_fix)
                 - _ll(tm[:-n_par], np.exp(tm[-n_par]), 1.0 / (1.0 + np.exp(-tm[-1])) if fit_pi else pi_fix)) / (2 * h_logth)])
    if fit_pi:
        dlogh_dpi = (sf - f) / np.clip(h, 1e-300, 1.0)
        gp = float(np.sum(np.where(unc, 0.0, pi * (1 - pi) * dlogh_dpi)))
        out.append([gp])
    return -ll, -np.concatenate(out)


def _fit(f, th0, bounds, maxiter=100):
    res = _lbg(f, th0, bounds, maxiter)
    return res


# ---------------- P-A: per cell ----------------

def fit_cell(rows, bG_p, bG_d):
    d = m05.m3.build_cell(rows)
    X, K = d['X'], d['K'].astype(float)
    C = d['C']
    promo = (d['promo'] == 1).astype(float)
    Z = np.column_stack([np.ones(len(K)), promo])
    n = int(X.shape[0])
    rec = dict(n_rows=n, bG_p=float(bG_p), bG_d=float(bG_d))

    fM = tobit.fit(X, K, C, b0=tobit.init_poisson_irls(X, K, C))
    rec.update(m_conv=fM['converged'], m_nit=fM['nit'], m_ll=fM['ll'],
               m_bp=fM['b'][NPROMO], m_bd=fM['b'][NDEPTH])
    D_M, lam_M = tobit.latent_day(fM['b'], X, K, C)
    rec['recA_M'] = float(D_M[d['promo'] == 1].mean() - D_M[d['promo'] == 0].mean())
    uc = C == 0
    # Pearson chi2/df on uncensored days (tex S3.9 spec item 1 dispersion diagnostic; df = n_unc - k)
    rec['pearson_df'] = float(np.sum((K[uc] - lam_M[uc]) ** 2 / lam_M[uc]) / max(int(uc.sum()) - X.shape[1], 1))

    # NB2 (reuse 07_pending_3 fit)
    fN = m07p3.fit_nb2(X, K, C, fM['b'], theta0=5.0)
    D_N = m07p3.recov_nb2(fN['b'], fN['theta'], X, K, C)
    rec.update(n2_conv=fN['converged'], n2_nit=fN['nit'], n2_ll=fN['ll'], n2_th=fN['theta'],
               n2_bp=fN['b'][NPROMO], n2_bd=fN['b'][NDEPTH],
               recA_n2=float(D_N[d['promo'] == 1].mean() - D_N[d['promo'] == 0].mean()))

    # ZIP Z=1 (reuse)
    fZ1 = m07p3.fit_zip(X, K, C, fM['b'])
    D_Z1 = m07p3.recov_zip(fZ1['b'], fZ1['pi'], X, K, C)
    rec.update(z1_conv=fZ1['converged'], z1_nit=fZ1['nit'], z1_ll=fZ1['ll'], z1_pi=fZ1['pi'],
               z1_bp=fZ1['b'][NPROMO], z1_bd=fZ1['b'][NDEPTH],
               recA_z1=float(D_Z1[d['promo'] == 1].mean() - D_Z1[d['promo'] == 0].mean()))

    # ZIP Z=[1,promo] (new)
    fZ2 = _fit(lambda t: zip2_ll_grad(t, X, Z, K, C),
               np.concatenate([fM['b'], [0.0, 0.0]]),
               [(None, None)] * X.shape[1] + [(-8.0, 8.0), (-8.0, 8.0)])
    rec.update(z2_conv=fZ2['converged'], z2_nit=fZ2['nit'], z2_ll=fZ2['ll'],
               z2_g0=fZ2['x'][-2], z2_g1=fZ2['x'][-1], z2_bp=fZ2['x'][NPROMO], z2_bd=fZ2['x'][NDEPTH])

    # mixture-censoring, Poisson base
    for tag, pif in (('m95', 0.95), ('m85', 0.85), ('m75', 0.75)):
        fm = _fit(lambda t: mix_poisson_ll_grad(t, X, K, C, pi_fix=pif), fM['b'].copy(),
                  [(None, None)] * X.shape[1])
        rec.update(**{f'{tag}_conv': fm['converged'], f'{tag}_ll': fm['ll'],
                      f'{tag}_bp': fm['x'][NPROMO], f'{tag}_bd': fm['x'][NDEPTH]})
    fmp = _fit(lambda t: mix_poisson_ll_grad(t, X, K, C, pi_fix=None),
               np.concatenate([fM['b'], [np.log(0.95 / 0.05)]]),
               [(None, None)] * X.shape[1] + [(-8.0, 8.0)])
    rec.update(mp_conv=fmp['converged'], mp_ll=fmp['ll'],
               mp_pi=1.0 / (1.0 + np.exp(-fmp['x'][-1])), mp_bp=fmp['x'][NPROMO], mp_bd=fmp['x'][NDEPTH])

    # NB2 x mixture (optional per task; included if runtime allows)
    for tag, pif in (('nm95', 0.95), ('nm85', 0.85), ('nm75', 0.75)):
        fnm = _fit(lambda t: nb2mix_ll_grad(t, X, K, C, pi_fix=pif),
                   np.concatenate([fN['b'], [np.log(fN['theta'])]]),
                   [(None, None)] * (X.shape[1] + 1))
        rec.update(**{f'{tag}_conv': fnm['converged'], f'{tag}_ll': fnm['ll'], f'{tag}_th': float(np.exp(fnm['x'][-1])),
                      f'{tag}_bp': fnm['x'][NPROMO], f'{tag}_bd': fnm['x'][NDEPTH]})
    fnmp = _fit(lambda t: nb2mix_ll_grad(t, X, K, C, pi_fix=None),
                np.concatenate([fN['b'], [np.log(fN['theta'])], [np.log(0.95 / 0.05)]]),
                [(None, None)] * (X.shape[1] + 1) + [(-8.0, 8.0)])
    rec.update(nmp_conv=fnmp['converged'], nmp_ll=fnmp['ll'],
               nmp_pi=1.0 / (1.0 + np.exp(-fnmp['x'][-1])), nmp_th=float(np.exp(fnmp['x'][-2])),
               nmp_bp=fnmp['x'][NPROMO], nmp_bd=fnmp['x'][NDEPTH])
    return rec


def pa_selftest_check(cells_df):
    """10-cell selftest: numeric gradient checks + M/alt fits reproduce pending_altmodels.csv
    within numerical-provenance tolerance. The CSV was generated by an earlier code/BLAS state,
    so fits near the ftol=1e-12 boundary can jitter a few Newton/L-BFGS iterations (observed
    |m_nit| dev up to 8, final pi/theta/recA dev < 1e-3); CSV stores pi/theta rounded to 5 digits.
    Asserts: m_conv flags equal, recA dev < 1e-2, theta dev < 0.05, pi dev < 5e-4, |m_nit| dev <= 10."""
    _grad_checks()
    alt = pd.read_csv(ALT_CSV)
    al = alt.set_index(['city', 'cat'])
    dev = {'recA_M': 0.0, 'recA_n2': 0.0, 'recA_z1': 0.0, 'n2_th': 0.0, 'z1_pi': 0.0, 'm_nit': 0}
    for r in cells_df.itertuples(index=False):
        a = al.loc[(r.city, r.cat)]
        assert bool(r.m_conv) == bool(a.m_converged), f'M conv mismatch {r.city}x{r.cat}'
        for c_new, c_old in (('recA_M', 'recA_M'), ('recA_n2', 'recA_nb2'), ('recA_z1', 'recA_zip')):
            dev[c_new] = max(dev[c_new], abs(float(getattr(r, c_new)) - float(a[c_old])))
        dev['n2_th'] = max(dev['n2_th'], abs(float(r.n2_th) - float(a.nb2_theta)))
        dev['z1_pi'] = max(dev['z1_pi'], abs(float(r.z1_pi) - float(a.zip_pi)))
        dev['m_nit'] = max(dev['m_nit'], abs(int(r.m_nit) - int(a.m_nit)))
    assert dev['recA_M'] < 1e-2 and dev['recA_n2'] < 1e-2 and dev['recA_z1'] < 1e-2, f'recA dev too large: {dev}'
    assert dev['n2_th'] < 0.05 and dev['z1_pi'] < 5e-4, f'theta/pi dev too large: {dev}'
    assert dev['m_nit'] <= 10, f'm_nit jitter too large: {dev}'
    print(f'P-A selftest OK: fits reproduce pending_altmodels.csv within provenance tolerance: '
          f'recA dev M/n2/z1 = {dev["recA_M"]:.1e}/{dev["recA_n2"]:.1e}/{dev["recA_z1"]:.1e}, '
          f'theta dev {dev["n2_th"]:.2e}, pi dev {dev["z1_pi"]:.2e}, |m_nit| dev {dev["m_nit"]}')


def _grad_checks():
    """Numeric checks for the NEW likelihoods (zip2, mix poisson, nb2mix) vs central differences."""
    rng = np.random.RandomState(42)
    Xs = np.column_stack([np.ones(400), (rng.rand(400) < 0.6).astype(float),
                          np.where(rng.rand(400) < 0.6, rng.rand(400) * 0.3, 0.0),
                          rng.normal(size=(400, 11))])
    Ks = rng.randint(0, 14, 400).astype(float)
    Cs = (rng.rand(400) < 0.35).astype(int)
    promo = Xs[:, 1]
    Zs = np.column_stack([np.ones(400), promo])
    eps = 1e-6

    def check(f, th0, tag):
        ll, g = f(th0)
        gn = np.zeros_like(th0)
        for j in range(len(th0)):
            tp, tm = th0.copy(), th0.copy()
            tp[j] += eps
            tm[j] -= eps
            gn[j] = (f(tp)[0] - f(tm)[0]) / (2 * eps)
        assert np.allclose(g, gn, atol=1e-5), f'{tag} grad mismatch: {np.abs(g - gn).max()}'

    check(lambda t: zip2_ll_grad(t, Xs, Zs, Ks, Cs), np.concatenate([rng.normal(0, 0.3, 14), [0.1, 0.2]]), 'zip2')
    check(lambda t: mix_poisson_ll_grad(t, Xs, Ks, Cs, pi_fix=0.85), rng.normal(0, 0.3, 14), 'mix-fixed')
    check(lambda t: mix_poisson_ll_grad(t, Xs, Ks, Cs, pi_fix=None),
          np.concatenate([rng.normal(0, 0.3, 14), [2.0]]), 'mix-pihat')
    check(lambda t: nb2mix_ll_grad(t, Xs, Ks, Cs, pi_fix=0.85),
          np.concatenate([rng.normal(0, 0.3, 14), [1.0]]), 'nb2mix-fixed')
    check(lambda t: nb2mix_ll_grad(t, Xs, Ks, Cs, pi_fix=None),
          np.concatenate([rng.normal(0, 0.3, 14), [1.0], [2.0]]), 'nb2mix-pihat')
    print('P-A numeric grad checks OK (zip2 / mix / nb2mix, fixed+pi_hat)')


# ---------------- wild-cluster CI (shared P-A / P-C) ----------------

def wild_ci(vals, cities, n_boot=N_BOOT, seed=SEED, kind='frac', thresh=0.20):
    """Rademacher wild-cluster CI. vals: per-cell values (abs gaps) aligned with cities list.
    kind='frac': T = mean(1{|v|>=thresh}); city contribution = per-city centered exceedance sums.
    kind='median': T = median(v); Bahadur linearization, city contribution = per-city sums of
                   (1{v<=m}-1/2)/fhat(m), fhat = local empirical density (delta=0.1*IQR).
    T(w) = T + (1/N) * sum_g w_g s_g; CI = 2.5/97.5 percentiles of T(w), w in {+-1}."""
    vals = np.asarray(vals, float)
    cities = np.asarray(cities)
    N = len(vals)
    rng = np.random.RandomState(seed)
    # FIX 2026-11-01: was rng.choice(2,...)-1 = {0,-1} (E[w]=-0.5, wrong); Rademacher is {+/-1}
    W = 2 * rng.choice(2, size=(n_boot, len(np.unique(cities)))) - 1
    if kind == 'frac':
        u = (np.abs(vals) >= thresh).astype(float)
        T = float(u.mean())
        s_c = np.array([(u[cities == g] - T).sum() for g in np.unique(cities)])
    elif kind == 'median':
        m = float(np.median(vals))
        q = np.percentile(vals, [25, 75])
        dlt = 0.1 * max(q[1] - q[0], 1e-9)
        F = lambda x: (vals <= x).mean()
        fhat = max((F(m + dlt) - F(m - dlt)) / (2 * dlt), 1e-12)
        a = (1.0 * (vals <= m) - 0.5) / fhat
        T = m
        s_c = np.array([a[cities == g].sum() for g in np.unique(cities)])
    else:
        raise ValueError(kind)
    Trep = T + W @ s_c / N
    return T, float(np.percentile(Trep, 2.5)), float(np.percentile(Trep, 97.5))


# ---------------- P-A: aggregate ----------------

def pa_aggregate(cells_df):
    cells = cells_df.set_index(['city', 'cat'])
    n = len(cells)
    rows_sum, nonconv = [], []
    for key, name in ALT_KEYS:
        for coef, c_g, c_a in (('depth', 'bG_d', f'{key}_bd'), ('promo', 'bG_p', f'{key}_bp')):
            bA = cells[c_a].astype(float).values
            bG = cells[c_g].astype(float).values
            gap = np.abs(bG - bA) / np.maximum(np.abs(bA), 1e-9)
            med = float(np.median(gap))
            frac = float((gap >= 0.20).mean())
            sf = float((np.sign(bG) != np.sign(bA)).mean())
            _, lo, hi = wild_ci(gap, cells.index.get_level_values(0).values, kind='median')
            # AIC/BIC/LRT vs M (converged cells of both)
            ok = cells[f'{key}_conv'].map(_is_true) & cells['m_conv'].map(_is_true)
            ll_a = cells.loc[ok, f'{key}_ll'].astype(float).values
            ll_m = cells.loc[ok, 'm_ll'].astype(float).values
            n_conv = int(ok.sum())
            k_a = 15 if key in ('n2', 'z1', 'mp', 'nm95', 'nm85', 'nm75') else (16 if key in ('z2', 'nmp') else 14)
            k_m = 14
            nn = len(ll_a)
            aic_d = 2 * (k_a - k_m) - 2 * (ll_a - ll_m)
            bic_d = (k_a - k_m) * np.log(nn) - 2 * (ll_a - ll_m)
            rows_sum.append(dict(alt=name, coef=coef, median_gap=med, frac_ge20=frac, signflip=sf,
                                 wild_lo=lo, wild_hi=hi, n_conv=n_conv,
                                 aic_pref_frac=float((aic_d < 0).mean()), aic_med_diff=float(np.median(aic_d)),
                                 bic_pref_frac=float((bic_d < 0).mean()), lrt_pref_frac=float((ll_a > ll_m).mean()),
                                 survive=bool(med >= 0.20 and frac >= 0.50 and lo > 0.20)))
            for c in cells.index[~cells[f'{key}_conv'].map(_is_true)]:
                nonconv.append(dict(alt=name, city=c[0], cat=c[1]))
    # data-preferred alternative: highest AIC preference among converged cells
    pref = {name: max(r['aic_pref_frac'] for r in rows_sum if r['alt'] == name)
            for name in dict(ALT_KEYS).values()}
    best = max(pref, key=pref.get)
    sumdf = pd.DataFrame(rows_sum)
    sumdf['data_preferred'] = sumdf['alt'] == best
    # non-converged list
    ndf = pd.DataFrame(nonconv).drop_duplicates() if nonconv else pd.DataFrame(columns=['alt', 'city', 'cat'])
    return sumdf, ndf, best, pref


def pa_summary_md(sumdf, ndf, best, pref, cells_df, timings):
    lines = ['## P-A — tex §3.9: recovery-model alternatives (DONE)',
             '',
             '254 ok cells (decomp_cells.csv keys), same X/log link, bG from decomp_cells.csv (no refit). '
             f'seed=42, deterministic. Non-converged fits kept, listed separately ({len(ndf)} cell-alternative failures). '
             f'Fit wall time: {sum(timings.values()):.0f}s total (this process). Dispersion diagnostic (tex spec item 1): Pearson chi2/df on uncensored days '
             f'median {cells_df.pearson_df.median():.3f} (>1 = overdispersion); NB2 LRT/AIC preference below.',
             '',
             '| alternative | coef | median gap | frac>=20% | sign-flip | wild 95% CI | AIC pref | BIC pref | LRT pref | survive |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in sumdf.itertuples(index=False):
        lines.append(f"| {r.alt} | {r.coef} | {r.median_gap:.4f} | {r.frac_ge20:.4f} | {r.signflip:.4f} "
                     f"| [{r.wild_lo:.4f}, {r.wild_hi:.4f}] | {r.aic_pref_frac:.3f} | {r.bic_pref_frac:.3f} "
                     f"| {r.lrt_pref_frac:.3f} | {'YES' if r.survive else 'no'} |")
    surv = sumdf.set_index(['alt', 'coef'])['survive']
    lines.append('')
    for c in ('depth', 'promo'):
        lines.append(f'- {c}: survives under ' + ', '.join(n for n in dict(ALT_KEYS).values() if surv.get((n, c), False)) or 'NONE of the 11 alternatives.')
    lines.append(f'- Data-preferred alternative (highest share of AIC preference over M): **{best}** '
                 f'(preference shares { {k: round(v, 3) for k, v in sorted(pref.items(), key=lambda kv: -kv[1])[:4]} }).')
    lines.append('- Decision rule (pre-fixed, tex §3.9): survives iff median gap >= 20% AND frac >= 20% >= 50% AND wild-CI lower bound > 20%. '
                 'If the rule fails under a data-preferred alternative, the corresponding 25.3%/52.4% claim is SENSITIVE to the recovery-model specification and substitution is withdrawn.')
    lines.append('- Convergence failures (listed, not dropped): ' + (', '.join(
        f"{r.alt} {r.city}x{r.cat}" for r in ndf.itertuples(index=False)) if len(ndf) else 'none'))
    lines.append('- Method: per-cell MLE (L-BFGS-B, maxiter=100, M/NB2/ZIP inits reused from 07_pending_3: b0 = M fit, theta0=5, pi0=0.1/0.95); '
                 'wild CI = Rademacher weights on 18 cities, 2000 reps, median via Bahadur linearization (shared §3.8 machinery). '
                 'Per-cell detail: pending_final_pA_cells.csv.')
    return '\n'.join(lines) + '\n'


def _pa_worker(wid, keys):
    """Forked worker: fit its cell shard, append rows to its own checkpoint file."""
    out = PA_CELLS + f'.w{wid}'
    n = 0
    with open(out, 'a') as fh:
        for key in keys:
            rec = fit_cell(_GROUPS[key], *_BG[key])
            rec['city'], rec['cat'] = key
            rec = {c: rec.get(c) for c in PA_COLS}
            pd.DataFrame([rec]).to_csv(fh, index=False, header=(n == 0))
            n += 1
            print(f'[P-A w{wid}] {n}/{len(keys)} cell {key[0]}x{key[1]} '
                  f'm_conv={rec["m_conv"]} n2_conv={rec["n2_conv"]}', flush=True)
    return n


def run_pA(selftest_n=None, nproc=1):
    t0 = time.time()
    decomp = pd.read_csv(DECOMP_CSV)
    keys = set(zip(decomp.city.astype(int), decomp.cat.astype(int)))
    assert len(keys) == 254, f'expected 254 ok cells, got {len(keys)}'
    bG = {(int(r.city), int(r.cat)): (float(r.beta_glm_promo), float(r.beta_glm_depth)) for r in decomp.itertuples(index=False)}
    groups = m05.build_groups(keys)
    print(f'[P-A] panel built for {len(groups)} cells in {time.time() - t0:.0f}s', flush=True)

    if selftest_n:
        todo = sorted(keys)[:selftest_n]
        t1 = time.time()
        recs = []
        for i, k in enumerate(todo):
            t1c = time.time()
            recs.append(fit_cell(groups[k], *bG[k]))
            print(f'[P-A selftest] cell {k[0]}x{k[1]} fit {time.time() - t1c:.1f}s (cum {time.time() - t1:.0f}s)', flush=True)
        tfit = time.time() - t1
        df = pd.DataFrame([{**r, 'city': k[0], 'cat': k[1]} for r, k in zip(recs, todo)]).reindex(columns=PA_COLS)
        pa_selftest_check(df)
        print(f'[P-A] {selftest_n}-cell fit time {tfit:.1f}s (incl. panel build {time.time() - t0 - tfit:.0f}s) '
              f'-> full 254-cell estimate ~{tfit * 254 / selftest_n / 60:.0f} min serial')
        return
    global _GROUPS, _BG
    _GROUPS, _BG = groups, bG
    if os.path.exists(PA_CELLS):
        pre = pd.read_csv(PA_CELLS)
        done = set(zip(pre.city.astype(int), pre.cat.astype(int)))
    else:
        pre, done = pd.DataFrame(columns=PA_COLS), set()
    todo = sorted(k for k in groups if k not in done)
    n_tot = len(done) + len(todo)
    print(f'[P-A] {len(done)} done, {len(todo)} todo, nproc={nproc}', flush=True)
    if nproc > 1 and len(todo) >= nproc:
        import multiprocessing as mp
        for i in range(nproc):  # remove stale worker files from a crashed run
            p = PA_CELLS + f'.w{i}'
            if os.path.exists(p):
                os.remove(p)
        shards = [todo[i::nproc] for i in range(nproc)]
        ctx = mp.get_context('fork')
        with ctx.Pool(nproc) as pool:
            pool.starmap(_pa_worker, [(i, shards[i]) for i in range(nproc)])
        parts = [pre] if len(pre) else []
        for i in range(nproc):
            p = PA_CELLS + f'.w{i}'
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
        df = pd.concat(parts, ignore_index=True).sort_values(['city', 'cat']).drop_duplicates(['city', 'cat'])
        df.to_csv(PA_CELLS, index=False)
        for i in range(nproc):
            os.remove(PA_CELLS + f'.w{i}')
    else:
        if not os.path.exists(PA_CELLS):
            pd.DataFrame(columns=PA_COLS).to_csv(PA_CELLS, index=False)
        for i, key in enumerate(todo):
            rec = fit_cell(groups[key], *bG[key])
            rec['city'], rec['cat'] = key
            rec = {c: rec.get(c) for c in PA_COLS}
            with open(PA_CELLS, 'a') as fh:
                pd.DataFrame([rec]).to_csv(fh, index=False, header=False)
            print(f'[P-A] [{len(done) + i + 1}/{n_tot}] cell {key[0]}x{key[1]} '
                  f'm_conv={rec["m_conv"]} n2_conv={rec["n2_conv"]} z2_conv={rec["z2_conv"]}', flush=True)
        df = pd.read_csv(PA_CELLS)
    assert len(df) == 254, f'expected 254 rows, got {len(df)}'
    sumdf, ndf, best, pref = pa_aggregate(df)
    sumdf.to_csv(PA_SUM, index=False)
    ndf.to_csv(PA_NONCONV, index=False)
    flush_report('PA', pa_summary_md(sumdf, ndf, best, pref, df, {'cells': time.time() - t0}))
    print(f'[P-A] DONE: {PA_CELLS} ({len(df)} cells) + {PA_SUM} ({len(sumdf)} rows) in {time.time() - t0:.0f}s', flush=True)
    return sumdf, ndf, best, pref


# ---------------- P-B: Study 1 activity matching + large-band ratio CI ----------------

def _load_yelp():
    ev = pd.read_parquet(YB + 'events.parquet')
    tr = pd.read_parquet(YB + 'trajectories.parquet')
    return ev, tr


def pb_build(seq):
    """Replicates 04_main_analysis.py panel/event/pseudo construction for one seq (deterministic)."""
    ev, tr = _load_yelp()
    e = ev[(ev.seq == seq) & ev.direction.isin(['hide', 'restore'])]
    t = tr[tr.seq == seq]
    N = int(t.crawl.nunique())
    g = t.groupby('business_id').size()
    panel = set(g[g == N].index)
    evbiz = set(e.business_id) & panel
    ctrl = panel - evbiz
    sz = t[t.n_rec > 0].groupby('business_id')['n_rec'].median()
    q1, q2 = sz.quantile([1 / 3, 2 / 3])

    def band_of(s):
        if s != s:
            return None
        if s <= q1:
            return 'small'
        if s <= q2:
            return 'mid'
        return 'large'

    R = {(r.business_id, r.ci): r.avg_rec for r in t.itertuples()}
    Rn = {(r.business_id, r.ci): r.n_rec for r in t.itertuples()}
    dR, bizl, dn_e = [], [], []
    for r in e.itertuples():
        a = R.get((r.business_id, r.window))
        b2 = R.get((r.business_id, r.window + 1))
        if a is None or b2 is None:
            continue
        n1, n2 = Rn.get((r.business_id, r.window)), Rn.get((r.business_id, r.window + 1))
        if n1 is None or n2 is None:
            continue
        dR.append(abs(b2 - a)); bizl.append(r.business_id); dn_e.append(n2 - n1)
    dR, bizl, dn_e = np.array(dR), np.array(bizl), np.array(dn_e, float)
    bands_ev = np.array([band_of(sz.get(b)) for b in bizl])
    rng = random.Random(SEED)
    p_bizl, p_dR, p_dn = [], [], []
    for b in sorted(ctrl):
        cis = [ci for ci in range(N - 1) if (b, ci) in R and (b, ci + 1) in R]
        if not cis:
            continue
        for w in rng.sample(cis, min(5, len(cis))):
            n1, n2 = Rn.get((b, w)), Rn.get((b, w + 1))
            p_bizl.append(b); p_dR.append(abs(R[(b, w + 1)] - R[(b, w)]))
            p_dn.append((n2 - n1) if (n1 is not None and n2 is not None) else float('nan'))
    p_dR = np.array(p_dR)
    p_dn = np.array(p_dn, float)
    bands_ps = np.array([band_of(sz.get(b)) for b in p_bizl])
    return dict(N=N, panel=panel, evbiz=evbiz, ctrl=ctrl, sz=sz, band_of=band_of, R=R, Rn=Rn,
                dR=dR, bizl=bizl, dn_e=dn_e, bands_ev=bands_ev,
                p_bizl=np.array(p_bizl), p_dR=p_dR, p_dn=p_dn, bands_ps=bands_ps)


def pb_match(seq, B, subsample=None):
    """Activity-matched comparison per size band (CEM: log median n_rec 0.25-bins x dn_rec pooled
    quartiles; 1:1 with-replacement within stratum, seed 42 per seq)."""
    rng = random.Random(SEED)
    rows = []
    for band in ['small', 'mid', 'large']:
        ev_idx = np.where(B['bands_ev'] == band)[0]
        ps_idx = np.where(B['bands_ps'] == band)[0]
        if subsample is not None:
            ev_idx = [i for i in ev_idx if B['bizl'][i] in subsample]
            ps_idx = [i for i in ps_idx if B['p_bizl'][i] in subsample]
        if not len(ev_idx) or not len(ps_idx):
            rows.append(dict(seq=seq, band=band, n_ev=int(len(ev_idx)), n_matched=0,
                             n_unmatched=int(len(ev_idx)), med_ev=np.nan, med_ct=np.nan,
                             mw_r=np.nan, mw_p=np.nan))
            continue
        ev_idx = np.array(ev_idx); ps_idx = np.array(ps_idx)
        szv = np.array([B['sz'].get(b) for b in list(B['bizl'][ev_idx]) + list(B['p_bizl'][ps_idx])])
        lbin = np.floor(np.log(szv) / 0.25).astype(int)
        dn_pool = np.concatenate([B['dn_e'][ev_idx], B['p_dn'][ps_idx]])
        dn_pool = dn_pool[np.isfinite(dn_pool)]
        cuts = np.quantile(dn_pool, [0.25, 0.5, 0.75])
        dnb_ev = np.searchsorted(cuts, B['dn_e'][ev_idx], side='right')
        dnb_ps = np.searchsorted(cuts, B['p_dn'][ps_idx], side='right')
        lbin_ev, lbin_ps = lbin[:len(ev_idx)], lbin[len(ev_idx):]
        strata = {}
        for j, i in enumerate(ps_idx):
            strata.setdefault((lbin_ps[j], dnb_ps[j]), []).append(i)
        m_ev, m_ct, n_un = [], [], 0
        for j, i in enumerate(ev_idx):
            key = (lbin_ev[j], dnb_ev[j])
            if key not in strata:
                n_un += 1
                continue
            m_ev.append(B['dR'][i])
            m_ct.append(B['p_dR'][strata[key][rng.randrange(len(strata[key]))]])
        m_ev, m_ct = np.array(m_ev), np.array(m_ct)
        if len(m_ev) and len(m_ct):
            _, p, r = _mann_whitney(m_ev, m_ct)
        else:
            p, r = (np.nan, np.nan)
        rows.append(dict(seq=seq, band=band, n_ev=int(len(ev_idx)), n_matched=int(len(m_ev)),
                         n_unmatched=n_un,
                         med_ev=float(np.median(m_ev)) if len(m_ev) else np.nan,
                         med_ct=float(np.median(m_ct)) if len(m_ct) else np.nan,
                         mw_r=float(r), mw_p=float(p)))
    return rows


def _mann_whitney(a, b):
    """Copied verbatim from yelp-reputation/scripts/04_main_analysis.py (same formula)."""
    from math import erf, sqrt

    def ncdf(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    combined = np.concatenate([a, b])
    order = combined.argsort(kind='mergesort')
    ranks = np.empty(len(combined), float)
    sorted_c = combined[order]
    i = 0
    while i < len(sorted_c):
        j = i
        while j + 1 < len(sorted_c) and sorted_c[j + 1] == sorted_c[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r1 = ranks[:na].sum()
    u1 = r1 - na * (na + 1) / 2
    _, counts = np.unique(sorted_c, return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    nn = na + nb
    var_u = na * nb / 12 * ((nn + 1) - tie_term / (nn * (nn - 1)))
    z = (u1 - na * nb / 2) / np.sqrt(var_u)
    return float(u1), float(2 * (1 - ncdf(abs(z)))), float(2 * u1 / (na * nb) - 1)


def pb_ratio_ci(seq, B, n_boot=N_BOOT, matched_rows=None):
    """Business-level resampling CI for the large-band event/pseudo median |dR| ratio.
    Raw: resample large-band event businesses (all their events) + control businesses (their
    original pseudo draws). Matched: resample event businesses, redraw each event's stratum match.
    One RandomState(42) per seq: raw draws first, then matched (deterministic order)."""
    sel_e = B['bands_ev'] == 'large'
    sel_p = B['bands_ps'] == 'large'
    ev_biz = sorted(set(B['bizl'][sel_e]))
    ev_by_b = {b: B['dR'][(sel_e) & (np.array(B['bizl']) == b)] for b in ev_biz}
    ev_by_b = {b: v for b, v in ev_by_b.items() if len(v)}
    ev_biz = [b for b in ev_biz if b in ev_by_b]
    ps_biz = sorted(set(B['p_bizl'][sel_p]))
    ps_by_b = {b: B['p_dR'][(sel_p) & (np.array(B['p_bizl']) == b)] for b in ps_biz}
    ps_biz = [b for b in ps_biz if b in ps_by_b]
    point = float(np.median(B['dR'][sel_e]) / max(np.median(B['p_dR'][sel_p]), 1e-9))

    def pool(sample_biz, by_b):
        return np.concatenate([by_b[b] for b in sample_biz])

    rs = np.random.RandomState(SEED)
    n_ev_b, n_ps_b = len(ev_biz), len(ps_biz)
    raw = []
    for _ in range(n_boot):
        se = [ev_biz[i] for i in rs.randint(0, n_ev_b, n_ev_b)]
        sp = [ps_biz[i] for i in rs.randint(0, n_ps_b, n_ps_b)]
        me, mp = pool(se, ev_by_b), pool(sp, ps_by_b)
        raw.append(np.median(me) / max(np.median(mp), 1e-9))
    out = dict(seq=seq, point=point, n_ev_biz=n_ev_b, n_ps_biz=n_ps_b,
               raw_lo=float(np.percentile(raw, 2.5)), raw_hi=float(np.percentile(raw, 97.5)),
               raw_med=float(np.median(raw)), n_boot=n_boot, seed=SEED)
    if matched_rows is not None:
        # matched pairs for the large band: rebuild deterministically (same seed/strata as pb_match),
        # grouped by event business for business-level resampling
        pairs, _ = _match_with_pairs(seq, B, 'large')
        pairs_by_b = {}
        for b, i, cv in pairs:
            pairs_by_b.setdefault(b, []).append((i, cv))
        ev_biz_m = [b for b in ev_biz if b in pairs_by_b]  # resample units that actually contribute
        n_ev_b_m = len(ev_biz_m)
        mat = []
        for _ in range(n_boot):
            se = [ev_biz_m[i] for i in rs.randint(0, n_ev_b_m, n_ev_b_m)]
            m_ev, m_ct = [], []
            for b in se:
                for (i, cv) in pairs_by_b[b]:
                    m_ev.append(B['dR'][i])
                    m_ct.append(cv)
            if m_ev:
                mat.append(np.median(m_ev) / max(np.median(m_ct), 1e-9))
        out.update(mat_lo=float(np.percentile(mat, 2.5)), mat_hi=float(np.percentile(mat, 97.5)),
                   mat_med=float(np.median(mat)), n_pairs=int(sum(len(v) for v in pairs_by_b.values())))
    return out


def _match_with_pairs(seq, B, band):
    """Same CEM as pb_match but returns (business, event_idx, matched_control_value) triples."""
    rng = random.Random(SEED)
    # consume RNG identically to pb_match: process small->mid->large, same draws
    for bnd in ['small', 'mid', 'large']:
        ev_idx = np.where(B['bands_ev'] == bnd)[0]
        ps_idx = np.where(B['bands_ps'] == bnd)[0]
        if not len(ev_idx) or not len(ps_idx):
            continue
        szv = np.array([B['sz'].get(b) for b in list(B['bizl'][ev_idx]) + list(B['p_bizl'][ps_idx])])
        lbin = np.floor(np.log(szv) / 0.25).astype(int)
        dn_pool = np.concatenate([B['dn_e'][ev_idx], B['p_dn'][ps_idx]])
        dn_pool = dn_pool[np.isfinite(dn_pool)]
        cuts = np.quantile(dn_pool, [0.25, 0.5, 0.75])
        dnb_ev = np.searchsorted(cuts, B['dn_e'][ev_idx], side='right')
        dnb_ps = np.searchsorted(cuts, B['p_dn'][ps_idx], side='right')
        lbin_ev, lbin_ps = lbin[:len(ev_idx)], lbin[len(ev_idx):]
        strata = {}
        for j, i in enumerate(ps_idx):
            strata.setdefault((lbin_ps[j], dnb_ps[j]), []).append(i)
        if bnd != band:
            for j, i in enumerate(ev_idx):
                key = (lbin_ev[j], dnb_ev[j])
                if key in strata:
                    rng.randrange(len(strata[key]))
            continue
        out = []
        for j, i in enumerate(ev_idx):
            key = (lbin_ev[j], dnb_ev[j])
            if key not in strata:
                continue
            cv = B['p_dR'][strata[key][rng.randrange(len(strata[key]))]]
            out.append((B['bizl'][i], i, cv))
        return out, cuts
    return [], None


def run_pB(selftest_n=None):
    t0 = time.time()
    if selftest_n:
        B = pb_build('CHI')
        evb = sorted(set(B['bizl'][B['bands_ev'] == 'large']))[:10]
        psv = np.where(B['bands_ps'] == 'large')[0]
        psb = sorted({B['p_bizl'][i] for i in psv})[:10]
        sub = set(evb) | set(psb)
        rows = pb_match('CHI', B, subsample=sub)
        out = pb_ratio_ci('CHI', B, n_boot=200, matched_rows=None)
        assert np.isfinite(out['point']) and out['raw_lo'] <= out['raw_hi']
        assert all(np.isfinite(r['mw_r']) or r['n_matched'] == 0 for r in rows)
        nmatch = sum(r['n_matched'] for r in rows)
        print(f'[P-B] selftest OK (20 businesses CHI large subsample): point={out["point"]:.4f} '
              f'raw CI=[{out["raw_lo"]:.4f},{out["raw_hi"]:.4f}] matched_pairs={nmatch} ({time.time() - t0:.1f}s)')
        return
    match_rows, ratio_rows = [], []
    for seq in ['CHI', 'UDIS']:
        B = pb_build(seq)
        mrows = pb_match(seq, B)
        # verify raw point ratio reproduces size_band_summary
        sel_e, sel_p = B['bands_ev'] == 'large', B['bands_ps'] == 'large'
        pt = float(np.median(B['dR'][sel_e]) / max(np.median(B['p_dR'][sel_p]), 1e-9))
        expect = {'CHI': 0.9164110318289548, 'UDIS': 2.1849781468210923}
        assert abs(pt - expect[seq]) < 1e-9, f'{seq} large-band ratio {pt} != shipped {expect[seq]}'
        match_rows += mrows
        ratio_rows.append(pb_ratio_ci(seq, B, matched_rows=mrows))
        print(f'[P-B] {seq}: raw point={pt:.4f} (shipped reproduced) '
              + ' '.join(f"{r['band']} matched={r['n_matched']}/{r['n_ev']} r={r['mw_r']:.3f}" for r in mrows),
              flush=True)
    mdf = pd.DataFrame(match_rows)
    rdf = pd.DataFrame(ratio_rows)
    mdf.to_csv(PB_MATCH, index=False)
    rdf.to_csv(PB_RATIO, index=False)
    flush_report('PB', pb_summary_md(mdf, rdf))
    print(f'[P-B] DONE: {PB_MATCH} + {PB_RATIO} in {time.time() - t0:.0f}s', flush=True)
    return mdf, rdf


def pb_summary_md(mdf, rdf):
    lines = ['## P-B — tex §4.3.6: Study 1 activity-matching + large-band ratio CI (DONE)',
             '',
             'Conventions = yelp-reputation/scripts/04_main_analysis.py (panel = businesses in all crawls, seed=42, '
             'size bands = terciles of business median n_rec, same pseudo-window draws as shipped). '
             'Large-band raw point ratio reproduces size_band_summary.csv exactly (CHI 0.9164, UDIS 2.1850).',
             '',
             '**1. Activity-matched comparison** (within size band x seq; CEM on log median n_rec in 0.25-log-unit bins '
             'x window new-review count dn_rec = n_rec(w+1) - n_rec(w) in pooled quartile bins; 1:1 with-replacement '
             'within stratum, seed 42):',
             '',
             '| seq | band | event windows | matched | unmatched | matched med \\|dR\\| event | matched med \\|dR\\| control | matched r | p |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in mdf.itertuples(index=False):
        lines.append(f"| {r.seq} | {r.band} | {r.n_ev} | {r.n_matched} | {r.n_unmatched} | {r.med_ev:.5f} | {r.med_ct:.5f} "
                     f"| {r.mw_r:+.4f} | {r.mw_p:.4f} |")
    lines += ['', '**2. Large-band ratio 95% percentile CI** (business-level resampling, 2000 reps, seed 42; events nested in businesses):',
              '',
              '| seq | sample | point | CI lo | CI hi | event biz | control biz | pairs | includes 1? |',
              '|---|---|---|---|---|---|---|---|---|']
    for r in rdf.itertuples(index=False):
        lines.append(f"| {r.seq} | raw | {r.point:.4f} | {r.raw_lo:.4f} | {r.raw_hi:.4f} | {r.n_ev_biz} | {r.n_ps_biz} | — | "
                     f"{'yes' if r.raw_lo <= 1 <= r.raw_hi else 'NO'} |")
        if 'mat_lo' in r._asdict() and np.isfinite(getattr(r, 'mat_lo', np.nan)):
            lines.append(f"| {r.seq} | matched | {r.point:.4f} | {r.mat_lo:.4f} | {r.mat_hi:.4f} | {r.n_ev_biz} | — | {r.n_pairs} | "
                         f"{'yes' if r.mat_lo <= 1 <= r.mat_hi else 'NO'} |")
    chi = rdf[rdf.seq == 'CHI'].iloc[0]
    verdict = ('CI excludes 1 -> the "not detectable in large businesses" statement is WITHDRAWN per the pre-fixed decision rule.'
               if not (chi.raw_lo <= 1 <= chi.raw_hi) and not (chi.mat_lo <= 1 <= chi.mat_hi)
               else 'CI includes 1 -> footprint in large businesses remains indistinguishable from the pseudo-event baseline '
                    '(inconclusive if the interval is wide, not a demonstrated null).')
    lines.append(f'\nDecision (pre-fixed, tex §4.3.6): CHI large-band raw CI [{chi.raw_lo:.4f}, {chi.raw_hi:.4f}]. {verdict}')
    lines.append('Method note: matched-sample bootstrap resamples event businesses and redraws each event within-stratum match; '
                 'raw-sample bootstrap resamples event + control businesses (all their events / original pseudo draws). '
                 'Detail: pending_final_pB_matching.csv, pending_final_pB_ratio_ci.csv.')
    return '\n'.join(lines) + '\n'


# ---------------- P-C: wild-cluster CI ----------------

PC_STATS = [
    ('P1a_median_abs_bias_ratio', 'main', 'bias_ratio', 'median', None),
    ('P1a_frac_ge20', 'main', 'bias_ratio', 'frac', 0.20),
    ('P1b_depth_median_abs_bias_pct', 'decomp', 'bias_pct_orig_depth', 'median', None),
    ('P1b_depth_frac_ge20', 'decomp', 'bias_pct_orig_depth', 'frac', 0.20),
    ('P1b_promo_median_abs_bias_pct', 'decomp', 'bias_pct_orig_promo', 'median', None),
    ('P1b_promo_frac_ge20', 'decomp', 'bias_pct_orig_promo', 'frac', 0.20),
    ('pure_censoring_depth_median', 'decomp', 'bias_pct_pure_censoring_depth', 'median', None),
    ('pure_censoring_depth_frac_ge20', 'decomp', 'bias_pct_pure_censoring_depth', 'frac', 0.20),
    ('pure_censoring_promo_median', 'decomp', 'bias_pct_pure_censoring_promo', 'median', None),
    ('pure_censoring_promo_frac_ge20', 'decomp', 'bias_pct_pure_censoring_promo', 'frac', 0.20),
]


def pc_run(n_boot=N_BOOT, seed=SEED, main_df=None, decomp_df=None):
    main_df = main_df if main_df is not None else pd.read_csv(MAIN_CSV)
    decomp_df = decomp_df if decomp_df is not None else pd.read_csv(DECOMP_CSV)
    srcs = {'main': main_df[main_df.status == 'ok'], 'decomp': decomp_df}
    rows = []
    for name, src, col, kind, thr in PC_STATS:
        d = srcs[src]
        v = d[col].astype(float).abs().values
        cits = d['city'].astype(int).astype(str).values
        T, lo, hi = wild_ci(v, cits, n_boot=n_boot, seed=seed, kind=kind, thresh=thr)
        rows.append(dict(stat=name, point=T, wild_ci_lo=lo, wild_ci_hi=hi))
    return pd.DataFrame(rows)


def pc_summary_md(df, block):
    lines = ['## P-C — tex §3.8: wild-cluster refinement (DONE)',
             '',
             f'Rademacher (+-1, uniform) wild weights on the 18 cities, {N_BOOT} reps, seed=42, percentile CI. '
             'City contributions per statistic: frac statistics = centered per-city exceedance sums (statistic is exactly '
             'additive in them); median statistics = Bahadur linearization at the median, per-city sums of '
             '(1{|v|<=m}-1/2)/fhat(m), fhat = local empirical density (delta=0.1*IQR). '
             'Compared with the existing city-block (resample cities with replacement) CIs from pending_joint_ci.csv '
             f'(same {N_BOOT} reps, seed 42).',
             '',
             '| statistic | point | wild 95% CI | block 95% CI | width ratio wild/block |',
             '|---|---|---|---|---|']
    for r in df.itertuples(index=False):
        b = block.loc[r.stat]
        w_w, w_b = r.wild_ci_hi - r.wild_ci_lo, b['ci_hi'] - b['ci_lo']
        lines.append(f"| {r.stat} | {r.point:.4f} | [{r.wild_ci_lo:.4f}, {r.wild_ci_hi:.4f}] "
                     f"| [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}] | {w_w / w_b:.2f}x |")
    dec = []
    for stat, thr, lab in (('pure_censoring_depth_median', 0.2, 'pure-censoring median depth gap > 20%'),
                           ('pure_censoring_promo_median', 0.2, 'pure-censoring median promo gap > 20%'),
                           ('pure_censoring_depth_frac_ge20', 0.5, 'depth frac >= 20% exceeds 50%'),
                           ('pure_censoring_promo_frac_ge20', 0.5, 'promo frac >= 20% exceeds 50%')):
        r = df[df.stat == stat].iloc[0]
        ok = r.wild_ci_lo > thr
        dec.append(f'- {lab}: wild CI [{r.wild_ci_lo:.4f}, {r.wild_ci_hi:.4f}] -> '
                   f"{'RETAINED at full strength' if ok else 'RESTATED as unresolved at the threshold level'} "
                   f"(lower bound {'>' if ok else '<='} {thr})")
    lines.append('\nDecision rule (pre-fixed, tex §3.8): a headline claim is retained at full strength only if the wild-cluster CI lower bound is above its threshold.\n')
    lines += dec
    lines.append('\nDetail: pending_final_wildci.csv.')
    return '\n'.join(lines) + '\n'


def run_pC(selftest_n=None):
    t0 = time.time()
    if selftest_n:
        mdf = pd.read_csv(MAIN_CSV)
        sub = mdf[mdf.status == 'ok'].groupby('city').size().sort_values(ascending=False)
        cities10 = sorted(sub.index[:10])
        m10 = mdf[(mdf.status == 'ok') & mdf.city.isin(cities10)].reset_index(drop=True)
        d10 = pd.read_csv(DECOMP_CSV)[lambda d: d.city.isin(cities10)].reset_index(drop=True)
        df1 = pc_run(n_boot=200, seed=1, main_df=m10, decomp_df=d10)
        for r in df1.itertuples(index=False):
            assert r.wild_ci_lo <= r.point <= r.wild_ci_hi, f'{r.stat}: point not in CI'
        df2 = pc_run(n_boot=200, seed=1, main_df=m10, decomp_df=d10)
        assert df1.equals(df2), 'not deterministic under fixed seed'
        print(f'[P-C] selftest OK (10 cities, n_boot=200, CI brackets point, deterministic) {time.time() - t0:.1f}s')
        return
    df = pc_run(n_boot=N_BOOT, seed=SEED)
    block = pd.read_csv(JOINT_CSV).set_index('stat')[['point', 'ci_lo', 'ci_hi']]
    for r in df.itertuples(index=False):
        jpt = block.loc[r.stat, 'point']
        assert abs(r.point - jpt) < 1e-12, f'{r.stat}: point {r.point} != joint-ci {jpt}'
    out = df.merge(block, left_on='stat', right_index=True, how='left')
    out['method_wild'] = (f'Rademacher wild-cluster percentile: 18 cities, weights +-1 uniform, n_boot={N_BOOT}, '
                          f'seed={SEED}; frac = centered city exceedance sums, median = Bahadur linearization')
    out['method_block'] = 'city-block (cities resampled with replacement), n_boot=2000, seed=42 (pending_joint_ci.csv)'
    out.to_csv(PC_WILD, index=False)
    flush_report('PC', pc_summary_md(df, block))
    print(f'[P-C] DONE: {PC_WILD} ({len(df)} stats) in {time.time() - t0:.0f}s', flush=True)
    for r in df.itertuples(index=False):
        b = block.loc[r.stat]
        print(f'[P-C] {r.stat}: point={r.point:.4f} wild=[{r.wild_ci_lo:.4f},{r.wild_ci_hi:.4f}] '
              f'block=[{b["ci_lo"]:.4f},{b["ci_hi"]:.4f}]', flush=True)
    return out


# ---------------- report plumbing ----------------

def init_report():
    txt = f'''# JAMS-50K final pending analyses — P-A §3.9 / P-B §4.3.6 / P-C §3.8 (RUNNING)

Script: `main/08_pending_final.py` — seed=42, deterministic. Spec: `latex/main.tex` §§3.9/4.3.6/3.8 (Prism pre-fixed).
Non-converged fits/cells are listed, not dropped. Report flushed per section (interruption-tolerant).

<!-- PA BEGIN -->
## P-A — tex §3.9: recovery-model alternatives — RUNNING
<!-- PA END -->

<!-- PB BEGIN -->
## P-B — tex §4.3.6: Study 1 activity matching + large-band ratio CI — RUNNING
<!-- PB END -->

<!-- PC BEGIN -->
## P-C — tex §3.8: wild-cluster refinement — RUNNING
<!-- PC END -->
'''
    with open(REPORT, 'w') as fh:
        fh.write(txt)


def flush_report(tag, md):
    txt = open(REPORT).read()
    sec = f'<!-- {tag} BEGIN -->\n{md}<!-- {tag} END -->'
    txt = re.sub(f'<!-- {tag} BEGIN -->.*?<!-- {tag} END -->', lambda m: sec, txt, flags=re.S)
    first, _, rest = txt.partition('\n')
    title_done = 'DONE' if 'RUNNING' not in rest else 'RUNNING'
    first = first.replace('(RUNNING)', f'({title_done})').replace('(DONE)', f'({title_done})')
    with open(REPORT, 'w') as fh:
        fh.write(first + '\n' + rest)


def main():
    selftest = '--selftest' in sys.argv
    nproc = int(sys.argv[sys.argv.index('--nproc') + 1]) if '--nproc' in sys.argv else 1
    if selftest:
        print('=== SELFTEST (P-A 10 cells / P-B 20 businesses / P-C 10 cities) ===', flush=True)
        run_pC(selftest_n=10)
        run_pB(selftest_n=20)
        run_pA(selftest_n=10)
        return
    init_report()
    pa = run_pA(nproc=nproc)
    pb = run_pB()
    pc = run_pC()
    # final headline (5 lines)
    sumdf = pd.read_csv(PA_SUM)
    ndf = pd.read_csv(PA_NONCONV)
    best = sumdf[sumdf.data_preferred]['alt'].iloc[0]
    d = sumdf[sumdf.alt == best].set_index('coef')
    surv_d = bool(d.loc['depth', 'survive']); surv_p = bool(d.loc['promo', 'survive'])
    mdf = pd.read_csv(PB_MATCH)
    rdf = pd.read_csv(PB_RATIO)
    ch = rdf[rdf.seq == 'CHI'].iloc[0]
    ud = rdf[rdf.seq == 'UDIS'].iloc[0]
    m_l = mdf[(mdf.seq == 'CHI') & (mdf.band == 'large')].iloc[0]
    wc = pd.read_csv(PC_WILD)

    def row(stat):
        return wc[wc.stat == stat].iloc[0]

    pc_d = row('pure_censoring_depth_median')
    pc_p = row('pure_censoring_promo_median')
    print('HEADLINE:')
    print(f'1) P-A: gaps vs M (25.3%/52.4%) under data-preferred {best}: survives under it: '
          f'depth={"YES" if surv_d else "NO"} ({d.loc["depth","median_gap"]:.3f} [wild {d.loc["depth","wild_lo"]:.3f},{d.loc["depth","wild_hi"]:.3f}]), '
          f'promo={"YES" if surv_p else "NO"} ({d.loc["promo","median_gap"]:.3f} [{d.loc["promo","wild_lo"]:.3f},{d.loc["promo","wild_hi"]:.3f}]) '
          f'({len(ndf)} non-converged cell-fits listed)')
    print(f'2) P-B: CHI large matched median |dR| ev={m_l.med_ev:.5f} vs ctl={m_l.med_ct:.5f} (r={m_l.mw_r:+.3f}, '
          f'{m_l.n_matched}/{m_l.n_ev} matched); raw large-ratio CIs: CHI 0.92 [{ch.raw_lo:.3f},{ch.raw_hi:.3f}] '
          f'matched [{ch.mat_lo:.3f},{ch.mat_hi:.3f}]; UDIS 2.18 [{ud.raw_lo:.3f},{ud.raw_hi:.3f}] matched [{ud.mat_lo:.3f},{ud.mat_hi:.3f}]')
    print(f'3) P-C wild vs block (width x): P1a med 0.456 [{wc[wc.stat=="P1a_median_abs_bias_ratio"].iloc[0].wild_ci_lo:.3f},'
          f'{wc[wc.stat=="P1a_median_abs_bias_ratio"].iloc[0].wild_ci_hi:.3f}] vs [0.419,0.518] '
          f'({(pc_d.wild_ci_hi-pc_d.wild_ci_lo)/(0.2915-0.2309):.2f}x depth-median)')
    print(f'4) P-C: pure-censoring depth median wild CI [{pc_d.wild_ci_lo:.3f},{pc_d.wild_ci_hi:.3f}] (block [0.231,0.292]); '
          f'promo [{pc_p.wild_ci_lo:.3f},{pc_p.wild_ci_hi:.3f}] (block [0.416,0.638]) — '
          f"{'both lower bounds > 20%: claims RETAINED' if (pc_d.wild_ci_lo > 0.2 and pc_p.wild_ci_lo > 0.2) else 'lower bound <= 20%: claim RESTATED unresolved'}")
    pf_d = row('pure_censoring_depth_frac_ge20'); pf_p = row('pure_censoring_promo_frac_ge20')
    print(f'5) P-C frac>=20% wild CIs: depth [{pf_d.wild_ci_lo:.3f},{pf_d.wild_ci_hi:.3f}] (block [0.556,0.644]), '
          f'promo [{pf_p.wild_ci_lo:.3f},{pf_p.wild_ci_hi:.3f}] (block [0.724,0.818]) — '
          f"{'both lower bounds > 50%: RETAINED' if (pf_d.wild_ci_lo > 0.5 and pf_p.wild_ci_lo > 0.5) else 'one lower bound <= 50%: RESTATED'}")


if __name__ == '__main__':
    main()
