# -*- coding: utf-8 -*-
"""10_fix_posthoc.py — post-hoc fix for the two 2026-11-01 bugs (no refit):

1. wild_ci weights were {0,-1}; fixed in 08_pending_final.py (Rademacher {+/-1}).
   -> recompute ALL P-A and P-C wild CIs from the stored per-cell values.
2. n2_ll / z1_ll stored -ll (mislabeled), so AIC/BIC/LRT preference was an artifact.
   -> flip sign, recompute AIC/BIC/LRT on converged cells only.

Outputs: fix_posthoc_pA.csv (per alt x coef: gap stats + CIs + AIC/BIC/LRT),
         fix_posthoc_pC.csv (9 stats, fixed wild CIs vs block CIs).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT, SEED = 2000, 42

def wild_ci(vals, cities, n_boot=N_BOOT, seed=SEED, kind='frac', thresh=0.20):
    """Corrected Rademacher wild-cluster CI (identical construction to 08, fixed weights)."""
    vals = np.asarray(vals, float)
    cities = np.asarray(cities)
    N = len(vals)
    rng = np.random.RandomState(seed)
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

# ---------------- P-A ----------------
a = pd.read_csv(os.path.join(HERE, 'pending_final_pA_cells.csv'))
assert len(a) == 254
cits = a['city'].astype(str).values
# sign fix: n2_ll and z1_ll were stored as -ll
for c in ('n2_ll', 'z1_ll'):
    assert (a[c] > 0).all() or (a[c] < 0).all(), f'{c} mixed signs'
    a[c] = -a[c]
    print(f'{c}: sign flipped; now min={a[c].min():.1f} max={a[c].max():.1f}')

ALT_KEYS = [('n2', 'NB2'), ('z1', 'ZIP_Z=1'), ('z2', 'ZIP_Z=[1,promo]'),
            ('m95', 'mix_pi.95'), ('m85', 'mix_pi.85'), ('m75', 'mix_pi.75'),
            ('mp', 'mix_pi_hat'), ('nm95', 'NB2+mix.95'), ('nm85', 'NB2+mix.85'),
            ('nm75', 'NB2+mix.75'), ('nmp', 'NB2+mix_pihat')]
rows = []
for key, name in ALT_KEYS:
    ok = (a[f'{key}_conv'] == 'True') | (a[f'{key}_conv'] == True)
    n_conv = int(ok.sum())
    ll_a = a.loc[ok, f'{key}_ll'].astype(float)
    ll_m = a.loc[ok, 'm_ll'].astype(float)
    k_a = 15 if key in ('n2', 'z1', 'mp', 'nm95', 'nm85', 'nm75') else (16 if key in ('z2', 'nmp') else 14)
    k_m = 14
    nn = len(ll_a)
    aic_d = 2 * (k_a - k_m) - 2 * (ll_a - ll_m)
    bic_d = (k_a - k_m) * np.log(nn) - 2 * (ll_a - ll_m)
    for coef, c_a, c_g in (('depth', f'{key}_bd', 'bG_d'), ('promo', f'{key}_bp', 'bG_p')):
        bG = a[c_g].astype(float).values
        bA_all = a[c_a].astype(float).values
        gap_all = np.abs(bG - bA_all) / np.maximum(np.abs(bA_all), 1e-9)
        bA_ok = bA_all[ok.values]
        gap_ok = np.abs(bG[ok.values] - bA_ok) / np.maximum(np.abs(bA_ok), 1e-9)
        c_ok = cits[ok.values]
        med_all, lo_all, hi_all = wild_ci(gap_all, cits, kind='median')
        med_ok, lo_ok, hi_ok = (np.median(gap_ok), np.nan, np.nan) if n_conv < 5 else wild_ci(gap_ok, c_ok, kind='median')
        frac_all = float((gap_all >= 0.20).mean())
        rows.append(dict(alt=name, coef=coef, n_conv=n_conv,
                         med_all_cells=round(float(med_all), 4), frac_all=round(frac_all, 4),
                         wild_all_lo=round(lo_all, 4), wild_all_hi=round(hi_all, 4),
                         med_conv=round(float(med_ok), 4),
                         wild_conv_lo=round(lo_ok, 4), wild_conv_hi=round(hi_ok, 4)))
    if nn >= 2:
        rows.append(dict(alt=name, coef='AIC/BIC/LRT', n_conv=nn,
                         aic_pref=round(float((aic_d < 0).mean()), 3),
                         aic_med_diff=round(float(np.median(aic_d)), 1),
                         bic_pref=round(float((bic_d < 0).mean()), 3),
                         lrt_pref=round(float((ll_a > ll_m).mean()), 3)))
pdf = pd.DataFrame(rows)
pdf.to_csv(os.path.join(HERE, 'fix_posthoc_pA.csv'), index=False)
pd.set_option('display.width', 200)
print(pdf.to_string(index=False))

# ---------------- P-C ----------------
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
main_df = pd.read_csv(os.path.join(HERE, 'main_cells.csv'))
main_df = main_df[main_df['status'] == 'ok']
assert len(main_df) == 254, len(main_df)
decomp_df = pd.read_csv(os.path.join(HERE, 'decomp_cells.csv'))
block = pd.read_csv(os.path.join(HERE, 'pending_joint_ci.csv'))
srcs = {'main': main_df, 'decomp': decomp_df}
out = []
for name, src, col, kind, thr in PC_STATS:
    d = srcs[src]
    v = d[col].astype(float).abs().values
    c = d['city'].astype(str).values
    T, lo, hi = wild_ci(v, c, kind=kind, thresh=thr)
    b = block[block['stat'] == name].iloc[0]
    out.append(dict(stat=name, point=round(T, 4),
                    wild_lo=round(lo, 4), wild_hi=round(hi, 4),
                    block_lo=round(float(b['ci_lo']), 4), block_hi=round(float(b['ci_hi']), 4),
                    width_ratio=round((hi - lo) / (float(b['ci_hi']) - float(b['ci_lo'])), 2)))
pC = pd.DataFrame(out)
pC.to_csv(os.path.join(HERE, 'fix_posthoc_pC.csv'), index=False)
print('\n--- P-C (fixed wild CI) ---')
print(pC.to_string(index=False))
print('\nDONE')
