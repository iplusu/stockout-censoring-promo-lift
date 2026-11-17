"""Pending #4 — joint uncertainty for the (naive, M) gap statistics.

City-level wild-cluster bootstrap: resample the 18 CITIES with replacement, pool
all cells of the resampled cities — each cell carries its (naive, M) PAIR, so the
gap statistics depend on both sides jointly within one resample (no refit: the
per-cell estimates are deterministic given the data; 'joint' = the pair is
resampled as a unit and every statistic below is a functional of both naive and M).

Reuses 06_bootstrap_ci.py verbatim for the machinery (load / frac_ge / median_abs /
sign_flip_frac / bootstrap_ci: same city-block resampling, same percentile CIs).
n_boot=2000, seed=42 (06 uses 5000; MC endpoint SE ~ sd/sqrt(2000) ~ 0.5-1 point,
small vs the 1-4 point CI widths reported — basis stated in report section 4).

Statistics (point + 95% percentile CI):
  gap distribution:  median |gap|, pooled 90th percentile |gap|  for
                    P1a bias_ratio, P1b bias_pct (depth, promo), P1b pure-censoring (depth, promo)
  threshold share:   frac |gap| >= 20% for the same six gaps (pooled depth+promo included)
  sign flips:        sign(naive) != sign(M) for bias_pct depth/promo (joint, existing 06 stats)

Outputs: pending_joint_ci.csv (stat, point, ci_lo, ci_hi, method) + pending_report.md section 4.
Selftest: 10-cell city-block subsample + n_boot=200 — CI brackets point, deterministic rerun.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('m06', os.path.join(HERE, '06_bootstrap_ci.py'))
m06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m06)

MAIN_CSV = os.path.join(HERE, 'main_cells.csv')
DECOMP_CSV = os.path.join(HERE, 'decomp_cells.csv')
OUT_CSV = os.path.join(HERE, 'pending_joint_ci.csv')
REPORT = os.path.join(HERE, 'pending_report.md')
SECMARK = 'SEC4'
N_BOOT = 2000
SEED = 42
METHOD = ('city-block wild-cluster percentile bootstrap: 18 cities resampled with replacement, '
          '(naive, M) cell pairs pooled jointly (no refit; per-cell estimates deterministic), '
          f'n_boot={N_BOOT}, seed={SEED}; CI = 2.5/97.5 percentiles')


def pooled_q90(rows, cols):
    vals = [abs(float(r[c])) for r in rows for c in cols]
    if not vals:
        return float('nan')
    return float(np.percentile(vals, 90))


def run(n_boot, seed, main_path=MAIN_CSV, decomp_path=DECOMP_CSV):
    main_cells = [r for r in m06.load(main_path) if r['status'] == 'ok']
    decomp_cells = m06.load(decomp_path)
    cities = sorted(set(r['city'] for r in main_cells))
    by_city_main = {c: [r for r in main_cells if r['city'] == c] for c in cities}
    by_city_decomp = {c: [r for r in decomp_cells if r['city'] == c] for c in cities}
    specs = [
        # (name, by_city, stat_fn)
        ('P1a_median_abs_bias_ratio', by_city_main, lambda rows: m06.median_abs(rows, 'bias_ratio')),
        ('P1a_pooled_q90_abs_bias_ratio', by_city_main, lambda rows: pooled_q90(rows, ['bias_ratio'])),
        ('P1a_frac_ge20', by_city_main, lambda rows: m06.frac_ge(rows, 'bias_ratio', 0.20)),
        ('P1b_depth_median_abs_bias_pct', by_city_decomp, lambda rows: m06.median_abs(rows, 'bias_pct_orig_depth')),
        ('P1b_depth_pooled_q90_abs_bias_pct', by_city_decomp, lambda rows: pooled_q90(rows, ['bias_pct_orig_depth'])),
        ('P1b_depth_frac_ge20', by_city_decomp, lambda rows: m06.frac_ge(rows, 'bias_pct_orig_depth', 0.20)),
        ('P1b_promo_median_abs_bias_pct', by_city_decomp, lambda rows: m06.median_abs(rows, 'bias_pct_orig_promo')),
        ('P1b_promo_pooled_q90_abs_bias_pct', by_city_decomp, lambda rows: pooled_q90(rows, ['bias_pct_orig_promo'])),
        ('P1b_promo_frac_ge20', by_city_decomp, lambda rows: m06.frac_ge(rows, 'bias_pct_orig_promo', 0.20)),
        ('P1b_pooled_dp_median_abs_bias_pct', by_city_decomp, lambda rows: float(np.median(np.abs([float(r['bias_pct_orig_depth']) for r in rows] + [float(r['bias_pct_orig_promo']) for r in rows])))),
        ('P1b_pooled_dp_frac_ge20', by_city_decomp, lambda rows: _pooled_frac(rows, 0.20)),
        ('pure_censoring_depth_median', by_city_decomp, lambda rows: m06.median_abs(rows, 'bias_pct_pure_censoring_depth')),
        ('pure_censoring_depth_frac_ge20', by_city_decomp, lambda rows: m06.frac_ge(rows, 'bias_pct_pure_censoring_depth', 0.20)),
        ('pure_censoring_promo_median', by_city_decomp, lambda rows: m06.median_abs(rows, 'bias_pct_pure_censoring_promo')),
        ('pure_censoring_promo_frac_ge20', by_city_decomp, lambda rows: m06.frac_ge(rows, 'bias_pct_pure_censoring_promo', 0.20)),
        ('sign_flip_orig_depth', by_city_decomp, lambda rows: m06.sign_flip_frac(rows, 'beta_naive_reduced_depth', 'beta_rec_depth')),
        ('sign_flip_orig_promo', by_city_decomp, lambda rows: m06.sign_flip_frac(rows, 'beta_naive_reduced_promo', 'beta_rec_promo')),
    ]
    out = {}
    for name, by_city, fn in specs:
        out[name] = m06.bootstrap_ci(fn, by_city, cities, n_boot=n_boot, seed=seed)
    return out, len(cities), len(main_cells), len(decomp_cells)


def _pooled_frac(rows, thresh):
    vals = [abs(float(r['bias_pct_orig_depth'])) for r in rows] + [abs(float(r['bias_pct_orig_promo'])) for r in rows]
    return sum(1 for v in vals if v >= thresh) / len(vals)


def write_section(df, n_city, n_main, n_decomp):
    lines = ['# Pending #4 — joint (naive, M) uncertainty: city-block wild-cluster bootstrap\n',
             f'Script: `main/07_pending_4_jointci.py`. {n_city} cities, {n_main} P1a cells, {n_decomp} P1b/decomp cells; '
             f'city-block resampling code REUSED from `main/06_bootstrap_ci.py` (bootstrap_ci / frac_ge / median_abs / sign_flip_frac, '
             f'n_boot={N_BOOT}, seed={SEED}). No refit — per-cell (naive, M) estimates are deterministic, so the bootstrap resamples the '
             'PAIR as a unit and each statistic is a joint functional of naive and M.\n',
             'n_boot rationale: MC standard error of a percentile endpoint ~ sd(stat)/sqrt(n_boot) ~ 0.5-1 point for these statistics, '
             '<= 0.25x the typical CI half-width (1-4 points); 06 used 5000 for 10 stats, 2000 keeps MC error negligible for these 17.\n',
             '\n| statistic | point | 95% CI lo | 95% CI hi |\n|---|---|---|---|']
    for r in df.itertuples(index=False):
        lines.append(f'| {r.stat} | {r.point:.4f} | {r.ci_lo:.4f} | {r.ci_hi:.4f} |')
    lines.append(f'\nMethod (all rows): {METHOD}.')
    lines.append('\nSee `pending_joint_ci.csv`.')
    md = '\n'.join(lines) + '\n'
    txt = open(REPORT).read()
    sec = f'<!-- {SECMARK} BEGIN -->\n{md}<!-- {SECMARK} END -->'
    txt = re.sub(f'<!-- {SECMARK} BEGIN -->.*?<!-- {SECMARK} END -->', lambda m: sec, txt, flags=re.S)
    with open(REPORT, 'w') as fh:
        fh.write(txt)
    return md


def main(selftest_n=None):
    if selftest_n:
        # 10-city subsample selftest: CI brackets point + deterministic rerun
        mdf = pd.read_csv(MAIN_CSV)
        sub = mdf[mdf.status == 'ok'].groupby('city').size().sort_values(ascending=False)
        cities10 = sorted(sub.index[:10])
        mdf10 = mdf[(mdf.status == 'ok') & mdf.city.isin(cities10)]
        ddf = pd.read_csv(DECOMP_CSV)
        mdf10.to_csv('/tmp/pj4_main10.csv', index=False)
        ddf[ddf.city.isin(cities10)].to_csv('/tmp/pj4_decomp10.csv', index=False)
        out1, n_c, n_m, n_d = run(n_boot=200, seed=1, main_path='/tmp/pj4_main10.csv', decomp_path='/tmp/pj4_decomp10.csv')
        for name, (point, lo, hi) in out1.items():
            assert lo <= point <= hi, f'{name}: point {point} not in [{lo}, {hi}]'
        out2, *_ = run(n_boot=200, seed=1, main_path='/tmp/pj4_main10.csv', decomp_path='/tmp/pj4_decomp10.csv')
        assert out1 == out2, 'not deterministic under fixed seed'
        assert n_c == 10
        print(f'selftest OK ({n_c} cities, {n_m} main cells, {n_d} decomp cells, n_boot=200)')
        return
    out, n_city, n_main, n_decomp = run(N_BOOT, SEED)
    df = pd.DataFrame([dict(stat=k, point=v[0], ci_lo=v[1], ci_hi=v[2], method=METHOD) for k, v in out.items()])
    df.to_csv(OUT_CSV, index=False)
    write_section(df, n_city, n_main, n_decomp)
    print(f'DONE: {OUT_CSV} ({len(df)} stats) + report section 4 ({n_city} cities, {n_main}/{n_decomp} cells)')
    for r in df.itertuples(index=False):
        print(f'{r.stat}: point={r.point:.4f} CI=[{r.ci_lo:.4f}, {r.ci_hi:.4f}]')


if __name__ == '__main__':
    n = int(sys.argv[sys.argv.index('--selftest') + 1]) if '--selftest' in sys.argv else None
    main(selftest_n=n)
