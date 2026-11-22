"""Combined predicted promotion effect L_m(d) = exp(b_promo + d*b_depth) - 1 for the matched
G and M models at five realistic discount depths, plus the paired 18-city block bootstrap
(Section 3, "Does the coefficient gap translate to a predicted-demand difference...").

Reported difference = G minus M, in percentage points.
Bootstrap: cities resampled with replacement; each drawn cell's stored G and M coefficients
(decomp_cells.csv) are reused. Because every cell is estimated separately, refitting a
resampled cell would return the same coefficients, so reuse == refit here.

Run from the package root:  python code/main/12_discount_scenario.py
Raw data path: env FRN_PARQUET (same as 05_decomp.py). Outputs go to data/main/.
Consolidated from the three original inline runs (cell_max_depth / discount_scenario_L /
discount_scenario_jointboot); outputs are identical to the paper's frozen files.
"""
import json, os
import numpy as np, pandas as pd

P = os.environ.get('FRN_PARQUET', '/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet')
OUT = os.environ.get('OUT_DIR', 'data/main')
D = {'p10': 0.036, 'p25': 0.087, 'p50': 0.144, 'p75': 0.234, 'p90': 0.334}  # p10..p90 of promo-day depth


def L(bp, bd, d):
    return np.exp(bp + d * bd) - 1.0


# 1) per-cell max observed promotion depth (extrapolation guard)
raw = pd.read_parquet(P, columns=['city_id', 'first_category_id', 'discount'])
raw.columns = ['city', 'cat', 'discount']
promo = raw[raw['discount'] < 1.0]
cellmax = (1 - promo['discount']).groupby([promo['city'], promo['cat']]).max().rename('max_obs_depth').reset_index()
cellmax.to_csv(f'{OUT}/cell_max_depth.csv', index=False)

dc = pd.read_csv(f'{OUT}/decomp_cells.csv').merge(cellmax, on=['city', 'cat'], how='left')
assert dc['max_obs_depth'].notna().all()

# 2) point summaries per depth
rows = []
for label, d in D.items():
    LG = L(dc['beta_glm_promo'], dc['beta_glm_depth'], d)
    LM = L(dc['beta_rec_promo'], dc['beta_rec_depth'], d)
    diff = 100.0 * (LG - LM)
    ok = ~(d > dc['max_obs_depth'])
    rows.append({'d_label': label, 'd': d,
                 'n_cells_in_range': int(ok.sum()), 'n_cells_extrapolated': int((~ok).sum()),
                 'median_L_G_pct': 100 * LG[ok].median(), 'median_L_M_pct': 100 * LM[ok].median(),
                 'median_diff_pp': diff[ok].median(), 'mean_abs_diff_pp': diff[ok].abs().mean(),
                 'pct_cells_diff_gt_5pp': (diff[ok].abs() > 5).mean() * 100,
                 'pct_same_sign': (np.sign(LG) == np.sign(LM))[ok].mean() * 100})
pd.DataFrame(rows).to_csv(f'{OUT}/discount_scenario_L.csv', index=False)

# 3) paired 18-city block bootstrap, B=5000, seed 42 (same draw order as the original run)
rng = np.random.default_rng(42)
cities = dc['city'].unique()
grp = {c: dc[dc['city'] == c] for c in cities}
res = {}
for label, d in D.items():
    boots = np.empty(5000)
    for b in range(5000):
        r = pd.concat([grp[c] for c in rng.choice(cities, size=len(cities), replace=True)], ignore_index=True)
        r = r[r['max_obs_depth'] >= d]
        boots[b] = (100 * (L(r['beta_glm_promo'], r['beta_glm_depth'], d) - L(r['beta_rec_promo'], r['beta_rec_depth'], d))).median()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    s = dc[dc['max_obs_depth'] >= d]
    point = (100 * (L(s['beta_glm_promo'], s['beta_glm_depth'], d) - L(s['beta_rec_promo'], s['beta_rec_depth'], d))).median()
    res[label] = {'point': point, 'lo': lo, 'hi': hi}
    print(f'{label} (d={d}): median G-M = {point:.2f}pp, 95% city-block CI [{lo:.2f}, {hi:.2f}]')
with open(f'{OUT}/discount_scenario_jointboot.json', 'w') as f:
    json.dump(res, f, indent=2)
