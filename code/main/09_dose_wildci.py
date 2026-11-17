# -*- coding: utf-8 -*-
"""09_dose_wildci.py — §3.5 dose-response quintile gap + wild-cluster (city) 95% CI.

Spec (pre-fixed, from tex §3.5): pooled promotion store-days across all 254 cells,
grouped by discount-depth quintile; gap = mean log1p(observed K) − mean log1p(recovered D̂).
Wild CI: Rademacher weights on the 18 cities, 2000 replicates, percentile interval.
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import tobit  # noqa: E402

_src = open(os.path.join(HERE, '03_main.py')).read()
_src = _src.replace("if __name__ == '__main__':\n    main()", '')
_ns = {'__name__': 'm03', '__file__': os.path.join(HERE, '03_main.py')}
exec(compile(_src, '03_main.py', 'exec'), _ns)
COLS = _ns['COLS']
P = os.environ.get('FRN_PARQUET', '/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet')
datetime_ord = _ns['datetime_ord']
build_cell = _ns['build_cell']

groups = {}
f = pq.ParquetFile(P)
for i in range(f.metadata.num_row_groups):
    t = f.read_row_group(i, columns=COLS)
    ci = t.column('city_id').to_numpy()
    fi = t.column('first_category_id').to_numpy()
    keys = np.stack([ci, fi], axis=1)
    for k in np.unique(keys, axis=0):
        m = (keys == k).all(axis=1)
        g = groups.setdefault(tuple(k), {c: [] for c in ['store', 'prod', 'dtord', 'hs', 'ss', 'disc', 'hol', 'act', 'prec', 'temp', 'hum', 'wind']})
        dts = t.column('dt').to_pylist()
        g['store'].append(t.column('store_id').to_numpy()[m])
        g['prod'].append(t.column('product_id').to_numpy()[m])
        g['dtord'].append(np.array([datetime_ord(d) for d in dts], dtype=np.int64)[m])
        g['hs'].append(np.asarray(t.column('hours_sale').combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 24)[m])
        g['ss'].append(np.asarray(t.column('hours_stock_status').combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.int8).reshape(-1, 24)[m])
        g['disc'].append(t.column('discount').to_numpy()[m])
        g['hol'].append(t.column('holiday_flag').to_numpy()[m])
        g['act'].append(t.column('activity_flag').to_numpy()[m])
        g['prec'].append(t.column('precpt').to_numpy()[m])
        g['temp'].append(t.column('avg_temperature').to_numpy()[m])
        g['hum'].append(t.column('avg_humidity').to_numpy()[m])
        g['wind'].append(t.column('avg_wind_level').to_numpy()[m])
print(f'cells found: {len(groups)}', flush=True)

D_list, K_list, DP_list, C_list = [], [], [], []
n_ok = n_skip = 0
order = sorted(groups)
for key in order:
    city, cat = key
    rows = groups[key]
    for k in rows:
        rows[k] = np.concatenate(rows[k])
    d = build_cell(rows)
    n = int(d['C'].shape[0])
    pm = d['promo'] == 1
    if n < 1000 or int(pm.sum()) < 30 or int((~pm).sum()) < 30:
        n_skip += 1
        del groups[key]
        continue
    b0 = tobit.init_poisson_irls(d['X'], d['K'], d['C'])
    fM = tobit.fit(d['X'], d['K'], d['C'], b0=b0)
    Dhat, _ = tobit.latent_day(fM['b'], d['X'], d['K'], d['C'])
    depth = np.where(pm, 1.0 - rows['disc'], 0.0)
    D_list.append(Dhat[pm])
    K_list.append(d['K'][pm])
    DP_list.append(depth[pm])
    C_list.append(np.full(int(pm.sum()), int(city), dtype=np.int32))
    n_ok += 1
    del groups[key]
    if n_ok % 25 == 0:
        print(f'[{n_ok}/{len(order)}] ok={n_ok} skip={n_skip}', flush=True)
print(f'fits done: ok={n_ok} skip={n_skip}', flush=True)

depth = np.concatenate(DP_list)
Kp = np.concatenate(K_list)
Dp = np.concatenate(D_list)
Cp = np.concatenate(C_list)
bnd = np.quantile(depth, [0.2, 0.4, 0.6, 0.8])
print(f'n_pool={len(depth)} bounds={np.round(bnd, 4)}', flush=True)

qs = np.digitize(depth, bnd)
logK = np.log1p(Kp.astype(np.float64))
logD = np.log1p(Dp.astype(np.float64))
rows_out = []
for q in range(5):
    m = qs == q
    rows_out.append(dict(quintile=q + 1, n=int(m.sum()),
                         naive=round(float(logK[m].mean()), 3),
                         recovered=round(float(logD[m].mean()), 3),
                         gap=round(float(logK[m].mean() - logD[m].mean()), 3)))
df_pt = pd.DataFrame(rows_out)
print(df_pt.to_string(index=False), flush=True)

# ponytail: standard wild bootstrap for a weighted mean — Rademacher on the
# numerator only, fixed positive denominator (sum of city counts).
cities = np.unique(Cp)
print(f'cities: {len(cities)}', flush=True)
ci_inv = np.searchsorted(cities, Cp)
y = logK - logD  # per-day gap
N_cq = np.zeros((len(cities), 5), dtype=np.int64)
S_cq = np.zeros((len(cities), 5))
for q in range(5):
    m = qs == q
    idx = ci_inv[m]
    N_cq[:, q] = np.bincount(idx, minlength=len(cities))
    S_cq[:, q] = np.bincount(idx, weights=y[m], minlength=len(cities))

rng = np.random.default_rng(42)
R = 2000
gaps = np.zeros((R, 5))
den_q = N_cq.sum(axis=0).astype(float)
W = rng.choice(np.array([-1.0, 1.0]), size=(R, len(cities)))
for q in range(5):
    gaps[:, q] = (W @ S_cq[:, q]) / den_q[q]
lo = np.percentile(gaps, 2.5, axis=0)
hi = np.percentile(gaps, 97.5, axis=0)
out = df_pt.copy()
out['lo'] = np.round(lo, 3)
out['hi'] = np.round(hi, 3)
print(out.to_string(index=False), flush=True)
out.to_csv(os.path.join(HERE, 'pending_final_dose_wildci.csv'), index=False)
print('DONE', flush=True)
