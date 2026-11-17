"""P0 파일럿 셀 기계적 선택: (city_id, first_category_id) 중 promo x stockout 중복(n_both) 최대.
train.parquet 전량 4500 row group 스캔, 5열만 읽음. 산출: pilot/cell_selection.csv
"""
import numpy as np, pyarrow.parquet as pq

P = '/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet'
f = pq.ParquetFile(P)
# key packing: cat < 1000 확인 (스케마 샘플 0..30); city*1000+cat
sums = {}  # packed -> [rows, promo, stock, both]
gmin = {}
gmax = {}
CATBASE = 1000
for i in range(f.metadata.num_row_groups):
    t = f.read_row_group(i, columns=['city_id', 'first_category_id', 'discount', 'hours_stock_status', 'dt'])
    city = np.asarray(t.column('city_id').to_numpy(), dtype=np.int64)
    cat = np.asarray(t.column('first_category_id').to_numpy(), dtype=np.int64)
    assert cat.max() < CATBASE
    disc = np.asarray(t.column('discount').to_numpy(), dtype=np.float64)
    status = np.asarray(t.column('hours_stock_status').combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.int8)
    status = status.reshape(-1, 24)
    assert status.shape[0] == len(city)
    promo = (disc < 1.0 - 1e-9)
    stock = (status == 1).any(axis=1)
    packed = city * CATBASE + cat
    order = np.argsort(packed, kind='stable')
    pk = packed[order]
    pr = promo[order].astype(np.int64)
    sk = stock[order].astype(np.int64)
    both = (pr & sk).astype(np.int64)
    uniq, inv = np.unique(pk, return_inverse=True)
    cnt = np.bincount(inv)
    cpr = np.bincount(inv, weights=pr)
    csk = np.bincount(inv, weights=sk)
    cbt = np.bincount(inv, weights=both)
    dts = t.column('dt').to_pylist()
    gmn, gmx = min(dts), max(dts)
    for u, c, cp, cs, cb in zip(uniq, cnt, cpr, csk, cbt):
        u = int(u)
        if u in sums:
            sums[u] = [sums[u][0] + c, sums[u][1] + cp, sums[u][2] + cs, sums[u][3] + cb]
            gmin[u] = min(gmin[u], gmn)
            gmax[u] = max(gmax[u], gmx)
        else:
            sums[u] = [c, cp, cs, cb]
            gmin[u] = gmn
            gmax[u] = gmx

rows = []
for u, a in sums.items():
    rows.append(dict(city_id=u // CATBASE, first_category_id=u % CATBASE,
                     rows=int(a[0]), promo=int(a[1]), stockout=int(a[2]), both=int(a[3]),
                     promo_rate=round(a[1] / a[0], 4), both_rate=round(a[3] / a[0], 4),
                     dt_min=gmin[u], dt_max=gmax[u]))
rows.sort(key=lambda r: r['both'], reverse=True)
import csv, os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cell_selection.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'cells={len(rows)} -> {out}')
for r in rows[:10]:
    print(r)
best = rows[0]
print('CHOSEN:', best['city_id'], best['first_category_id'], 'both=', best['both'], 'rows=', best['rows'],
      'dt:', best['dt_min'], '..', best['dt_max'])
print('global dt range:', min(r['dt_min'] for r in rows), '..', max(r['dt_max'] for r in rows))
