"""P1b — 프로모션 탄력성 계수 편향 (C안 Study 2 앵커 수치).
naive OLS(관측, 검열 무시) log(판매)~depth/promo vs M-모형(검열수요 회복) log-linear 계수.
재사용: 03_main (COLS, datetime_ord, build_cell, 단일패스 그룹화 패턴) + pilot/tobit (P0/P1a 검증).
산출물: elast_cells.csv, elast_summary.md, elast_DONE
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import tobit  # noqa: E402
import importlib.util
_spec = importlib.util.spec_from_file_location('m3', os.path.join(HERE, '03_main.py'))
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)

P = os.environ.get('FRN_PARQUET', '/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet')

COLS_E = ['city', 'cat', 'status', 'n_rows', 'n_promo', 'm_nit', 'm_converged', 'm_grad',
          'beta_naive_depth', 'beta_rec_depth', 'bias_pct_depth',
          'beta_naive_promo', 'beta_rec_promo', 'bias_pct_promo']
CKPT = os.path.join(HERE, 'elast_cells.csv')


def load_done():
    if not os.path.exists(CKPT):
        return set()
    done = set()
    for r in pd.read_csv(CKPT, header=None, dtype=object).itertuples(index=False):
        try:
            done.add((int(r[0]), int(r[1])))
        except Exception:
            continue
    return done


def run_cell(rows):
    """rows: build_cell 입력 dict (store-day 패널). 반환 (rec, pooled) — pooled: (depth, K, DhatM) promo 행."""
    d = m3.build_cell(rows)
    n = int(d['C'].shape[0])
    pm = d['promo'] == 1
    rec = dict(n_rows=n, n_promo=int(pm.sum()))
    if n < 1000 or int(pm.sum()) < 30 or int((~pm).sum()) < 30:
        rec['status'] = 'skip:insufficient'
        return rec, None
    fM = tobit.fit(d['X'], d['K'], d['C'], b0=tobit.init_poisson_irls(d['X'], d['K'], d['C']))
    DhatM, _ = tobit.latent_day(fM['b'], d['X'], d['K'], d['C'])
    # M-모형 log-linear 계수: X = [1, promo, depth, dow1-5, hol, act, prec, temp, hum, wind]
    b_rec_promo = float(fM['b'][1])
    b_rec_depth = float(fM['b'][2])
    # naive OLS (관측 K, 검열 무시) — M과 동일한 제어변수
    y = np.log1p(d['K'].astype(float))
    X_nd = np.delete(d['X'], 1, axis=1)  # promo 제거 → depth 계수 (위치 1)
    X_np = np.delete(d['X'], 2, axis=1)  # depth 제거 → promo 계수 (위치 1)
    b_naive_depth = float(np.linalg.lstsq(X_nd, y, rcond=None)[0][1])
    b_naive_promo = float(np.linalg.lstsq(X_np, y, rcond=None)[0][1])
    rec.update(status='ok', m_nit=int(fM['nit']), m_converged=bool(fM['converged']),
               m_grad=round(float(fM['grad_norm']), 4),
               beta_naive_depth=round(b_naive_depth, 5), beta_rec_depth=round(b_rec_depth, 5),
               bias_pct_depth=round((b_naive_depth - b_rec_depth) / max(abs(b_rec_depth), 1e-9), 4),
               beta_naive_promo=round(b_naive_promo, 5), beta_rec_promo=round(b_rec_promo, 5),
               bias_pct_promo=round((b_naive_promo - b_rec_promo) / max(abs(b_rec_promo), 1e-9), 4))
    return rec, (d['X'][:, 2][pm], d['K'][pm].astype(float), DhatM[pm])


def main():
    done = load_done()
    if not os.path.exists(CKPT):
        pd.DataFrame(columns=COLS_E).to_csv(CKPT, index=False)
    print(f'resume: {len(done)} cells already done', flush=True)

    groups = {}
    f = pq.ParquetFile(P)
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=m3.COLS)
        ci = t.column('city_id').to_numpy()
        fi = t.column('first_category_id').to_numpy()
        keys = np.stack([ci, fi], axis=1)
        dts = t.column('dt').to_pylist()
        for k in np.unique(keys, axis=0):
            m = (keys == k).all(axis=1)
            g = groups.setdefault(tuple(k), {c: [] for c in ['store', 'dtord', 'hs', 'ss', 'disc', 'hol', 'act', 'prec', 'temp', 'hum', 'wind']})
            g['store'].append(t.column('store_id').to_numpy()[m])
            g['dtord'].append(np.array([m3.datetime_ord(d) for d in dts], dtype=np.int64)[m])
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

    pooled_parts = []
    n_done = len(done)
    for key in sorted(groups):
        city, cat = key
        if (city, cat) in done:
            continue
        rows = groups[key]
        for c in rows:
            rows[c] = np.concatenate(rows[c])
        del groups[key]
        rec, pooled = run_cell(rows)
        rec['city'], rec['cat'] = city, cat
        rec = {c: rec.get(c) for c in COLS_E}
        with open(CKPT, 'a') as fh:
            pd.DataFrame([rec]).to_csv(fh, index=False, header=False)
        if pooled is not None:
            pooled_parts.append(pooled)
        n_done += 1
        print(f'[{n_done}/{len(groups) + n_done}] cell {city}x{cat}: {rec.get("status")} bias_pct_depth={rec.get("bias_pct_depth")}', flush=True)

    if pooled_parts:
        dep = np.concatenate([p[0] for p in pooled_parts])
        K = np.concatenate([p[1] for p in pooled_parts])
        Dh = np.concatenate([p[2] for p in pooled_parts])
        q = pd.qcut(dep, 5, duplicates='drop')
        tab = pd.DataFrame({'depth_q': q, 'K': K, 'D': Dh}).groupby('depth_q', observed=True).agg(
            n=('K', 'size'), naive_logK=('K', lambda s: float(np.log1p(s).mean())),
            rec_logD=('D', lambda s: float(np.log1p(s).mean())))
        tab['gap'] = tab['naive_logK'] - tab['rec_logD']
    else:
        tab = None

    df = pd.read_csv(CKPT)
    ok = df[df.status == 'ok']
    lines = ['# P1b 요약 — 탄력성 계수 편향 (naive OLS vs M-모형, 검열 회복)\n',
             f'- n_ok: {len(ok)} / {len(df)}']
    if len(ok):
        lines += [f'- med |bias_pct_depth|: {float(ok["bias_pct_depth"].abs().median()):.4f}',
                  f'- med bias_pct_depth (signed): {float(ok["bias_pct_depth"].median()):.4f}',
                  f'- frac |bias_pct_depth|>=20%: {float((ok["bias_pct_depth"].abs() >= 0.2).mean()):.4f}',
                  f'- med |bias_pct_promo|: {float(ok["bias_pct_promo"].abs().median()):.4f}']
        try:
            p1 = pd.read_csv(os.path.join(HERE, 'main_cells.csv'), usecols=['city', 'cat', 'bias_ratio'])
            j = ok.merge(p1, on=['city', 'cat'])
            if len(j):
                lines.append(f'- corr(|bias_pct_depth|, P1a bias_ratio): {float(np.corrcoef(j.bias_pct_depth.abs(), j.bias_ratio)[0, 1]):.4f} (n={len(j)})')
        except Exception as e:
            lines.append(f'- P1a 교차: 불가 ({e})')
    if tab is not None:
        lines.append('\n## 도즈-반응 (풀링, promo store-day, depth 분위별)\n')
        lines.append(tab.to_string())
    with open(os.path.join(HERE, 'elast_summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(HERE, 'elast_DONE'), 'w') as fh:
        fh.write(json.dumps({'n_ok': int(len(ok))}))
    print('DONE', open(os.path.join(HERE, 'elast_DONE')).read(), flush=True)


if __name__ == '__main__':
    main()
