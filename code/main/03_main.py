"""P1 메인: 전 셀(397) 편향 정량화 — 셀별 M(글로벌) 회복 + naive A/B, 시간대 프로파일, 가용성 변이 진단.
F(store-day FE 회복)는 n_both 상위 20셀만 (robustness). 단일 패스 추출, 셀별 체크포인트(재실행 시 완료 셀 스킵).
시드 42, 결정론적. 산출물: main_cells.csv, main_hourly.parquet, main_topF.csv, main_summary.md, DONE
"""
import datetime as dtm
import json, os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import tobit  # noqa: E402

P = os.environ.get('FRN_PARQUET', '/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet')
COLS = ['city_id', 'first_category_id', 'store_id', 'product_id', 'dt',
        'hours_sale', 'hours_stock_status', 'stock_hour6_22_cnt', 'discount',
        'holiday_flag', 'activity_flag', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
WEEK0 = dtm.date(2023, 12, 25).toordinal()
XNAMES = ['intercept', 'promo', 'depth', 'dow1', 'dow2', 'dow3', 'dow4', 'dow5',
          'holiday', 'activity', 'precpt_c', 'temp_c', 'hum_c', 'wind_c']
np.random.seed(42)


def datetime_ord(s):
    y, m, d = map(int, s.split('-'))
    return dtm.date(y, m, d).toordinal()


def apportion(k_h, ST, C, K, Dhat):
    """잠복 수요를 스톡아웃 시간에 hour-of-day 프로파일(uncensored일)로 분배."""
    P = k_h[~C].mean(axis=0)
    L = np.maximum(Dhat - K, 0.0)
    den = (ST * P[None, :]).sum(axis=1)
    out = k_h.astype(np.float64).copy()
    good = (C == 1) & (den > 1e-9)
    out[good] += L[good, None] * (ST[good] * P[None, :] / den[good, None])
    bad = (C == 1) & (den <= 1e-9)
    if bad.any():
        nsh = ST[bad].sum(axis=1)
        out[bad] += L[bad, None] * (ST[bad] / np.maximum(nsh[:, None], 1))
    return out


def build_cell(rows):
    """rows: 각 컬럼의 numpy/리스트. 셀 패널 구조 반환 (파일럿 02_pilot.py §1-2 재사용)."""
    k_h = np.rint(rows['hs'] / 0.1).astype(np.int32)
    ST = (rows['ss'] == 1)
    C = ST.any(axis=1).astype(np.int8)
    K = k_h.sum(axis=1).astype(np.int64)
    promo = (rows['disc'] < 1.0 - 1e-9).astype(np.int8)
    depth = np.where(promo == 1, 1.0 - rows['disc'], 0.0)
    dow = (rows['dtord'] + 3) % 7
    week = (rows['dtord'] - WEEK0) // 7
    sd_uniq, sd_inv = np.unique(np.stack([rows['store'], rows['dtord']], axis=1), axis=0, return_inverse=True)
    med = {nm: float(np.median(rows[nm])) for nm in ['prec', 'temp', 'hum', 'wind']}
    dow_d = [(dow == d).astype(float) for d in range(1, 6)]
    X = np.column_stack([np.ones(len(rows['store'])), promo.astype(float), depth, *dow_d,
                         rows['hol'].astype(float), rows['act'].astype(float),
                         rows['prec'] - med['prec'], rows['temp'] - med['temp'],
                         rows['hum'] - med['hum'], rows['wind'] - med['wind']])
    return dict(k_h=k_h, ST=ST, C=C, K=K, promo=promo, week=week, X=X,
                sd_uniq=sd_uniq, sd_inv=sd_inv)


def fit_cell(d, run_F):
    """셀별 추정. 반환: 셀 요약 dict + 시간대 프로필 배열."""
    n = int(d['C'].shape[0])
    pm = d['promo'] == 1
    rec = dict(n_rows=n, n_days=int(len(np.unique(d['sd_uniq'][:, 1]))),
               n_stores=int(len(np.unique(d['sd_uniq'][:, 0]))),
               promo_share=round(float(pm.mean()), 4),
               stockout_day_share=round(float(d['C'].mean()), 4),
               both_share=round(float((pm & (d['C'] == 1)).mean()), 4))
    if n < 1000 or int(pm.sum()) < 30 or int((~pm).sum()) < 30:
        rec['status'] = 'skip:insufficient'
        return rec, None
    b0 = tobit.init_poisson_irls(d['X'], d['K'], d['C'])
    fM = tobit.fit(d['X'], d['K'], d['C'], b0=b0)
    DhatM, _ = tobit.latent_day(fM['b'], d['X'], d['K'], d['C'])
    DhatM_h = apportion(d['k_h'], d['ST'], d['C'], d['K'], DhatM)
    Yf = d['k_h'].astype(np.float64)
    naiveA = float(Yf[pm].mean() - Yf[~pm].mean())
    recA = float(DhatM_h[pm].mean() - DhatM_h[~pm].mean())
    rec.update(status='ok', m_nit=fM['nit'], m_converged=bool(fM['converged']),
               m_grad=round(fM['grad_norm'], 4),
               promo_beta_M=round(float(fM['b'][1]), 5),
               naiveA=round(naiveA, 5), recA=round(recA, 5), biasA=round(naiveA - recA, 5),
               bias_ratio=round(abs(naiveA - recA) / max(abs(naiveA), 1e-9), 4))
    # 시간대 프로필 (promo vs non-promo 시간평균, naive vs 회복)
    naive_h = Yf[pm].mean(axis=0) - Yf[~pm].mean(axis=0)
    rec_h = DhatM_h[pm].mean(axis=0) - DhatM_h[~pm].mean(axis=0)
    hourly = dict(city=rec.get('city'), cat=rec.get('cat'),
                  naive_h=naive_h, rec_h=rec_h, bias_h=naive_h - rec_h)
    if run_F:
        fF = tobit.fit_fe(d['X'], d['K'], d['C'], d['sd_inv'], max_iter=25)
        DhatF_h = apportion(d['k_h'], d['ST'], d['C'], d['K'], tobit.latent_day(fF['b'], d['X'], d['K'], d['C'], offset=fF['alpha'][d['sd_inv']])[0])
        sdu, sdi = d['sd_uniq'], d['sd_inv']
        n_p = np.bincount(sdi, weights=pm, minlength=len(sdu)).astype(float)
        n_np = np.bincount(sdi, weights=1 - pm, minlength=len(sdu)).astype(float)
        sp = np.bincount(sdi, weights=np.where(pm[:, None], Yf, 0).sum(axis=1), minlength=len(sdu))
        snp = np.bincount(sdi, weights=np.where(pm[:, None], 0, Yf).sum(axis=1), minlength=len(sdu))
        n_all = n_p + n_np
        p_bar = n_p / np.maximum(n_all, 1)
        ybar = (sp + snp) / np.maximum(n_all, 1)
        num = (1 - p_bar) * (sp - n_p * ybar) - p_bar * (snp - n_np * ybar)
        den = (1 - p_bar) ** 2 * n_p + p_bar ** 2 * n_np
        both = (n_p > 0) & (n_np > 0)
        naiveB = float(np.sum(num[both]) / max(np.sum(den[both]), 1e-9))
        sf_p = np.bincount(sdi, weights=np.where(pm[:, None], DhatF_h, 0).sum(axis=1), minlength=len(sdu))
        sf_np = np.bincount(sdi, weights=np.where(pm[:, None], 0, DhatF_h).sum(axis=1), minlength=len(sdu))
        ybarF = (sf_p + sf_np) / np.maximum(n_all, 1)
        numF = (1 - p_bar) * (sf_p - n_p * ybarF) - p_bar * (sf_np - n_np * ybarF)
        recF_c = float(np.sum(numF[both]) / max(np.sum(den[both]), 1e-9))
        rec.update(naiveB=round(naiveB, 5), recF=round(recF_c, 5), biasB=round(naiveB - recF_c, 5),
                   f_nit=fF['nit'], f_converged=bool(fF['converged']), n_groups=fF['n_groups'])
    # 가용성 변이 진단 (store-day 수준: promo vs 스톡아웃 share)
    sdu, sdi = d['sd_uniq'], d['sd_inv']
    so_sd = np.bincount(sdi, weights=d['ST'].sum(axis=1), minlength=len(sdu)) / 24.0
    promo_sd = np.bincount(sdi, weights=pm, minlength=len(sdu)) / np.maximum(np.bincount(sdi, minlength=len(sdu)), 1)
    rec.update(so_share_promo=round(float(so_sd[promo_sd == 1].mean()), 4) if (promo_sd == 1).any() else None,
               so_share_nonpromo=round(float(so_sd[promo_sd == 0].mean()), 4) if (promo_sd == 0).any() else None,
               corr_promo_so=round(float(np.corrcoef(promo_sd, so_sd)[0, 1]), 4))
    return rec, hourly


CKPT_COLS = ['n_rows', 'n_days', 'n_stores', 'promo_share', 'stockout_day_share', 'both_share',
             'status', 'm_nit', 'm_converged', 'm_grad', 'promo_beta_M', 'naiveA', 'recA',
             'biasA', 'bias_ratio', 'so_share_promo', 'so_share_nonpromo', 'corr_promo_so',
             'city', 'cat']
CKPT_COLS_F = CKPT_COLS + ['f_nit', 'f_converged', 'f_naiveA', 'f_recA', 'f_biasA', 'f_bias_ratio']


def normalize_ckpt(path):
    # 한 스키마(26열)로 통일: 옛 20열 행은 f_* 컬럼 비워 채운다
    with open(path) as fh:
        lines = [l.rstrip('\n') for l in fh if l.strip()]
    if len(lines) < 2:
        return
    if all(len(l.split(',')) == 26 for l in lines[1:]):
        return
    rows = []
    for l in lines[1:]:
        f = l.split(',')
        rows.append(f[:26] + [''] * (26 - len(f)))
    pd.DataFrame(rows, columns=CKPT_COLS_F).to_csv(path, index=False)


def load_done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    # city,cat은 26열 스키마에서 고정 인덱스 18,19 (f_* 컬럼이 뒤에 있어 '마지막 두 열'이 아님)
    for r in pd.read_csv(path, header=None, dtype=object).itertuples(index=False):
        try:
            done.add((int(r[18]), int(r[19])))
        except Exception:
            continue
    return done


def append_ckpt(rec, path):
    row = {c: rec.get(c) for c in CKPT_COLS_F}  # 항상 통일 스키마 26열 (혼합 너비 → read_csv 크래시 방지)
    with open(path, 'a') as fh:
        pd.DataFrame([row]).to_csv(fh, index=False, header=False)


def main():
    cells_csv = os.path.join(HERE, 'main_cells.csv')
    if os.path.exists(cells_csv):
        normalize_ckpt(cells_csv)
    done = load_done(cells_csv)
    # 단일 패스: 셀로 그룹화
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
    topF = set(pd.read_csv(os.path.join(os.path.dirname(HERE), 'pilot', 'cell_selection.csv'))
               .nlargest(20, 'both')[['city_id', 'first_category_id']].itertuples(index=False, name=None))
    topF = {(int(a), int(b)) for a, b in topF}
    hourly_rows = []
    tF_rows = []
    n_ok = 0
    for key in sorted(groups):
        city, cat = key
        if key in done:
            continue
        rows = groups[key]
        for k in rows:
            rows[k] = np.concatenate(rows[k])
        d = build_cell(rows)
        rec, hourly = fit_cell(d, run_F=(key in topF))
        rec['city'], rec['cat'] = city, cat
        if hourly is not None:
            hourly_rows.append({**{f'naive_h{h}': round(float(v), 5) for h, v in enumerate(hourly['naive_h'])},
                                **{f'rec_h{h}': round(float(v), 5) for h, v in enumerate(hourly['rec_h'])},
                                **{f'bias_h{h}': round(float(v), 5) for h, v in enumerate(hourly['bias_h'])},
                                'city': city, 'cat': cat})
            if 'biasB' in rec:
                tF_rows.append({k: rec[k] for k in ['city', 'cat', 'n_rows', 'promo_share', 'stockout_day_share', 'both_share',
                                                    'naiveA', 'recA', 'biasA', 'naiveB', 'recF', 'biasB',
                                                    'promo_beta_M', 'm_nit', 'f_nit', 'n_groups',
                                                    'so_share_promo', 'so_share_nonpromo', 'corr_promo_so'] if k in rec})
        append_ckpt(rec, cells_csv)
        del groups[key]
        n_ok += 1
        print(f'[{n_ok}/{len(groups)}] cell {city}x{cat}: {rec.get("status")} rows={rec.get("n_rows")} biasA={rec.get("biasA")} ratio={rec.get("bias_ratio")}', flush=True)
    if hourly_rows:
        pd.DataFrame(hourly_rows).to_parquet(os.path.join(HERE, 'main_hourly.parquet'), index=False)
    if tF_rows:
        pd.DataFrame(tF_rows).to_csv(os.path.join(HERE, 'main_topF.csv'), index=False)
    # 요약
    df = pd.read_csv(cells_csv)
    ok = df[df['status'] == 'ok']
    r = dict(n_cells_total=int(len(df)), n_ok=int(len(ok)),
             med_bias_ratio=float(np.median(ok['bias_ratio'])) if len(ok) else None,
             mean_bias_ratio=float(np.mean(ok['bias_ratio'])) if len(ok) else None,
             frac_neg_naive=float((ok['naiveA'] < 0).mean()) if len(ok) else None,
             corr_promo_so_median=float(np.median(ok['corr_promo_so'].dropna())) if 'corr_promo_so' in ok else None)
    with open(os.path.join(HERE, 'main_summary.md'), 'w') as fh:
        fh.write('# P1 요약 — 전 셀 편향 정량화\n\n')
        for k, v in r.items():
            fh.write(f'- {k}: {v}\n')
        fh.write(f'\n세부: main_cells.csv (셀별), main_hourly.parquet (시간대 프로필), main_topF.csv (상위 20셀 F robustness)\n')
    with open(os.path.join(HERE, 'DONE'), 'w') as fh:
        fh.write(json.dumps(r))
    print('DONE', json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
