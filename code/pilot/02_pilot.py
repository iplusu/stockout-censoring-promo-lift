"""P0 파일럿 메인: 셀(city 0, first_cat 4) 추출 → naive A/B, 검열 Poisson 회복 M/F, 주간 편향, go/no-go.
시드 42, 단일 프로세스, 결정론적. 실행: python 02_pilot.py
fit_state.npz 캐시가 있으면 추출+피팅을 건너뜀 (반복 실행 가속).
"""
import datetime as dtm
import json, os
import numpy as np
import pyarrow.parquet as pq

import tobit

P = '/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet'
HERE = os.path.dirname(os.path.abspath(__file__))
CITY, CAT = 0, 4
STATE = os.path.join(HERE, 'fit_state.npz')
np.random.seed(42)
WEEK0 = dtm.date(2023, 12, 25).toordinal()   # 2024 ISO 1주 월요일


def datetime_ord(s):
    y, m, d = map(int, s.split('-'))
    return dtm.date(y, m, d).toordinal()


def date_str(ordinal):
    return dtm.date.fromordinal(int(ordinal)).isoformat()


def main_tail(k_h, ST, C, K, promo, week, sd_uniq, sd_inv, n,
              fitM, fitF, DhatM, DhatF, n_stores, n_prods):
    Xnames = ['intercept', 'promo', 'depth', 'dow1', 'dow2', 'dow3', 'dow4', 'dow5',
              'holiday', 'activity', 'precpt_c', 'temp_c', 'hum_c', 'wind_c']
    # ---------- 시간 배분 (hour-of-day 프로파일, uncensored일 기준) ----------
    def apportion(Dhat):
        P = k_h[~C].mean(axis=0)                          # 24, uncensored일의 시간 형상
        L = np.maximum(Dhat - K, 0.0)
        den = (ST * P[None, :]).sum(axis=1)               # censored일: 스톡아웃시간 형상 합
        out = k_h.astype(np.float64).copy()
        good = (C == 1) & (den > 1e-9)
        out[good] += L[good, None] * (ST[good] * P[None, :] / den[good, None])
        bad = (C == 1) & (den <= 1e-9)
        if bad.any():
            nsh = ST[bad].sum(axis=1)
            out[bad] += L[bad, None] * (ST[bad] / np.maximum(nsh[:, None], 1))
        return out
    DhatM_h = apportion(DhatM)
    DhatF_h = apportion(DhatF)

    # ---------- Naive ----------
    Yf = k_h.astype(np.float64)
    pm = promo == 1
    naiveA = float(Yf[pm].mean() - Yf[~pm].mean())
    # Naive B: store-day FE OLS 계수 (within 추정자, 시간 수준)
    n_p = np.bincount(sd_inv, weights=pm, minlength=len(sd_uniq)).astype(float)
    n_np = np.bincount(sd_inv, weights=1 - pm, minlength=len(sd_uniq)).astype(float)
    sum_p = np.bincount(sd_inv, weights=np.where(pm[:, None], Yf, 0).sum(axis=1), minlength=len(sd_uniq))
    sum_np = np.bincount(sd_inv, weights=np.where(pm[:, None], 0, Yf).sum(axis=1), minlength=len(sd_uniq))
    n_all = n_p + n_np
    p_bar = n_p / np.maximum(n_all, 1)
    ybar = (sum_p + sum_np) / np.maximum(n_all, 1)
    num = (1 - p_bar) * (sum_p - n_p * ybar) - p_bar * (sum_np - n_np * ybar)
    den = (1 - p_bar) ** 2 * n_p + p_bar ** 2 * n_np
    both = (n_p > 0) & (n_np > 0)
    naiveB = float(np.sum(num[both]) / max(np.sum(den[both]), 1e-9))
    # 회복 F의 apple-to-apple: 같은 store-day FE 설계로 잠복(DhatF)의 promo 계수
    sumF_p = np.bincount(sd_inv, weights=np.where(pm[:, None], DhatF_h, 0).sum(axis=1), minlength=len(sd_uniq))
    sumF_np = np.bincount(sd_inv, weights=np.where(pm[:, None], 0, DhatF_h).sum(axis=1), minlength=len(sd_uniq))
    ybarF = (sumF_p + sumF_np) / np.maximum(n_all, 1)
    numF = (1 - p_bar) * (sumF_p - n_p * ybarF) - p_bar * (sumF_np - n_np * ybarF)
    recF_coef = float(np.sum(numF[both]) / max(np.sum(den[both]), 1e-9))
    biasB = naiveB - recF_coef

    # ---------- 주간 집계 ----------
    sd_week = (sd_uniq[:, 1] - WEEK0) // 7
    rows = []
    for w in np.unique(week):
        mw = week == w
        pmw = mw & pm
        if pmw.sum() == 0 or (mw & ~pm).sum() == 0:
            continue
        naA = float(Yf[pmw].mean() - Yf[mw & ~pm].mean())
        inw = sd_week == w
        naB_w = float(np.sum(num[inw & both]) / max(np.sum(den[inw & both]), 1e-9))
        rcM = float(DhatM_h[pmw].mean() - DhatM_h[mw & ~pm].mean())
        rcF_c = float(np.sum(numF[inw & both]) / max(np.sum(den[inw & both]), 1e-9))
        rows.append(dict(week=str(int(w)), week_start=date_str(WEEK0 + 7 * w),
                         n_days=int(mw.sum()), n_promo_days=int(pmw.sum()),
                         cens_promo=round(float(C[pmw].mean()), 4), cens_nonpromo=round(float(C[mw & ~pm].mean()), 4),
                         naiveA=round(naA, 5), naiveB=round(naB_w, 5),
                         recM_hour=round(rcM, 5), recF_coef=round(rcF_c, 5),
                         biasA=round(naA - rcM, 5), biasB=round(naB_w - rcF_c, 5)))

    biasA = np.array([r['biasA'] for r in rows])
    naA = np.array([r['naiveA'] for r in rows])
    n_weeks = len(rows)
    pos, neg = int((biasA > 0).sum()), int((biasA < 0).sum())
    frac_same_dir = max(pos, neg) / n_weeks
    med_naive = float(np.median(np.abs(naA)))
    thr = np.where(np.abs(naA) > 1e-9, 0.20 * np.abs(naA), 0.20 * med_naive)
    frac_big = float((np.abs(biasA) >= thr).mean())
    tiny_thr = np.where(np.abs(naA) > 1e-9, 0.05 * np.abs(naA), 0.05 * med_naive)
    frac_tiny = float((np.abs(biasA) < tiny_thr).mean())
    a_ok = bool(fitM['converged'])
    b_ok = bool(frac_same_dir >= 0.80)
    c_ok = bool(frac_big > 0.50)
    if a_ok and b_ok and c_ok:
        verdict = 'GO'
    elif (not b_ok) or frac_tiny >= 0.95:
        verdict = 'NO-GO'
    else:
        verdict = 'CONDITIONAL'

    # ---------- 출력 ----------
    recM_overall = float(DhatM_h[pm].mean() - DhatM_h[~pm].mean())
    rows.append(dict(week='overall', week_start='', n_days=int(n), n_promo_days=int(pm.sum()),
                     cens_promo=round(float(C[pm].mean()), 4), cens_nonpromo=round(float(C[~pm].mean()), 4),
                     naiveA=round(naiveA, 5), naiveB=round(naiveB, 5),
                     recM_hour=round(recM_overall, 5), recF_coef=round(recF_coef, 5),
                     biasA=round(naiveA - recM_overall, 5), biasB=round(biasB, 5)))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, 'pilot_results.csv'), index=False)
    df.to_parquet(os.path.join(HERE, 'pilot_results.parquet'), index=False)

    cens = C == 1
    meta = dict(
        seed=42, cell=dict(city_id=CITY, first_category_id=CAT, rows=int(n),
                           stores=int(n_stores), products=int(n_prods)),
        data_quality=dict(censored_days=int(cens.sum()),
                          censored_days_with_sales=round(float((cens & (K > 0)).sum()) / max(int(cens.sum()), 1), 4),
                          stockout_hours_with_sales=round(float((ST & (k_h > 0)).sum()) / max(int(ST.sum()), 1), 4)),
        model_M=dict(params=dict(zip(Xnames, [round(float(v), 5) for v in fitM['b']])),
                     ll=round(fitM['ll'], 1), nit=fitM['nit'], converged=bool(fitM['converged']),
                     grad_norm=round(fitM['grad_norm'], 6)),
        model_F=dict(params=dict(zip(Xnames, [round(float(v), 5) for v in fitF['b']])),
                     ll=round(fitF['ll'], 1), nit=fitF['nit'], converged=bool(fitF['converged']),
                     n_store_day_groups=fitF['n_groups']),
        headline=dict(naiveA=round(naiveA, 5), naiveB=round(naiveB, 5),
                      recM_day_beta_promo=round(float(fitM['b'][Xnames.index('promo')]), 5),
                      recF_day_beta_promo=round(float(fitF['b'][Xnames.index('promo')]), 5),
                      recM_hour=round(recM_overall, 5), recF_coef=round(recF_coef, 5),
                      biasA=round(naiveA - recM_overall, 5), biasB=round(biasB, 5)),
        gate=dict(n_weeks=n_weeks, weeks_same_dir=[pos, neg], frac_same_dir=round(frac_same_dir, 4),
                  frac_big=round(frac_big, 4), frac_tiny=round(frac_tiny, 4),
                  a_convergence=a_ok, b_dir=b_ok, c_size=c_ok, verdict=verdict),
    )
    with open(os.path.join(HERE, 'results_meta.json'), 'w') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(df.to_string(index=False))


if os.path.exists(STATE):
    print('loading fit state cache — skipping extract+fit')
    z = np.load(STATE)
    fitM = dict(b=z['bM'], ll=float(z['llM']), nit=int(z['nitM']), converged=bool(z['convM']),
                grad_norm=float(z['gnM']))
    fitF = dict(b=z['bF'], ll=float(z['llF']), nit=int(z['nitF']), converged=bool(z['convF']),
                alpha=z['alphaF'], n_groups=int(z['ngF']))
    main_tail(k_h=z['k_h'], ST=z['ST'], C=z['C'], K=z['K'], promo=z['promo'], week=z['week'],
              sd_uniq=z['sd_uniq'], sd_inv=z['sd_inv'], n=int(z['n']),
              fitM=fitM, fitF=fitF, DhatM=z['DhatM'], DhatF=z['DhatF'],
              n_stores=int(z['n_stores']), n_prods=int(z['n_prods']))
else:
    # ---------- 1) 셀 추출 (row group 루프, 셀 행만) ----------
    buf = {k: [] for k in ['store', 'prod', 'dtord', 'hs', 'ss', 'disc', 'hol', 'act', 'prec', 'temp', 'hum', 'wind']}
    cnt_bad = 0
    f = pq.ParquetFile(P)
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=['city_id', 'first_category_id', 'store_id', 'product_id', 'dt',
                                         'hours_sale', 'hours_stock_status', 'stock_hour6_22_cnt',
                                         'discount', 'holiday_flag', 'activity_flag',
                                         'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level'])
        m = (np.asarray(t.column('city_id').to_numpy()) == CITY) & (np.asarray(t.column('first_category_id').to_numpy()) == CAT)
        if not m.any():
            continue
        dts = t.column('dt').to_pylist()
        ss_i = np.asarray(t.column('hours_stock_status').combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.int8).reshape(-1, 24)[m]
        cnt = np.asarray(t.column('stock_hour6_22_cnt').to_numpy())[m]
        cnt_bad += int((cnt != (ss_i == 1)[:, 6:22].sum(axis=1)).sum())
        buf['store'].append(np.asarray(t.column('store_id').to_numpy())[m])
        buf['prod'].append(np.asarray(t.column('product_id').to_numpy())[m])
        buf['dtord'].append(np.array([datetime_ord(d) for d in dts], dtype=np.int64)[m])
        buf['hs'].append(np.asarray(t.column('hours_sale').combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 24)[m])
        buf['ss'].append(ss_i)
        buf['disc'].append(np.asarray(t.column('discount').to_numpy())[m])
        buf['hol'].append(np.asarray(t.column('holiday_flag').to_numpy())[m])
        buf['act'].append(np.asarray(t.column('activity_flag').to_numpy())[m])
        buf['prec'].append(np.asarray(t.column('precpt').to_numpy())[m])
        buf['temp'].append(np.asarray(t.column('avg_temperature').to_numpy())[m])
        buf['hum'].append(np.asarray(t.column('avg_humidity').to_numpy())[m])
        buf['wind'].append(np.asarray(t.column('avg_wind_level').to_numpy())[m])
    for k in buf:
        buf[k] = np.concatenate(buf[k])
    n = len(buf['store'])
    print(f'cell rows: {n}, cnt check mismatches: {cnt_bad}')
    assert cnt_bad == 0, 'stock_hour6_22_cnt와 스톡아웃(6..21시) 불일치 — 의미 재확인 필요'

    # ---------- 2) 패널 구성 ----------
    k_h = np.rint(buf['hs'] / 0.1).astype(np.int32)      # 시간별 판매 (count, 1 = 0.1 판매)
    ST = (buf['ss'] == 1)                                # 스톡아웃 시간
    C = ST.any(axis=1).astype(np.int8)                    # 일 검열 지시자
    K = k_h.sum(axis=1).astype(np.int64)                  # 일 판매 count
    promo = (buf['disc'] < 1.0 - 1e-9).astype(np.int8)
    depth = np.where(promo == 1, 1.0 - buf['disc'], 0.0)
    dow = (buf['dtord'] + 3) % 7                          # Mon=0
    week = (buf['dtord'] - WEEK0) // 7
    sd_uniq, sd_inv = np.unique(np.stack([buf['store'], buf['dtord']], axis=1), axis=0, return_inverse=True)

    med = {nm: float(np.median(buf[nm])) for nm in ['prec', 'temp', 'hum', 'wind']}
    dow_d = [(dow == d).astype(float) for d in range(1, 6)]
    X = np.column_stack([np.ones(n), promo.astype(float), depth, *dow_d,
                         buf['hol'].astype(float), buf['act'].astype(float),
                         buf['prec'] - med['prec'], buf['temp'] - med['temp'],
                         buf['hum'] - med['hum'], buf['wind'] - med['wind']])

    # ---------- 3) 회복 M (primary, global design) ----------
    b0 = tobit.init_poisson_irls(X, K, C)
    fitM = tobit.fit(X, K, C, b0=b0)
    DhatM, _ = tobit.latent_day(fitM['b'], X, K, C)

    # ---------- 4) 회복 F (store-day FE, robustness) ----------
    fitF = tobit.fit_fe(X, K, C, sd_inv, max_iter=25)
    DhatF, _ = tobit.latent_day(fitF['b'], X, K, C, offset=fitF['alpha'][sd_inv])

    np.savez(STATE, k_h=k_h, ST=ST, C=C, K=K, promo=promo, week=week,
             sd_uniq=sd_uniq, sd_inv=sd_inv, n=np.int64(n),
             bM=fitM['b'], llM=fitM['ll'], nitM=fitM['nit'], convM=fitM['converged'], gnM=fitM['grad_norm'],
             bF=fitF['b'], llF=fitF['ll'], nitF=fitF['nit'], convF=fitF['converged'],
             alphaF=fitF['alpha'], ngF=fitF['n_groups'],
             DhatM=DhatM, DhatF=DhatF,
             n_stores=np.int64(len(np.unique(buf['store']))), n_prods=np.int64(len(np.unique(buf['prod']))))
    print('fit state cached ->', STATE)
    main_tail(k_h=k_h, ST=ST, C=C, K=K, promo=promo, week=week,
              sd_uniq=sd_uniq, sd_inv=sd_inv, n=n,
              fitM=fitM, fitF=fitF, DhatM=DhatM, DhatF=DhatF,
              n_stores=len(np.unique(buf['store'])), n_prods=len(np.unique(buf['prod'])))
