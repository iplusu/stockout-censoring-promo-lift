"""P2 decomposition — separate censoring-correction effect from covariate-omission
and distributional/link-function confounds in the P1b naive-vs-M elasticity bias
(bot-debate agreed 4-part decomposition, see
~/.hermes/shared/bot_debates/jams-c-journal-topic-20261029/round3_summary.md).

Reuses: 03_main.build_cell (panel construction), pilot/tobit (not refit here —
M-model betas are already in elast_cells.csv, no need to re-run the expensive
tobit MLE). Only two NEW model fits per cell:
  (a) full-spec naive OLS: log1p(K) ~ FULL X (promo AND depth both included)
  (c) plain Poisson GLM (log link, IRLS, no censoring correction) on observed K, full X
(b) is arithmetic: reduced-spec naive (elast_cells.csv) vs full-spec naive (a).

Output: decomp_cells.csv (one row per eligible cell), decomp_report.md.
Checkpointed (append mode) every cell -- same crash-recovery pattern as 04_elasticity.py.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import importlib.util
_spec = importlib.util.spec_from_file_location('m3', os.path.join(HERE, '03_main.py'))
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)

P = os.environ.get('FRN_PARQUET', '/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet')
ELAST_CSV = os.path.join(HERE, 'elast_cells.csv')
CKPT = os.path.join(HERE, 'decomp_cells.csv')
EPOCH_ORD = 719163  # date(1970,1,1).toordinal()

COLS_D = ['city', 'cat', 'n_rows', 'n_promo',
          'beta_naive_reduced_depth', 'beta_naive_reduced_promo',
          'beta_naive_full_depth', 'beta_naive_full_promo',
          'beta_glm_depth', 'beta_glm_promo',
          'beta_rec_depth', 'beta_rec_promo',
          'bias_pct_orig_depth', 'bias_pct_orig_promo',
          'bias_pct_a_depth', 'bias_pct_a_promo',
          'bias_pct_omission_depth', 'bias_pct_omission_promo',
          'bias_pct_dist_depth', 'bias_pct_dist_promo',
          'bias_pct_pure_censoring_depth', 'bias_pct_pure_censoring_promo',
          'glm_nit', 'glm_converged']


def poisson_glm_irls(X, y, n_iter=100, tol=1e-10):
    """Plain Poisson GLM (log link), no censoring awareness -- fit on observed y directly."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(float(y.mean()) + 0.3, 1e-3))
    converged = False
    it = 0
    for it in range(1, n_iter + 1):
        eta = np.clip(X @ b, -30, 30)
        lam = np.exp(eta)
        w = np.maximum(lam, 1e-8)
        z = eta + (y - lam) / w
        XtW = X.T @ (w[:, None] * X)
        try:
            b_new = np.linalg.solve(XtW + 1e-10 * np.eye(X.shape[1]), X.T @ (w * z))
        except np.linalg.LinAlgError:
            break
        delta = np.linalg.norm(b_new - b)
        b = b_new
        if delta < tol * max(1.0, np.linalg.norm(b)):
            converged = True
            break
    return b, it, converged


def bias_pct(num, den):
    return float(num / max(abs(den), 1e-9))


def run_cell(rows, orig):
    """orig: dict with beta_naive_reduced_depth/promo, beta_rec_depth/promo, bias_pct_orig_depth/promo (from elast_cells.csv)."""
    d = m3.build_cell(rows)
    X, K = d['X'], d['K'].astype(float)
    n = int(X.shape[0])
    pm = d['promo'] == 1

    # (a) full-spec naive OLS: log1p(K) ~ full X (promo AND depth both present)
    y = np.log1p(K)
    b_full = np.linalg.lstsq(X, y, rcond=None)[0]
    b_naive_full_promo, b_naive_full_depth = float(b_full[1]), float(b_full[2])

    # (c) plain Poisson GLM, log link, IRLS, NO censoring correction, full X, observed K
    b_glm, glm_nit, glm_conv = poisson_glm_irls(X, K)
    b_glm_promo, b_glm_depth = float(b_glm[1]), float(b_glm[2])

    b_rec_depth, b_rec_promo = orig['beta_rec_depth'], orig['beta_rec_promo']

    rec = dict(n_rows=n, n_promo=int(pm.sum()),
               beta_naive_reduced_depth=orig['beta_naive_depth'],
               beta_naive_reduced_promo=orig['beta_naive_promo'],
               beta_naive_full_depth=round(b_naive_full_depth, 5),
               beta_naive_full_promo=round(b_naive_full_promo, 5),
               beta_glm_depth=round(b_glm_depth, 5), beta_glm_promo=round(b_glm_promo, 5),
               beta_rec_depth=b_rec_depth, beta_rec_promo=b_rec_promo,
               bias_pct_orig_depth=orig['bias_pct_depth'], bias_pct_orig_promo=orig['bias_pct_promo'],
               glm_nit=int(glm_nit), glm_converged=bool(glm_conv))

    # (a) full-spec naive vs M: censoring + distributional contribution, covariates now matched
    rec['bias_pct_a_depth'] = round(bias_pct(b_naive_full_depth - b_rec_depth, b_rec_depth), 4)
    rec['bias_pct_a_promo'] = round(bias_pct(b_naive_full_promo - b_rec_promo, b_rec_promo), 4)
    # (b) covariate-omission contribution = orig headline bias_pct minus (a), exact arithmetic
    #     = (beta_naive_reduced - beta_naive_full) / |beta_rec|
    rec['bias_pct_omission_depth'] = round(rec['bias_pct_orig_depth'] - rec['bias_pct_a_depth'], 4)
    rec['bias_pct_omission_promo'] = round(rec['bias_pct_orig_promo'] - rec['bias_pct_a_promo'], 4)
    # (c) distributional/link contribution: full-spec OLS-log1p vs full-spec Poisson-GLM (both uncensored, same X)
    rec['bias_pct_dist_depth'] = round(bias_pct(b_naive_full_depth - b_glm_depth, b_glm_depth), 4)
    rec['bias_pct_dist_promo'] = round(bias_pct(b_naive_full_promo - b_glm_promo, b_glm_promo), 4)
    # (c) PURE censoring-correction effect: GLM (no censoring) vs M (censoring-corrected), same X, same link/distribution
    rec['bias_pct_pure_censoring_depth'] = round(bias_pct(b_glm_depth - b_rec_depth, b_rec_depth), 4)
    rec['bias_pct_pure_censoring_promo'] = round(bias_pct(b_glm_promo - b_rec_promo, b_rec_promo), 4)
    return rec


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


def build_groups(eligible_keys):
    """Single-pass vectorized panel build (whole table fits comfortably in RAM -- 4.5M rows).
    Faster than the 4500-row-group loop in 03_main/04_elasticity since we group once with
    np.unique + argsort instead of re-scanning per row-group."""
    cols = ['city_id', 'first_category_id', 'store_id', 'dt', 'hours_sale', 'hours_stock_status',
            'discount', 'holiday_flag', 'activity_flag', 'precpt', 'avg_temperature',
            'avg_humidity', 'avg_wind_level']
    t = pq.read_table(P, columns=cols)
    ci = t.column('city_id').to_numpy()
    fi = t.column('first_category_id').to_numpy()
    keys = np.stack([ci, fi], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    inv_sorted = inv[order]
    bounds = np.searchsorted(inv_sorted, np.arange(len(uniq) + 1))

    dt_days = pd.to_datetime(t.column('dt').to_pandas()).values.astype('datetime64[D]').astype(np.int64)
    dtord_all = dt_days + EPOCH_ORD
    store_all = t.column('store_id').to_numpy()
    hs_all = np.asarray(t.column('hours_sale').combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 24)
    ss_all = np.asarray(t.column('hours_stock_status').combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.int8).reshape(-1, 24)
    disc_all = t.column('discount').to_numpy()
    hol_all = t.column('holiday_flag').to_numpy()
    act_all = t.column('activity_flag').to_numpy()
    prec_all = t.column('precpt').to_numpy()
    temp_all = t.column('avg_temperature').to_numpy()
    hum_all = t.column('avg_humidity').to_numpy()
    wind_all = t.column('avg_wind_level').to_numpy()

    groups = {}
    for g, (city, cat) in enumerate(map(tuple, uniq)):
        if (int(city), int(cat)) not in eligible_keys:
            continue
        idx = order[bounds[g]:bounds[g + 1]]
        groups[(int(city), int(cat))] = dict(
            store=store_all[idx], dtord=dtord_all[idx], hs=hs_all[idx], ss=ss_all[idx],
            disc=disc_all[idx], hol=hol_all[idx], act=act_all[idx], prec=prec_all[idx],
            temp=temp_all[idx], hum=hum_all[idx], wind=wind_all[idx])
    return groups


def main():
    elast = pd.read_csv(ELAST_CSV)
    ok = elast[elast.status == 'ok'].copy()
    eligible_keys = set(zip(ok.city.astype(int), ok.cat.astype(int)))
    orig_by_key = {(int(row['city']), int(row['cat'])): row for row in ok.to_dict('records')}
    print(f'eligible cells: {len(eligible_keys)}', flush=True)

    done = load_done()
    if not os.path.exists(CKPT):
        pd.DataFrame(columns=COLS_D).to_csv(CKPT, index=False)
    print(f'resume: {len(done)} cells already done', flush=True)

    groups = build_groups(eligible_keys)
    print(f'panel built for {len(groups)} cells', flush=True)

    todo = sorted(k for k in groups if k not in done)
    n_done = len(done)
    n_total = len(todo) + n_done
    for key in todo:
        city, cat = key
        rec = run_cell(groups[key], orig_by_key[key])
        rec['city'], rec['cat'] = city, cat
        rec = {c: rec.get(c) for c in COLS_D}
        with open(CKPT, 'a') as fh:
            pd.DataFrame([rec]).to_csv(fh, index=False, header=False)
        n_done += 1
        print(f'[{n_done}/{n_total}] cell {city}x{cat}: pure_censoring depth={rec["bias_pct_pure_censoring_depth"]} '
              f'promo={rec["bias_pct_pure_censoring_promo"]}', flush=True)

    # ---- summary / gate metrics ----
    df = pd.read_csv(CKPT)
    med_depth = float(df.bias_pct_pure_censoring_depth.abs().median())
    med_promo = float(df.bias_pct_pure_censoring_promo.abs().median())
    frac_depth = float((df.bias_pct_pure_censoring_depth.abs() >= 0.2).mean())
    frac_promo = float((df.bias_pct_pure_censoring_promo.abs() >= 0.2).mean())
    orig_med_depth = float(ok.bias_pct_depth.abs().median())
    orig_med_promo = float(ok.bias_pct_promo.abs().median())
    orig_frac_depth = float((ok.bias_pct_depth.abs() >= 0.2).mean())
    orig_frac_promo = float((ok.bias_pct_promo.abs() >= 0.2).mean())

    gate_depth = (med_depth >= 0.2) and (frac_depth >= 0.5)
    gate_promo = (med_promo >= 0.2) and (frac_promo >= 0.5)

    summary = dict(n_cells=len(df),
                    orig_med_abs_bias_pct_depth=orig_med_depth, orig_med_abs_bias_pct_promo=orig_med_promo,
                    orig_frac_ge20pct_depth=orig_frac_depth, orig_frac_ge20pct_promo=orig_frac_promo,
                    pure_censoring_med_abs_bias_pct_depth=med_depth, pure_censoring_med_abs_bias_pct_promo=med_promo,
                    pure_censoring_frac_ge20pct_depth=frac_depth, pure_censoring_frac_ge20pct_promo=frac_promo,
                    gate_depth_pass=gate_depth, gate_promo_pass=gate_promo)
    with open(os.path.join(HERE, 'decomp_DONE'), 'w') as fh:
        fh.write(json.dumps(summary, indent=2))
    print('DONE', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
