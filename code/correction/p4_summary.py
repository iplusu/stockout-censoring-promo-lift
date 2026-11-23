"""P4 summary: hourly H0/HA/HB per cell + F legacy vs corrected (20 cells). Writes out/p4_hourly_summary.csv, out/p4_F_summary.csv."""
import glob, json, os
import numpy as np, pandas as pd

rows, F = [], []
hs = pd.read_parquet('../main/main_hourly.parquet').set_index(['city', 'cat'])
topF = pd.read_csv('../main/main_topF.csv').set_index(['city', 'cat'])
for p in sorted(glob.glob('out/p4/[0-9]*_[0-9]*.json')):
    r = json.load(open(p)); k = (r['city'], r['cat'])
    a = np.load(p.replace('.json', '_hourly.npz'))
    row = {c: r.get(c) for c in ['city', 'cat', 'n', 'n_uncensored', 'corrected_M_status', 'naiveA', 'recA_H0', 'recA_HA', 'recA_HB',
                                'HA_unavailable', 'HA_n_zero_profile_fallback', 'HA_n_dhat_below_K', 'HB_n_dhat_below_K',
                                'HA_chk_nonneg', 'HA_chk_only_flagged_hours', 'HA_chk_uncensored_unchanged', 'HA_chk_max_daily_total_err',
                                'HB_chk_max_daily_total_err', 'daily_total_maxdiff_H0_HA', 'daily_recovered_mean_legacy', 'daily_recovered_mean_corrected']}
    row['H0_vs_stored_rec_h_maxdiff'] = float(np.max(np.abs(hs.loc[k, [f'rec_h{i}' for i in range(24)]].values - a['rec_h_H0']))) if k in hs.index else np.nan
    for t in ['H0', 'HA', 'HB']:
        if f'rec_h_{t}' in a:
            v = a[f'rec_h_{t}']; row[f'peak_rec_{t}'] = int(np.argmax(v)); row[f'peak_bias_{t}'] = int(np.argmax(a['naive_h'] - v))
            row[f'trough_bias_{t}'] = int(np.argmin(a['naive_h'] - v))
    row['max_abs_rec_h_change_HA_vs_H0'] = float(np.max(np.abs(a['rec_h_HA'] - a['rec_h_H0']))) if 'rec_h_HA' in a else np.nan
    row['max_abs_rec_h_change_HB_vs_HA'] = float(np.max(np.abs(a['rec_h_HB'] - a['rec_h_HA']))) if 'rec_h_HB' in a else np.nan
    rows.append(row)
    if 'F_corrected' in r:
        c, l = r['F_corrected'], r['F_legacy']; s = topF.loc[k]
        F.append(dict(city=k[0], cat=k[1], n=r['n'], stored_naiveB=s.naiveB, stored_recF=s.recF, stored_f_nit=s.f_nit,
                      legacy_replay_recF=l['recF'], legacy_replay_nit=l['nit'], legacy_converged_flag=l['converged'],
                      legacy_b_promo=l['b'][1], legacy_b_depth=l['b'][2], replay_naiveB=r['F_naiveB'],
                      status=c['status'], identified_cols=str(c['identified_cols']), n_groups=c['n_groups'],
                      sep_all_censored=c['sep_groups_all_censored'], sep_all_zero=c['sep_groups_all_zero'], sep_rows=c['sep_rows'],
                      screen=c.get('screen'), nit=c.get('nit'), g_beta=c.get('final', {}).get('g_beta'),
                      g_alpha_group=c.get('final', {}).get('g_alpha_group'),
                      b_promo=c.get('beta', {}).get('1'), b_depth=c.get('beta', {}).get('2'), b_activity=c.get('beta', {}).get('9'),
                      recF=c.get('recF'), naiveB_same_rows=c.get('naiveB_same_rows'), rows_without_recovery=c.get('rows_without_finite_recovery')))
R = pd.DataFrame(rows); R.to_csv('out/p4_hourly_summary.csv', index=False)
FF = pd.DataFrame(F); FF.to_csv('out/p4_F_summary.csv', index=False)
print('hourly cells', len(R), '| unavailable(no uncensored row)', int(R.HA_unavailable.sum()), '| HB available', int(R.recA_HB.notna().sum()))
print('checks HA all pass:', bool(R.HA_chk_nonneg.all() and R.HA_chk_only_flagged_hours.all() and R.HA_chk_uncensored_unchanged.all()),
      '| max daily-total err HA', R.HA_chk_max_daily_total_err.max(), 'HB', R.HB_chk_max_daily_total_err.max(),
      '| daily total H0 vs HA max', R.daily_total_maxdiff_H0_HA.max(), '| zero-profile fallback rows', int(R.HA_n_zero_profile_fallback.sum()),
      '| Dhat<K rows legacy', int(R.HA_n_dhat_below_K.sum()), 'corrected', int(R.HB_n_dhat_below_K.fillna(0).sum()))
print('H0 replay vs stored main_hourly: max', R.H0_vs_stored_rec_h_maxdiff.max(), 'cells >1e-4', int((R.H0_vs_stored_rec_h_maxdiff > 1e-4).sum()), 'of', int(R.H0_vs_stored_rec_h_maxdiff.notna().sum()))
print('recA (daily mean) H0 vs HA max', (R.recA_H0 - R.recA_HA).abs().max(), '| HA vs HB max', (R.recA_HA - R.recA_HB).abs().max())
print('peak rec hour changed H0->HA:', int((R.peak_rec_H0 != R.peak_rec_HA).sum()), '/', len(R), '| HA->HB:', int((R.peak_rec_HA != R.peak_rec_HB).sum()),
      '| peak bias hour changed H0->HA:', int((R.peak_bias_H0 != R.peak_bias_HA).sum()))
print('peak rec hour distribution H0', R.peak_rec_H0.value_counts().head(5).to_dict(), 'HA', R.peak_rec_HA.value_counts().head(5).to_dict())
print('peak bias hour distribution H0', R.peak_bias_H0.value_counts().head(5).to_dict(), 'HA', R.peak_bias_HA.value_counts().head(5).to_dict())
print('max |rec_h change| HA vs H0: median', R.max_abs_rec_h_change_HA_vs_H0.median(), 'max', R.max_abs_rec_h_change_HA_vs_H0.max())
pd.set_option('display.width', 250)
print(FF[['city', 'cat', 'stored_recF', 'legacy_replay_recF', 'stored_f_nit', 'legacy_replay_nit', 'status', 'sep_all_censored', 'n_groups', 'rows_without_recovery',
          'legacy_b_promo', 'b_promo', 'legacy_b_depth', 'b_depth', 'replay_naiveB', 'recF', 'naiveB_same_rows']].round(4).to_string(index=False))
