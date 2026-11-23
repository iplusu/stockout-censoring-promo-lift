"""P6: fit_diagnostics (one row per attempted dataset/model, original and corrected statuses separate) +
before_after columns (absolute change, comparison basis). Full coefficients: out/p3_e2_fits.parquet, out/p3/coef_*.npz,
out/p3emp/*.json, out/p4/*.json."""
import glob, json, os
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, 'out')
KEY = ['city', 'cat', 'arm', 'param', 'rep']

# ---- E2: corrected M/NB2/MIX + legacy NB2/MIX from P3, legacy M (replay) and original shard statuses
f = pd.read_parquet(os.path.join(O, 'p3_e2_fits.parquet'))
rp = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(O, 'e2_replay_*.csv'))]); rp['param'] = rp.param.fillna(-1.0)
orig = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(H, '..', 'main', 'E2_misspec_cells_s*.csv'))) if '_v2' not in x])
orig['param'] = orig.param.fillna(-1.0)
c = f[f.impl == 'corrected'].copy()
c['original_shard_converged'] = pd.Series([None] * len(c), dtype=object, index=c.index)
for col in ['legacy_replay_converged', 'legacy_replay_b_p', 'legacy_replay_b_d']:
    c[col] = pd.Series([None] * len(c), dtype=object, index=c.index)
for m, pre in [('M', 'm'), ('NB2', 'nb2'), ('MIX', 'mix')]:
    o = orig.set_index(KEY)[f'{pre}_conv']; r = rp.set_index(KEY)[f'{pre}_conv'] if f'{pre}_conv' in rp else None
    idx = c.model == m; k = pd.MultiIndex.from_frame(c.loc[idx, KEY])
    c.loc[idx, 'original_shard_converged'] = o.reindex(k).values
    if m == 'M':
        c.loc[idx, 'legacy_replay_converged'] = r.reindex(k).values
        c.loc[idx, 'legacy_replay_b_p'] = rp.set_index(KEY)['m_p'].reindex(k).values
        c.loc[idx, 'legacy_replay_b_d'] = rp.set_index(KEY)['m_d'].reindex(k).values
lg = f[(f.impl == 'legacy')].set_index(KEY + ['model'])
k = pd.MultiIndex.from_frame(c[KEY + ['model']])
for col, src in [('legacy_replay_converged', 'status'), ('legacy_replay_b_p', 'b_p'), ('legacy_replay_b_d', 'b_d')]:
    v = lg[src].reindex(k).values
    if src == 'status':
        v = pd.Series(v).map({'legacy_conv': True, 'legacy_nonconv': False}).values
    m = c.model.isin(['NB2', 'MIX']).values
    c.loc[m, col] = v[m]
c = c.rename(columns={'status': 'corrected_status', 'valid': 'corrected_valid', 'b_p': 'corrected_b_p', 'b_d': 'corrected_b_d'})
c.insert(0, 'source', 'E2')
st = pd.concat([pd.read_csv(x, usecols=KEY + ['impl', 'model', 'start', 'msg', 'nfev']) for x in glob.glob(os.path.join(O, 'p3', 'starts_*_c*.csv'))])
st = st[st.impl == 'corrected'].rename(columns={'start': 'selected_start', 'msg': 'optimizer_message', 'nfev': 'optimizer_nfev'}).drop(columns='impl')
st['param'] = st['param'].astype(float); c['param'] = c['param'].astype(float)
c = c.merge(st, on=KEY + ['model', 'selected_start'], how='left', validate='m:1')
c['full_coefficients'] = 'out/p3/coef_<city>_<cat>_c<chunk>.npz (corrected), out/p3_e2_fits.parquet'

# ---- empirical 254 cells: M, NB2, MIX
rows = []
for p in sorted(glob.glob(os.path.join(O, 'p3emp', '*.json'))):
    o = json.load(open(p))
    for m in ['M', 'NB2', 'MIX']:
        cr = o[f'corrected_{m}']; lr = o['legacy_M'] if m == 'M' else o[f'legacy_{m}']
        rows.append(dict(source='empirical', city=o['city'], cat=o['cat'], model=m, n=o['n'], n_cens=o['n_cens'],
                         corrected_status=cr['status'], corrected_valid=cr['status'] == 'valid', g_orig=cr.get('g_orig'), g_std=cr.get('g_std'),
                         nit=cr.get('nit'), info=cr.get('info'), corrected_b_p=(cr.get('x') or [np.nan] * 3)[1], corrected_b_d=(cr.get('x') or [np.nan] * 3)[2],
                         legacy_replay_converged=lr.get('converged'), legacy_replay_b_p=(lr.get('b') or [np.nan] * 3)[1],
                         legacy_replay_b_d=(lr.get('b') or [np.nan] * 3)[2], legacy_grad_norm_sum_L2=lr.get('grad_norm_sum_L2'),
                         objective=cr.get('objective'), screen=cr.get('screen'), screen_J=cr.get('screen_J'), eta_min=cr.get('eta_min'),
                         eta_max=cr.get('eta_max'), eig_min=cr.get('eig_min'), eig_max=cr.get('eig_max'), cond=cr.get('cond'),
                         at_bound=cr.get('at_bound'), selected_start=cr.get('selected_start'),
                         full_coefficients=f'out/p3emp/{o["city"]}_{o["cat"]}.json'))
# ---- F model 20 cells
for p in sorted(glob.glob(os.path.join(O, 'p4', '[0-9]*_[0-9]*.json'))):
    r = json.load(open(p))
    if 'F_corrected' in r:
        cr, lr = r['F_corrected'], r['F_legacy']
        rows.append(dict(source='empirical_F', city=r['city'], cat=r['cat'], model='F_storeday', n=r['n'], corrected_status=cr['status'],
                         corrected_valid=cr['status'] == 'valid', g_orig=cr.get('final', {}).get('g_beta'), g_std=cr.get('final', {}).get('g_beta_std'),
                         nit=cr.get('nit'), corrected_b_p=cr.get('beta', {}).get('1'), corrected_b_d=cr.get('beta', {}).get('2'),
                         legacy_replay_converged=lr['converged'], legacy_replay_b_p=lr['b'][1], legacy_replay_b_d=lr['b'][2],
                         note=f"group-offset max mean score {cr.get('final', {}).get('g_alpha_group')}; separated groups all-censored {cr['sep_groups_all_censored']}, all-zero {cr['sep_groups_all_zero']}",
                         full_coefficients=f'out/p4/{r["city"]}_{r["cat"]}.json'))
D = pd.concat([c, pd.DataFrame(rows)], ignore_index=True)
x = os.path.join(O, 'p6b_e2_extra.parquet')   # p6b_diagnostics.py: censoring by promotion status, legacy-safeguard counts
if os.path.exists(x):
    e = pd.read_parquet(x).rename(columns={'n_lambda_clip_low': 'legacy_safeguard_lambda_clip_low_rows',
                                           'n_lambda_clip_high': 'legacy_safeguard_lambda_clip_high_rows',
                                           'n_tail_floor': 'legacy_safeguard_tail_floor_rows'})
    D['param'] = D['param'].astype(float)
    D = D.merge(e, on=KEY + ['model'], how='left', validate='m:1')
    D['legacy_safeguard_note'] = np.where(D.source == 'E2', 'rows where legacy clip/floor would activate, evaluated at the corrected coefficients (legacy full vectors were never saved)', '')
D.to_csv(os.path.join(O, 'fit_diagnostics.csv'), index=False)
print('fit_diagnostics rows', len(D), D.groupby(['source', 'model']).size().to_dict())
print(D.groupby(['source', 'model', 'corrected_status']).size().to_string())

# ---- before_after: absolute change + basis
B = pd.read_csv(os.path.join(O, 'before_after.csv'))
def absch(r):
    try:
        return abs(float(r.after) - float(r.before))
    except (TypeError, ValueError):
        return np.nan
B['abs_change'] = B.apply(absch, axis=1)
def basis(r):
    if r.section.startswith(('A4_F2', 'E2_F1')):
        return 'before: original 16 shards; legacy_on_valid: legacy estimates on IDs valid under both legacy and corrected; after: corrected, validity per solver contract (different eligible subsets)'
    if pd.notna(r.n_valid) and pd.notna(r.n_before) and r.n_valid != r.n_before:
        return f'before on {int(r.n_before)} cells; legacy_on_valid and after on the same {int(r.n_valid)} cells with a valid corrected fit'
    return 'same cells/IDs before and after' if pd.isna(r.n_after) or pd.isna(r.n_before) or r.n_after == r.n_before else f'different eligible subsets ({r.n_before} vs {r.n_after})'
B['comparison_basis'] = B.apply(basis, axis=1)
B.to_csv(os.path.join(O, 'before_after.csv'), index=False)
print('before_after rows', len(B))
