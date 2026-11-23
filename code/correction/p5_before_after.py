"""P5: before_after.csv -- every changed downstream quantity, before (live main/, as published) vs after (corrected
sandbox / P3 / P4 / P5-E2). Empirical statistics are recomputed from cell-level CSVs with explicit denominators:
  before_all     : legacy values, all cells as published
  legacy_on_valid: legacy values restricted to cells whose corrected fit is valid  (isolates exclusion effect)
  after          : corrected values on the same valid cells                       (estimate effect on same cells)
Writes out/before_after.csv (long form) and prints it.
"""
import glob, json, os
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); LIVE = os.path.join(os.path.dirname(H), 'main'); SB = os.path.join(H, 'sandbox', 'main')
R = []


def add(section, item, before, after, n_before=None, n_after=None, legacy_on_valid=None, n_valid=None, note=''):
    R.append(dict(section=section, item=item, before=before, legacy_on_valid=legacy_on_valid, after=after,
                  n_before=n_before, n_valid=n_valid, n_after=n_after, note=note))


def medabs(x): x = np.abs(np.asarray(x, float)); x = x[np.isfinite(x)]; return float(np.median(x)) if len(x) else np.nan
def frac(x, t=.2): x = np.abs(np.asarray(x, float)); x = x[np.isfinite(x)]; return float((x >= t).mean()) if len(x) else np.nan


# ---------------------------------------------------------------- empirical M status / coefficients
E = pd.read_csv(os.path.join(H, 'out', 'p3_empirical_summary.csv'))
st = E.cM.value_counts().to_dict()
add('empirical_fit_status', 'M valid / 254 (legacy: m_converged=True)', 254, int(st.get('valid', 0)), 254, 254,
    note=f'corrected statuses {st}; legacy flag was res.success only (254/254 had gradient > 1e-6)')
for m, col in [('NB2', 'cNB2'), ('MIX', 'cMIX')]:
    add('empirical_fit_status', f'{m} valid / 254', int(pd.read_csv(os.path.join(LIVE, 'pending_altmodels.csv'))[f'{m.lower().replace("mix", "zip")}_converged'].sum()),
        int((E[col] == 'valid').sum()), 254, 254, note='before = legacy converged flag (maxiter-100 era file)')
ok = E.cM == 'valid'
for c in ['promo', 'depth']:
    d = (E[f'cM_{c[0]}'] - E[f'leg_{c[0]}']).abs()[ok]
    add('empirical_coefficients', f'|b_M corrected - legacy| {c}: median', 0.0, float(d.median()), n_after=int(ok.sum()))
    add('empirical_coefficients', f'|b_M corrected - legacy| {c}: max', 0.0, float(d.max()), n_after=int(ok.sum()))

# ---------------------------------------------------------------- Table commondenom + reclassification (decomp_cells)
Db = pd.read_csv(os.path.join(LIVE, 'decomp_cells.csv')); Da = pd.read_csv(os.path.join(SB, 'decomp_cells.csv'))
key = ['city', 'cat']; Da = Da.set_index(key).loc[Db.set_index(key).index].reset_index()
valid = Da.beta_rec_depth.notna() & Da.beta_rec_promo.notna()
for c in ['depth', 'promo']:
    def comps(D):
        bM = D[f'beta_rec_{c}'].abs()
        return {'unmatched N-M': (D[f'beta_naive_reduced_{c}'] - D[f'beta_rec_{c}']) / bM,
                'omission N-NF': (D[f'beta_naive_reduced_{c}'] - D[f'beta_naive_full_{c}']) / bM,
                'dist NF-G': (D[f'beta_naive_full_{c}'] - D[f'beta_glm_{c}']) / bM,
                'matched G-M (censoring)': (D[f'beta_glm_{c}'] - D[f'beta_rec_{c}']) / bM}
    cb, ca = comps(Db), comps(Da)
    for k in cb:
        add('table_commondenom', f'{c} {k}: median |gap|', medabs(cb[k]), medabs(ca[k][valid]), len(Db), int(valid.sum()), medabs(cb[k][valid]), int(valid.sum()))
        add('table_commondenom', f'{c} {k}: share >= 20%', frac(cb[k]), frac(ca[k][valid]), len(Db), int(valid.sum()), frac(cb[k][valid]), int(valid.sum()))
    def recl(cc, m):
        u = cc['unmatched N-M'][m].abs() >= .2; g = cc['matched G-M (censoring)'][m].abs() >= .2
        return int((u != g).sum()), int((u & ~g).sum()), int((g & ~u).sum())
    allm = pd.Series(True, index=Db.index)
    b, lv, a = recl(cb, allm), recl(cb, valid), recl(ca, valid)
    for i, nm in enumerate(['cells changing flag', 'flagged only unmatched', 'flagged only matched']):
        add('reclassification_20pct', f'{c}: {nm}', b[i], a[i], len(Db), int(valid.sum()), lv[i], int(valid.sum()),
            note='published: depth 90 (66/24), promo 64 (39/25) of 254' if i == 0 else '')
    sm = lambda D: int(((D[f'beta_rec_{c}'].abs()) < .05).sum())
    add('table_commondenom', f'{c}: cells with |b_M| < 0.05', sm(Db), sm(Da[valid]), len(Db), int(valid.sum()))
    sf = lambda D, m: float((np.sign(D[f'beta_glm_{c}'][m]) != np.sign(D[f'beta_rec_{c}'][m])).mean())
    add('sign_flips', f'{c}: sign(G) != sign(M) share', sf(Db, allm), sf(Da, valid), len(Db), int(valid.sum()), sf(Db, valid), int(valid.sum()))

# ---------------------------------------------------------------- P1a mean-effect gap (main_cells)
Mb = pd.read_csv(os.path.join(LIVE, 'main_cells.csv')); Ma = pd.read_csv(os.path.join(SB, 'main_cells.csv'))
Mb, Ma = Mb[Mb.status == 'ok'], Ma[Ma.status == 'ok'].set_index(key).loc[Mb[Mb.status == 'ok'].set_index(key).index].reset_index()
va = Ma.bias_ratio.notna().values
for nm, f in [('median bias_ratio', lambda s: float(np.nanmedian(s))), ('share bias_ratio >= 20%', lambda s: frac(s))]:
    add('P1a_mean_effect', nm, f(Mb.bias_ratio), f(Ma.bias_ratio[va]), len(Mb), int(va.sum()), f(Mb.bias_ratio.values[va]), int(va.sum()))
add('P1a_mean_effect', 'median |recA corrected - legacy|', 0.0, float(np.nanmedian(np.abs(Ma.recA.values[va] - Mb.recA.values[va]))), n_after=int(va.sum()))

# ---------------------------------------------------------------- combined lift
jb = json.load(open(os.path.join(LIVE, 'discount_scenario_jointboot.json'))); ja = json.load(open(os.path.join(SB, 'discount_scenario_jointboot.json')))
for k in jb:
    add('combined_lift_G_minus_M_pp', f'{k} point [95% city-block CI]', f"{jb[k]['point']:.2f} [{jb[k]['lo']:.2f}, {jb[k]['hi']:.2f}]",
        f"{ja[k]['point']:.2f} [{ja[k]['lo']:.2f}, {ja[k]['hi']:.2f}]", 254, 249, note='seed 42, 5,000 draws; after excludes 5 cells without valid corrected M')

# ---------------------------------------------------------------- E2 / A4
S = {s: (pd.read_csv(os.path.join(H, 'out', 'p5_e2', s, 'e2_f2_tests2.csv')), pd.read_csv(os.path.join(H, 'out', 'p5_e2', s, 'e2_f1_tests2.csv')))
     for s in ['orig', 'replay', 'legacy_on_common', 'corrected_on_common', 'corrected']}
fmt = lambda r: f"{r.point:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] cells={r.n_cells} p_holm={r.p_holm:.4f}{' *' if r.sig_holm else ''}"
for fam, j in [('A4_F2_12_tests', 0), ('E2_F1_10_tests', 1)]:
    for i in range(len(S['orig'][j])):
        r = {s: S[s][j].iloc[i] for s in S}
        add(fam, f"{r['orig'].arm} rho/param={r['orig'].param} coef={r['orig'].coef} {r['orig']['compare']}", fmt(r['orig']), fmt(r['corrected']),
            legacy_on_valid=fmt(r['legacy_on_common']),
            note=f"replay(legacy, reconstructed inputs)={fmt(r['replay'])}; corrected_on_common={fmt(r['corrected_on_common'])}")
F = pd.read_parquet(os.path.join(H, 'out', 'p3_e2_fits.parquet'))
rp = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(H, 'out', 'e2_replay_*.csv'))]); rp['param'] = rp.param.fillna(-1.0)
orig = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(LIVE, 'E2_misspec_cells_s*.csv'))) if '_v2' not in f]); orig['param'] = orig.param.fillna(-1.0)
for rho in [0.0, 0.5, 1.0]:
    a4 = lambda d: d[(d.arm == 'A4_nonignorable') & np.isclose(d.param, rho)]
    cm = a4(F[(F.impl == 'corrected') & (F.model == 'M')])
    add('A4_failures', f'rho={rho}: M fits without validity (all attempts 2700)', int((~a4(orig).m_conv).sum()), int((~cm.valid).sum()), 2700, 2700,
        legacy_on_valid=int((~a4(rp).m_conv).sum()),
        note=f"after by cause {cm.status.value_counts().to_dict()}; legacy_on_valid column = legacy flag on reconstructed inputs")
    add('A4_failures', f'rho={rho}: median censored share (same generated data)', None, float(cm.cens_share.median()), n_after=len(cm),
        note='censoring share was not stored in the original shards; value from the frozen reconstructed inputs')
    for c, tc in [('p', 'beta_true_p'), ('d', 'beta_true_d')]:
        def mederr(d, col, convcol, truth):
            d = d[d[convcol].astype(bool) & (d[truth].abs() >= .05)]
            e = ((d[col] - d[truth]).abs() / d[truth].abs())
            return float(e.groupby([d.city, d.cat]).median().median()), len(d)
        b, nb = mederr(a4(orig), f'm_{c}', 'm_conv', tc)
        cc = cm.merge(a4(rp)[['city', 'cat', 'rep', tc]], on=['city', 'cat', 'rep']); cc[f'm_{c}'] = cc[f'b_{c}']
        a, na = mederr(cc, f'm_{c}', 'valid', tc)
        add('A4_coefficient_error', f'rho={rho} coef={c}: M median relative error (cell medians -> median)', b, a, nb, na)

# ---------------------------------------------------------------- hourly / F (P4)
Hh = pd.read_csv(os.path.join(H, 'out', 'p4_hourly_summary.csv'))
add('hourly', 'cells whose recovered-demand peak hour changes (mask fix)', 0, int((Hh.peak_rec_H0 != Hh.peak_rec_HA).sum()), 254, 254)
add('hourly', 'cells whose bias peak hour changes (mask fix)', 0, int((Hh.peak_bias_H0 != Hh.peak_bias_HA).sum()), 254, 254)
add('hourly', 'cells whose recovered peak changes (mask + corrected M vs original)', 0, int((Hh.peak_rec_H0 != Hh.peak_rec_HB).sum()), 254, int(Hh.peak_rec_HB.notna().sum()))
for h in [Hh.peak_rec_H0.mode()[0]]:
    add('hourly', 'modal recovered-demand peak hour', int(Hh.peak_rec_H0.mode()[0]), int(Hh.peak_rec_HA.mode()[0]), 254, 254)
add('hourly', 'daily totals max |change| from mask fix (should be 0)', 0.0, float(Hh.daily_total_maxdiff_H0_HA.max()), 254, 254)
Fz = pd.read_csv(os.path.join(H, 'out', 'p4_F_summary.csv'))
add('F_model', 'F fits jointly valid / 20', int((Fz.stored_f_nit > 1).sum()), int((Fz.status == 'valid').sum()), 20, 20,
    note='before = cells where the legacy alternating loop ran >1 iteration (legacy has no joint check; 6 cells stopped at f_nit=1)')
for _, r in Fz.iterrows():
    add('F_model', f'cell {r.city},{r["cat"]}: recF (store-day within, recovered)', r.stored_recF, r.recF,
        note=f"b_promo {r.legacy_b_promo:.4f}->{r.b_promo:.4f}, b_depth {r.legacy_b_depth:.4f}->{r.b_depth:.4f}; rows w/o finite recovery {r.rows_without_recovery}")

# ---------------------------------------------------------------- stage-2 summary tables (live main/ vs corrected sandbox)
def table_diff(fname, keys, section, drop=()):
    b = pd.read_csv(os.path.join(LIVE, fname)); a = pd.read_csv(os.path.join(SB, fname))
    j = b.merge(a, on=keys, how='outer', suffixes=('_b', '_a'), indicator=True)
    for _, r in j.iterrows():
        for c in [c for c in b.columns if c not in keys and c not in drop]:
            vb, va = r.get(f'{c}_b'), r.get(f'{c}_a')
            try:
                fb, fa = float(vb), float(va)
                same = (np.isnan(fb) and np.isnan(fa)) or abs(fb - fa) <= 1e-12
            except (TypeError, ValueError):
                same = str(vb) == str(va)
            if not same:
                add(section, ' '.join(str(r[k]) for k in keys) + f' {c}', vb, va, note=f'merge={r["_merge"]}')

table_diff('fix_posthoc_pA.csv', ['alt', 'coef'], 'sec3.9_altmodel_gap_fix_posthoc_pA')
table_diff('fix_posthoc_pC.csv', ['stat'], 'wild_cluster_CI_fix_posthoc_pC')
table_diff('pending_joint_ci.csv', ['stat'], 'city_block_CI_pending_joint_ci', drop=('method',))
table_diff('pending_final_dose_wildci.csv', ['quintile'], 'dose_response_wildci')
for fname, sec, cols in [('pending_fe_vif.csv', 'product_FE', ['bias_pct_fe_depth', 'bias_pct_fe_promo']),
                         ('pending_decomp_order.csv', 'decomp_order', ['cens_B_depth', 'cens_B_promo', 'omit_B_depth', 'omit_B_promo'])]:
    b = pd.read_csv(os.path.join(LIVE, fname)); a = pd.read_csv(os.path.join(SB, fname))
    a = a.set_index(key).loc[b.set_index(key).index].reset_index()
    for c in cols:
        if c in b:
            v = a[c].notna()
            add(sec, f'{c}: median |.|', medabs(b[c]), medabs(a[c][v]), len(b), int(v.sum()), medabs(b[c][v]), int(v.sum()))
            add(sec, f'{c}: share >= 20%', frac(b[c]), frac(a[c][v]), len(b), int(v.sum()), frac(b[c][v]), int(v.sum()))
    if 'fe_converged' in b:
        add(sec, 'product-FE fits converged', int(b.fe_converged.astype(str).eq('True').sum()), int(a.fe_converged.astype(str).eq('True').sum()), len(b), None, None, len(a))
rb = pd.read_csv(os.path.join(LIVE, 'risk_coverage_cells.csv')); ra = pd.read_csv(os.path.join(SB, 'risk_coverage_cells.csv'))
add('risk_coverage_sim', 'cells ok', int((rb.status == 'ok').sum()), int((ra.status == 'ok').sum()), len(rb), None, None, len(ra))

out = pd.DataFrame(R); out.to_csv(os.path.join(H, 'out', 'before_after.csv'), index=False)
pd.set_option('display.width', 300, 'display.max_colwidth', 120, 'display.max_rows', 500)
for _, r in out.iterrows():
    print(f"{r.section} | {r['item']} | before={r.before} | legacy_on_valid={r.legacy_on_valid} | after={r.after} | n={r.n_before}/{r.n_valid}/{r.n_after}")
