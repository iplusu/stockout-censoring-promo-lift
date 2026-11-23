"""Aggregation, paired contrasts, bootstrap, tables (protocol §6-8, §10).

  python analysis.py --stream evaluation      # reads out/evaluation/rep_*.json -> results/*.csv
Conventions (recorded in run_manifest): median = ordinary midpoint median (numpy); quantiles = numpy
'linear'; MCSE = bootstrap SD with denominator B_usable - 1. Bootstrap = bootstrap child stream
SeedSequence(20261122).spawn(3)[1], one Generator(PCG64), iteration b = row b of integers(0, R, (2000, R)).
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

CONDS = [(0.0, 0.2), (0.6, 0.2), (0.0, 0.6), (0.6, 0.6), (0.0, 0.95), (0.6, 0.95)]   # Table 6 fixed order
CNAME = {c: f'eta{c[0]}_q{c[1]}' for c in CONDS}
QS = [0.2, 0.6, 0.95]
NBOOT = 2000


# ------------------------------------------------------------------ core statistics (pure, reused by bootstrap)
def paired_median(B, idx=None):
    """median over eligible IDs of within-ID differences B (NaN = ineligible). Returns (value or nan, n_eligible)."""
    v = B if idx is None else B[idx]
    v = v[~np.isnan(v)]
    return (float(np.median(v)) if v.size else np.nan), int(v.size)


def theta_vector(B0, B6):
    """within-ID dependence difference B(.6,q) - B(0,q); NaN unless ID is in W_q = V_0q ∩ V_.6q."""
    return B6 - B0


def boot_medians(B, idxmat):
    """Vectorized paired_median over bootstrap ID vectors (rows of idxmat, multiplicities kept)."""
    M = B[idxmat]
    n = (~np.isnan(M)).sum(1)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med = np.nanmedian(M, axis=1)
    med[n == 0] = np.nan
    return med, n


def summarize(point, n_orig, boots):
    u = boots[~np.isnan(boots)]
    out = dict(point=point, n_valid_orig=n_orig, B_usable=int(u.size), B_total=len(boots))
    if n_orig == 0:
        out.update(estimable=False, lo=np.nan, hi=np.nan, mcse=np.nan, note='not estimable: no original valid pairs')
    elif u.size < 2:
        out.update(estimable=True, lo=np.nan, hi=np.nan, mcse=np.nan, note='interval/SE unavailable: <2 usable resamples')
    else:
        lo, hi = np.percentile(u, [2.5, 97.5])
        out.update(estimable=True, lo=float(lo), hi=float(hi), mcse=float(np.std(u, ddof=1)),
                   note='' if u.size == len(boots) else f'interval based on {u.size} estimable resamples of {len(boots)}')
    return out


# ------------------------------------------------------------------ load
def load(stream):
    files = sorted(glob.glob(f'out/{stream}/rep_*.json'))
    tabs = {k: [] for k in ['datasets', 'fits', 'starts', 'losses', 'information']}
    for f in files:
        o = json.load(open(f))
        for k in tabs:
            tabs[k] += o[k]
    return {k: pd.DataFrame(v) for k, v in tabs.items()}, len(files)


def condition_arrays(fits, losses, rids):
    """For each condition: per-ID arrays (index = position in rids) of A, E_P, E_d, ae_d for M and G (NaN if invalid)."""
    pos = {r: i for i, r in enumerate(rids)}
    R = len(rids)
    arr = {}
    lo = losses.set_index(['rid', 'condition', 'model'])
    for c in CONDS:
        cn = CNAME[c]
        for m in ['M', 'G']:
            sub = lo.xs((cn, m), level=('condition', 'model'))
            for col in ['A_pp', 'E_P', 'E_d', 'ae_05_pp', 'ae_15_pp', 'ae_30_pp']:
                a = np.full(R, np.nan)
                for r, v in sub[col].items():
                    a[pos[r]] = v
                arr[(cn, m, col)] = a
    return arr


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream', default='evaluation')
    ap.add_argument('--outdir', default='results')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    T, nrep = load(a.stream)
    for k, df in T.items():
        df.to_csv(f'{a.outdir}/{k}.csv', index=False, float_format='%.17g')
    rids = sorted(T['datasets']['rid'].unique())
    R = len(rids)
    arr = condition_arrays(T['fits'], T['losses'], rids)

    # paired.csv
    prow, Bvec = [], {}
    for c in CONDS:
        cn = CNAME[c]
        d = {col: arr[(cn, 'M', col)] - arr[(cn, 'G', col)] for col in ['A_pp', 'E_P', 'E_d', 'ae_05_pp', 'ae_15_pp', 'ae_30_pp']}
        Bvec[cn] = d
        for i, r in enumerate(rids):
            inV = not np.isnan(d['A_pp'][i])
            cat = ''
            if inV:
                both = d['E_P'][i] < 0 and d['E_d'][i] < 0
                lift = d['A_pp'][i] < 0
                cat = {(True, True): 'both_coef_and_lift', (True, False): 'both_coef_not_lift',
                       (False, True): 'lift_not_both_coef', (False, False): 'neither'}[(both, lift)]
            prow.append(dict(rid=r, condition=cn, eta=c[0], q=c[1], in_V=inV, B_lift_pp=d['A_pp'][i],
                             dE_P=d['E_P'][i], dE_d=d['E_d'][i], d_ae_05_pp=d['ae_05_pp'][i],
                             d_ae_15_pp=d['ae_15_pp'][i], d_ae_30_pp=d['ae_30_pp'][i], agreement=cat))
    paired = pd.DataFrame(prow)
    paired.to_csv(f'{a.outdir}/paired.csv', index=False, float_format='%.17g')

    # dependence.csv
    drow, Tvec = [], {}
    for q in QS:
        tv = theta_vector(Bvec[f'eta0.0_q{q}']['A_pp'], Bvec[f'eta0.6_q{q}']['A_pp'])
        Tvec[q] = tv
        for i, r in enumerate(rids):
            drow.append(dict(rid=r, q=q, in_W=not np.isnan(tv[i]), within_id_diff_pp=tv[i]))
    pd.DataFrame(drow).to_csv(f'{a.outdir}/dependence.csv', index=False, float_format='%.17g')

    # bootstrap
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261122).spawn(3)[1]))
    idxmat = rng.integers(0, R, size=(NBOOT, R))
    stats = {}
    for c in CONDS:
        cn = CNAME[c]
        for col, nm in [('A_pp', 'Delta_lift_pp'), ('E_d', 'Delta_E_d'), ('E_P', 'Delta_E_P'),
                        ('ae_05_pp', 'Delta_ae_05_pp'), ('ae_15_pp', 'Delta_ae_15_pp'), ('ae_30_pp', 'Delta_ae_30_pp')]:
            stats[(nm, cn)] = Bvec[cn][col]
    for q in QS:
        stats[('Theta', f'q{q}')] = Tvec[q]
    brow, srow = [], []
    for (nm, cond), vec in stats.items():
        pt, n0 = paired_median(vec)
        bm, bn = boot_medians(vec, idxmat)
        for b in range(NBOOT):
            brow.append(dict(iteration=b, statistic=nm, condition=cond, value=bm[b], missing=bool(np.isnan(bm[b])),
                             eligible_pairs=int(bn[b])))
        s = summarize(pt, n0, bm)
        srow.append(dict(statistic=nm, condition=cond, primary=(nm == 'Theta' and cond == 'q0.6'), **s))
    pd.DataFrame(brow).to_csv(f'{a.outdir}/bootstrap.csv', index=False, float_format='%.17g')
    # agreement categories (descriptive)
    for c in CONDS:
        cn = CNAME[c]
        v = paired[(paired.condition == cn) & paired.in_V]
        for k in ['both_coef_and_lift', 'both_coef_not_lift', 'lift_not_both_coef', 'neither']:
            n = int((v.agreement == k).sum())
            srow.append(dict(statistic=f'agreement_{k}', condition=cn, point=(n / len(v) if len(v) else np.nan),
                             n_valid_orig=len(v), note=f'count={n} / |V|={len(v)} (descriptive)'))
    summ = pd.DataFrame(srow)
    summ.to_csv(f'{a.outdir}/summaries.csv', index=False, float_format='%.17g')

    # failure_summary.csv + Table A8 inputs (denominator = attempted fits, never starts)
    fits, ds, info = T['fits'], T['datasets'], T['information']
    frow = []
    for c in CONDS + [None]:
        cn = CNAME[c] if c else ''
        for m in (['G', 'M'] if c else ['O']):
            f = fits[(fits.condition == cn) & (fits.model == m)]
            n = len(f)
            row = dict(condition=cn or 'oracle', model=m, n_attempted=n)
            for st in ['valid', 'structural_nonexistence', 'unresolved_screen', 'nonconvergence',
                       'nonfinite_objective_coefficient', 'nonfinite_lift_loss']:
                row[f'n_{st}'] = int((f.status == st).sum())
            for fl in ['f_struct_nonexist', 'f_struct_unresolved', 'f_nonconv', 'f_nonfinite_obj_coef', 'f_nonfinite_loss']:
                row[f'flag_{fl}'] = int(f[fl].sum())
            p = 1 - row['n_valid'] / n if n else np.nan
            row.update(invalid_frac=p, invalid_frac_mcse=np.sqrt(p * (1 - p) / n) if n else np.nan)
            if c:
                dd, ii = ds[ds.condition == cn], info[info.condition == cn]
                row.update(mean_cens_share=dd.cens_share.mean(), median_min_uncens=dd.min_uncens_stratum.median(),
                           n_all_censored=int((dd.cens_share == 1).sum()), n_rank_deficient=int((dd.uncens_rank < 3).sum()))
                for s in ['no_promo', 'shallow_promo', 'moderate_promo', 'deep_promo']:
                    row[f'mean_cens_{s}'] = dd[f'cens_share_{s}'].mean()
                if m == 'M':
                    elig = ii.merge(f[['rid', 'valid']], on='rid')
                    elig = elig[elig.finite_candidate & elig.valid]
                    row.update(median_cond_eligibleM=elig.cond.median() if len(elig) else np.nan, n_cond_denominator=len(elig),
                               **{f'diag_{k}': int((ii.diag_status == k).sum()) for k in ['ok', 'singular_or_nearly', 'indefinite', 'missing', 'no_candidate']})
            frow.append(row)
    pd.DataFrame(frow).to_csv(f'{a.outdir}/failure_summary.csv', index=False, float_format='%.17g')

    # oracle table
    lo = T['losses']
    o = lo[lo.model == 'O']
    orow = dict(n_attempted=len(o), n_valid=int(o.A_pp.notna().sum()))
    for col in ['E_P', 'E_d', 'A_pp']:
        v = o[col].dropna()
        orow.update({f'{col}_median': v.median(), f'{col}_p10': v.quantile(.1), f'{col}_p90': v.quantile(.9)})
    pd.DataFrame([orow]).to_csv(f'{a.outdir}/oracle_summary.csv', index=False, float_format='%.17g')
    print(f'{a.stream}: {nrep} replicate files, {len(T["datasets"])} datasets, {len(fits)} fits, '
          f'{len(T["starts"])} starts -> {a.outdir}/')


if __name__ == '__main__':
    main()
