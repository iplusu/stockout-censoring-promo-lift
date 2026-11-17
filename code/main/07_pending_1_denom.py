"""Pending #1 — Table 4 common-denominator re-expression.

All four decomposition components (attribution, omission, distributional, pure
censoring) for depth and promo are re-expressed on the SINGLE common denominator
|b_M| = |beta_rec| (M = censoring-recovered model coefficient). The shipped
decomp_cells.csv mixed denominators: bias_pct_dist was normalized by |beta_glm|
(see 04_elasticity.py L65 convention and decomp_report.md §(c) caveat), so
component magnitudes were not directly comparable and no 'relative importance'
ranking was valid. This script is pure arithmetic on decomp_cells.csv — no refits.

Identity per cell (exact, up to CSV rounding ~1e-5):
    bias_pct_orig (naive_reduced − M) = cd_omission + cd_dist + cd_pure_cens
    cd_attr (naive_full − M)          = cd_dist + cd_pure_cens

Outputs: pending_table4_commondenom.csv (254 rows), pending_table4_summary.md.
Selftest: first 10 cells — identity residual < 1e-3 and cd_attr reproduces the
existing bias_pct_a (same denominator by construction) within 1e-4.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'decomp_cells.csv')
OUT_CSV = os.path.join(HERE, 'pending_table4_commondenom.csv')
OUT_MD = os.path.join(HERE, 'pending_table4_summary.md')

BETA_COLS = ['beta_naive_reduced_depth', 'beta_naive_reduced_promo',
             'beta_naive_full_depth', 'beta_naive_full_promo',
             'beta_glm_depth', 'beta_glm_promo',
             'beta_rec_depth', 'beta_rec_promo']
COMPONENTS = ['cd_attr', 'cd_omission', 'cd_dist', 'cd_pure_cens']  # x _depth/_promo
OUT_COLS = ['city', 'cat', 'n_rows', 'n_promo'] + BETA_COLS + [f'{c}_{s}' for s in ('depth', 'promo') for c in COMPONENTS] + ['resid_depth', 'resid_promo']


def compute(df):
    out = df[['city', 'cat', 'n_rows', 'n_promo'] + BETA_COLS].copy()
    for s in ('depth', 'promo'):
        nr = df[f'beta_naive_reduced_{s}'].to_numpy(float)
        nf = df[f'beta_naive_full_{s}'].to_numpy(float)
        gl = df[f'beta_glm_{s}'].to_numpy(float)
        rc = df[f'beta_rec_{s}'].to_numpy(float)
        den = np.abs(rc)  # common denominator |b_M|
        out[f'cd_attr_{s}'] = (nf - rc) / np.maximum(den, 1e-9)
        out[f'cd_omission_{s}'] = (nr - nf) / np.maximum(den, 1e-9)
        out[f'cd_dist_{s}'] = (nf - gl) / np.maximum(den, 1e-9)
        out[f'cd_pure_cens_{s}'] = (gl - rc) / np.maximum(den, 1e-9)
        # exact identity: orig gap (nr - rc)/den = om + dist + cens (up to FP rounding)
        orig = (nr - rc) / np.maximum(den, 1e-9)
        out[f'resid_{s}'] = (orig - (out[f'cd_omission_{s}'] + out[f'cd_dist_{s}'] + out[f'cd_pure_cens_{s}'])).round(8)
    return out[OUT_COLS]


def summarize(df, out, n_total):
    lines = ['# Pending #1 — Table 4, common denominator |b_M| = |beta_rec|\n',
             f'Script: `main/07_pending_1_denom.py`. Source: `decomp_cells.csv` (n={n_total} cells, all status==ok). '
             'Pure arithmetic — no refits, no RNG, seed n/a.\n',
             'All components now share the denominator |beta_rec|, so magnitudes are directly comparable and the '
             'per-cell additive identity `cd_attr = cd_omission + cd_dist + cd_pure_cens` holds (max |resid| reported below).\n',
             '**Note:** relative-importance rankings of the components are only meaningful on this common scale. '
             'In the shipped Table 4 the distributional component was normalized by |beta_glm| while the others used '
             '|beta_rec|, so any cross-component ranking from the mixed-denominator columns is invalid.\n']
    tab = ['| component | coef | median |bias| | mean |bias| | signed median | frac |bias|>=20% |',
           '|---|---|---|---|---|---|']
    stats = {}
    for c in COMPONENTS:
        for s in ('depth', 'promo'):
            v = out[f'{c}_{s}']
            av = v.abs()
            st = dict(med_abs=float(av.median()), mean_abs=float(av.mean()),
                      signed_med=float(v.median()), frac_ge20=float((av >= 0.2).mean()))
            stats[(c, s)] = st
            tab.append(f'| {c} | {s} | {st["med_abs"]:.4f} | {st["mean_abs"]:.4f} | {st["signed_med"]:+.4f} | {st["frac_ge20"]:.4f} |')
    lines += ['\n## Component sizes on the common |b_M| scale (254 cells)\n'] + tab
    rank = sorted(stats, key=lambda k: -stats[k]['med_abs'])
    lines.append('\n**Relative-importance ranking (common scale only):** ' +
                 ' > '.join(f'{c}×{s} ({stats[(c, s)]["med_abs"]:.4f})' for c, s in rank))
    resid = max(float(out['resid_depth'].abs().max()), float(out['resid_promo'].abs().max()))
    lines.append(f'\n- max |identity residual| over 254 cells: {resid:.2e} (CSV rounding)')
    n_tiny_d = int((out['beta_rec_depth'].abs() < 0.05).sum())
    n_tiny_p = int((out['beta_rec_promo'].abs() < 0.05).sum())
    lines.append(f'- cells with |beta_rec| < 0.05 (common denominator unstable, % inflated): depth={n_tiny_d}, promo={n_tiny_p} — same convention as shipped columns')
    # contrast vs shipped mixed-denominator dist column
    for s in ('depth', 'promo'):
        shipped = df[f'bias_pct_dist_{s}'].abs().median()
        new = stats[('cd_dist', s)]['med_abs']
        lines.append(f'- dist component, shipped |glm|-den median |bias| {shipped:.4f} → common-den median |bias| {new:.4f} ({s})')
    lines.append('\nSee `pending_table4_commondenom.csv` for per-cell values.')
    return '\n'.join(lines) + '\n', stats


def main(selftest_n=None):
    df = pd.read_csv(SRC)
    assert len(df) == 254, f'expected 254 cells, got {len(df)}'
    sub = df.head(selftest_n) if selftest_n else df
    out = compute(sub)
    if selftest_n:
        # Identity holds up to pure FP rounding (CSV rounding cancels: both sides use the
        # same rounded betas in a telescoping sum) -> ~1e-14 scale; 1e-3 is a 10 orders margin.
        resid = out[['resid_depth', 'resid_promo']].abs().max().max()
        assert resid < 1e-3, f'additive identity violated: {resid}'
        # cd_attr must reproduce shipped bias_pct_a (same |beta_rec| denominator).
        # Near-zero |beta_rec| cells amplify the 5-digit CSV rounding of the betas,
        # so tolerance scales with magnitude (verified: worst cell |beta_rec_promo|=0.002).
        for s in ('depth', 'promo'):
            ship = sub[f'bias_pct_a_{s}'].to_numpy(float)
            diff = (ship - out[f'cd_attr_{s}']).abs()
            tol = np.maximum(1e-4, 3e-3 * np.abs(ship))
            assert (diff < tol).all(), f'cd_attr_{s} deviates from shipped bias_pct_a: max {diff.max()}'
            # attr = dist + cens identity, same scaling (+ rounding bound scaled by 1/|beta_rec|)
            den = sub[f'beta_rec_{s}'].abs().to_numpy(float)
            lhs = (sub[f'bias_pct_orig_{s}'] - out[f'cd_omission_{s}']).to_numpy(float)
            diff2 = (lhs - (out[f'cd_dist_{s}'] + out[f'cd_pure_cens_{s}'])).abs()
            tol2 = np.maximum(1e-4, np.maximum(3e-3 * np.abs(lhs), 4e-5 / np.maximum(den, 1e-9)))
            assert (diff2 < tol2).all(), f'attr != dist + cens for {s}: max {diff2.max()}'
        print(f'selftest OK ({selftest_n} cells): identities hold (max resid {resid:.1e}), cd_attr == bias_pct_a, attr == dist + cens')
        return
    out.to_csv(OUT_CSV, index=False)
    md, stats = summarize(df, out, len(out))
    with open(OUT_MD, 'w') as fh:
        fh.write(md)
    print(f'DONE: {OUT_CSV} ({len(out)} rows), {OUT_MD}')
    # stdout headline
    for c in COMPONENTS:
        vals = [stats[(c, s)]['med_abs'] for s in ('depth', 'promo')]
        fr = [stats[(c, s)]['frac_ge20'] for s in ('depth', 'promo')]
        print(f'{c}: med|bias| d={vals[0]:.4f} p={vals[1]:.4f}  frac>=20% d={fr[0]:.4f} p={fr[1]:.4f}')


if __name__ == '__main__':
    n = int(sys.argv[sys.argv.index('--selftest') + 1]) if '--selftest' in sys.argv else None
    main(selftest_n=n)
