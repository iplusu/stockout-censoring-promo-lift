"""E2 re-analysis on corrected data (post 4dae0dd fixes). Supersedes e2_analysis.py's
output on the invalid (SUPERSEDED) 35,100-row draw. Same metric/aggregation as before,
but G now uses its own convergence flag (g_conv, new column -- real poisson_glm_irls can
fail to converge, unlike the old always-succeeds IRLS warm-start)."""
import numpy as np, pandas as pd

df = pd.read_parquet('/tmp/e2_all2.parquet')

estimators = {'naive':('naive_p','naive_d',None), 'g':('g_p','g_d','g_conv'),
              'm':('m_p','m_d','m_conv'), 'nb2':('nb2_p','nb2_d','nb2_conv'),
              'mix':('mix_p','mix_d','mix_conv')}

def rel_err(df, p_col, d_col, conv_col):
    sub = df.copy()
    if conv_col is not None:
        sub = sub[sub[conv_col]==True]
    out = {}
    for coef,col,truecol in [('p',p_col,'beta_true_p'),('d',d_col,'beta_true_d')]:
        mask = sub[truecol].abs() >= 0.05
        e = (sub[col]-sub[truecol]).abs()/sub[truecol].abs()
        tmp = sub.loc[mask, ['arm','param','city','cat','rep','is_anchor']].copy()
        tmp['e'] = e[mask]
        tmp['coef']=coef
        out[coef]=tmp
    return pd.concat(out.values(), ignore_index=True)

all_err = []
for name,(pc,dc,cc) in estimators.items():
    e = rel_err(df, pc, dc, cc)
    e['estimator']=name
    all_err.append(e)
err = pd.concat(all_err, ignore_index=True)
err.to_parquet('/tmp/e2_err2.parquet')

cell_med = err.groupby(['arm','param','coef','estimator','city','cat'], dropna=False)['e'].median().reset_index()
headline = cell_med.groupby(['arm','param','coef','estimator'], dropna=False)['e'].median().reset_index()
headline_tbl = headline.pivot_table(index=['arm','param','coef'], columns='estimator', values='e')
print("=== Median relative error (CORRECTED data, 4dae0dd) ===")
print(headline_tbl.to_string())
headline_tbl.to_csv('/tmp/e2_headline_tbl2.csv')

n_tbl = err.groupby(['arm','param','coef','estimator'], dropna=False)['e'].count().reset_index()
print()
print("=== row counts (post convergence filter incl. new g_conv) ===")
print(n_tbl.pivot_table(index=['arm','param','coef'], columns='estimator', values='e').to_string())
