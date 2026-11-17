import numpy as np, pandas as pd
err = pd.read_parquet('/tmp/e2_err2.parquet')

print("=== A0 self-check (corrected data) ===")
a0 = err[err.arm=='A0_poisson']
print(a0.groupby(['coef','estimator'])['e'].median())

print()
print("=== Full descriptive table, all arms, all estimators (median error) ===")
cellmed = err.groupby(['arm','coef','estimator','city','cat'])['e'].median().reset_index()
agg = cellmed.groupby(['arm','coef','estimator'])['e'].median().reset_index()
print(agg.pivot_table(index=['arm','coef'], columns='estimator', values='e').to_string())

print()
print("=== A4 extreme-value sanity check: how many CONVERGED m rows have e>1 at rho=0.5/1.0? ===")
a4 = err[(err.arm=='A4_nonignorable') & (err.estimator=='m') & (err.param.isin([0.5,1.0]))]
print(a4.groupby('param')['e'].describe())
