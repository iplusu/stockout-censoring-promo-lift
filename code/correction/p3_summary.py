"""P3 summary tables: censoring share and corrected-M status by arm x param; legacy M on nonexistent datasets."""
import numpy as np, pandas as pd, glob
f = pd.read_parquet('out/p3_e2_fits.parquet')
c = f[(f.impl == 'corrected') & (f.model == 'M')].copy()
g = c.groupby(['arm', 'param'])
t = pd.DataFrame({'n': g.size(), 'median_cens': g.cens_share.median(), 'share_cens_ge_0.99': g.cens_share.apply(lambda s: (s >= .99).mean()),
                  'valid': g.valid.sum(), 'nonexist': g.status.apply(lambda s: (s == 'structural_nonexistence').sum()),
                  'nonconv': g.status.apply(lambda s: (s == 'nonconvergence').sum())})
print(t.round(3).to_string())
rp = pd.concat([pd.read_csv(x) for x in glob.glob('out/e2_replay_*.csv')])
rp['param'] = rp['param'].fillna(-1.0)
j = c.merge(rp[['city', 'cat', 'arm', 'param', 'rep', 'm_p', 'm_d', 'm_conv']], on=['city', 'cat', 'arm', 'param', 'rep'])
ne = j[j.status == 'structural_nonexistence']
print('legacy M on the', len(ne), 'no-finite-MLE datasets (same inputs): m_conv True', int(ne.m_conv.sum()),
      '| |m_p| median', round(ne.m_p.abs().median(), 3), 'p90', round(ne.m_p.abs().quantile(.9), 1), '| |m_d| median', round(ne.m_d.abs().median(), 3))
t.to_csv('out/p3_status_by_condition.csv')
