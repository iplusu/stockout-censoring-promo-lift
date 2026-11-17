import numpy as np, pandas as pd
err = pd.read_parquet('/tmp/e2_err2.parquet')

def city_cluster_boot(sub, est_a, est_b, n_boot=4000, seed=42):
    a = sub[sub.estimator==est_a].set_index(['city','cat','rep'])['e']
    b = sub[sub.estimator==est_b].set_index(['city','cat','rep'])['e']
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    d = (a - b)
    cellkey = pd.Series([(t[0],t[1]) for t in idx], index=idx)
    cell_med = d.groupby(cellkey).median()
    cities = pd.Series([t[0] for t in cell_med.index])
    point = cell_med.median()
    rng = np.random.default_rng(seed)
    uniq_cities = cities.unique()
    cell_by_city = {c: cell_med.values[cities.values==c] for c in uniq_cities}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        samp = rng.choice(uniq_cities, size=len(uniq_cities), replace=True)
        boots[i] = np.median(np.concatenate([cell_by_city[c] for c in samp]))
    lo, hi = np.percentile(boots, [2.5,97.5])
    frac_le0, frac_ge0 = (boots<=0).mean(), (boots>=0).mean()
    p = min(2*min(frac_le0, frac_ge0), 1.0)
    return point, lo, hi, len(uniq_cities), len(cell_med), p

def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0
    for rank,i in enumerate(order):
        a = (m-rank)*pvals[i]
        running = max(running, a)
        adj[i] = min(running,1.0)
    return adj

# Family 1: M vs NB2 (A1), M vs MIX (A2) -- same family as before, corrected data
fam1 = []
for param in [0.10,0.50,1.00]:
    for coef in ['p','d']:
        fam1.append(('A1_nb2', param, coef, 'm','nb2'))
for param in [0.05,0.15]:
    for coef in ['p','d']:
        fam1.append(('A2_zip', param, coef, 'm','mix'))
rows=[]
for arm,param,coef,ea,eb in fam1:
    sub = err[(err.arm==arm)&(err.param==param)&(err.coef==coef)]
    point,lo,hi,ng,nc,p = city_cluster_boot(sub, ea, eb)
    rows.append(dict(family='F1_dist_match',arm=arm,param=param,coef=coef,compare=f'{ea}_minus_{eb}',
                      point=point,ci_lo=lo,ci_hi=hi,n_cities=ng,n_cells=nc,p_raw=p))
f1 = pd.DataFrame(rows)
f1['p_holm'] = holm(f1['p_raw'].values)
f1['sig_holm'] = f1['p_holm']<0.05
print("=== Family 1 (corrected data) ===")
print(f1.to_string(index=False))
f1.to_csv('/tmp/e2_f1_tests2.csv', index=False)

# Family 2 (REVISED): does M fail under genuine non-ignorable censoring (A4)? m vs naive, m vs g
fam2 = []
for param in [0.0,0.5,1.0]:
    for coef in ['p','d']:
        fam2.append(('A4_nonignorable', param, coef, 'm','naive'))
        fam2.append(('A4_nonignorable', param, coef, 'm','g'))
rows=[]
for arm,param,coef,ea,eb in fam2:
    sub = err[(err.arm==arm)&(err.param==param)&(err.coef==coef)]
    point,lo,hi,ng,nc,p = city_cluster_boot(sub, ea, eb)
    rows.append(dict(family='F2_nonignorable_m_vs_naive_g',arm=arm,param=param,coef=coef,compare=f'{ea}_minus_{eb}',
                      point=point,ci_lo=lo,ci_hi=hi,n_cities=ng,n_cells=nc,p_raw=p))
f2 = pd.DataFrame(rows)
f2['p_holm'] = holm(f2['p_raw'].values)
f2['sig_holm'] = f2['p_holm']<0.05
print()
print("=== Family 2 REVISED: M vs naive/G under genuine non-ignorable A4 ===")
print(f2.to_string(index=False))
f2.to_csv('/tmp/e2_f2_tests2.csv', index=False)
