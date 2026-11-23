"""S10: Figure 5 from actual outputs (protocol §11). python figure5.py -> results/figure5.pdf, results/figure5.png (300 dpi)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = pd.read_csv('results/summaries.csv')
P = pd.read_csv('results/paired.csv')
CONDS = [(0.0, 0.2), (0.6, 0.2), (0.0, 0.6), (0.6, 0.6), (0.0, 0.95), (0.6, 0.95)]
CN = [f'eta{e}_q{q}' for e, q in CONDS]
LAB = [f'η={e:g}\nq={q:.2f}' for e, q in CONDS]
col = {0.0: '#4c72b0', 0.6: '#dd8452'}


def get(stat, cond):
    r = S[(S.statistic == stat) & (S.condition == cond)].iloc[0]
    return r.point, r.lo, r.hi, r.estimable if 'estimable' in r else True


fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))
x = np.arange(6)

# A: Delta_eta,q with 95% Monte Carlo intervals
a = ax[0, 0]
for i, c in enumerate(CN):
    p, lo, hi, est = get('Delta_lift_pp', c)
    if np.isnan(p):
        a.text(i, 0, 'not\nestimable', ha='center', va='center', fontsize=7)
        continue
    a.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt='o', color=col[CONDS[i][0]], capsize=4)
a.axhline(0, color='k', lw=0.8)
a.set_xticks(x, LAB, fontsize=8); a.set_ylabel('Median paired M − G lift error (pp)')
a.set_title('A  Δ(η,q): median of within-ID differences, 95% MC interval', fontsize=9, loc='left')

# B: paired-difference distributions at q = .60 (median + middle-80% range; NOT a CI)
b = ax[0, 1]
for j, e in enumerate([0.0, 0.6]):
    v = P[(P.condition == f'eta{e}_q0.6') & P.in_V].B_lift_pp.values
    parts = b.violinplot(v, positions=[j], showextrema=False, widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor(col[e]); pc.set_alpha(0.35)
    q10, q50, q90 = np.percentile(v, [10, 50, 90])
    b.vlines(j, q10, q90, color=col[e], lw=4, label='middle 80% of IDs' if j == 0 else None)
    b.plot(j, q50, 'ws', mec='k', ms=7, label='median' if j == 0 else None)
b.axhline(0, color='k', lw=0.8)
b.set_xticks([0, 1], ['η=0, q=0.60', 'η=0.6, q=0.60'], fontsize=8)
b.set_ylabel('B = A_M − A_G per replicate ID (pp)')
b.set_title('B  Distribution of paired differences at q = .60', fontsize=9, loc='left')
b.legend(fontsize=7, loc='lower right')

# C: Theta_q with intervals, q = .60 highlighted (primary)
c = ax[1, 0]
for i, q in enumerate([0.2, 0.6, 0.95]):
    p, lo, hi, est = get('Theta', f'q{q}')
    prim = q == 0.6
    if np.isnan(p):
        c.text(i, 0, 'not\nestimable', ha='center', va='center', fontsize=7)
        continue
    c.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt='D' if prim else 'o', ms=8 if prim else 6,
               color='#c44e52' if prim else '0.35', capsize=4, lw=2 if prim else 1)
c.axhline(0, color='k', lw=0.8)
c.set_xticks([0, 1, 2], ['q=0.20', 'q=0.60\n(primary)', 'q=0.95'], fontsize=8)
c.set_ylabel('Θ_q = median[B(η=.6) − B(η=0)] (pp)')
c.set_title('C  Dependence contrast Θ_q, 95% MC interval', fontsize=9, loc='left')

# D: 100 |V| / 1000
d = ax[1, 1]
V = [P[(P.condition == cn) & P.in_V].shape[0] for cn in CN]
d.bar(x, [100 * v / 1000 for v in V], color=[col[e] for e, _ in CONDS])
for i, v in enumerate(V):
    d.text(i, 100 * v / 1000 + 1, f'{v}', ha='center', fontsize=7)
d.set_ylim(0, 110); d.set_xticks(x, LAB, fontsize=8); d.set_ylabel('100·|V(η,q)| / 1000')
d.set_title('D  Jointly valid M and G fits', fontsize=9, loc='left')

fig.suptitle('Figure 5. Controlled simulation (1,000 replicate IDs × 6 conditions); negative = smaller error for M',
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('results/figure5.pdf')
fig.savefig('results/figure5.png', dpi=300)
print('saved results/figure5.pdf, results/figure5.png')
