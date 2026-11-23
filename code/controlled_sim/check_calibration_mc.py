"""S3 independent check: realized Pr(D>C) under the actual generator (protocol §2-3 mechanism)
vs. calibrated q, 4e6 draws per cell, all 24 cells. Uses the 'engineering' seed child (not evaluation)."""
import numpy as np, pandas as pd
from scipy import stats
cal = pd.read_csv('calibration.csv')
eng = np.random.SeedSequence(20261122).spawn(3)[2]
rng = np.random.Generator(np.random.PCG64(eng.spawn(1)[0]))
N = 4_000_000; worst = 0
for _, r in cal.iterrows():
    zd, zc = rng.standard_normal(N), rng.standard_normal(N)
    D = stats.poisson.ppf(stats.norm.cdf(zd), r.lambda_s)
    C = stats.poisson.ppf(stats.norm.cdf(r.eta * zd + np.sqrt(1 - r.eta**2) * zc), r.kappa)
    phat = np.mean(D > C); z = (phat - r.q) / np.sqrt(r.q * (1 - r.q) / N); worst = max(worst, abs(z))
    print(f'{r.stratum:15s} eta={r.eta} q={r.q}: MC={phat:.5f} z={z:+.2f}')
assert worst < 4.0, worst   # 24 cells, |z|<4 bound
print(f'MC CHECK PASS (max |z| = {worst:.2f})')
