"""P6b: per-attempt diagnostics missing from fit_diagnostics (instructions sec. 3):
  - realized censoring by promotion status, uncensored counts (from the frozen generated data)
  - counts of legacy safeguards that would activate at the reported coefficients: lambda clip (legacy 1e-8 / 1e6;
    NB2/ZIP 1e-8 / 1e5) and tail floor (strict log tail < log 1e-300) on censored rows
Coefficients: corrected full vectors (out/p3/coef_*.npz); NaN where no coefficient exists (no-finite-MLE screen).
Writes out/p6b_e2_extra.parquet (one row per E2 dataset x model)."""
import glob, math, os, sys
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
FLOOR = math.log(1e-300)


def cell(path):
    import mfit_corrected as mf
    z = np.load(path); city, cat = map(int, os.path.basename(path)[8:-4].split('_'))
    X = z['X']; promo = X[:, 1] > 0
    coefs = {}
    for f in glob.glob(os.path.join(H, 'out', 'p3', f'coef_{city}_{cat}_c*.npz')):
        c = np.load(f); coefs.update({k: c[k] for k in c.files})
    rows = []
    for i in range(len(z['rep'])):
        K = z['K'][i].astype(float); C = z['C'][i].astype(bool)
        base = dict(city=city, cat=cat, arm=str(z['arm'][i]), param=float(z['param'][i]), rep=int(z['rep'][i]),
                    cens_share_promo=float(C[promo].mean()) if promo.any() else np.nan,
                    cens_share_nonpromo=float(C[~promo].mean()) if (~promo).any() else np.nan,
                    n_uncensored=int((~C).sum()), n_uncensored_promo=int((~C & promo).sum()), n_uncensored_positive=int(((~C) & (K > 0)).sum()))
        for m in ('M', 'NB2', 'MIX'):
            b = coefs.get(f'{i}_{m}'); r = dict(base, model=m)
            if b is None:
                r.update(n_lambda_clip_low=np.nan, n_lambda_clip_high=np.nan, n_tail_floor=np.nan); rows.append(r); continue
            eta = X @ b[:14]; hi = 1e6 if m == 'M' else 1e5
            with np.errstate(all='ignore'):
                lam = np.exp(eta)
                if m == 'NB2':
                    ls = mf.nb_logsf(K[C], lam[C], math.exp(b[14])) if C.any() else np.array([])
                else:
                    ls = mf.pois_logsf(K[C], lam[C]) if C.any() else np.array([])
            r.update(n_lambda_clip_low=int((lam < 1e-8).sum()), n_lambda_clip_high=int((lam > hi).sum()), n_tail_floor=int((ls < FLOOR).sum()))
            rows.append(r)
    return rows


if __name__ == '__main__':
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    paths = sorted(glob.glob(os.path.join(H, 'frozen', 'e2_cell_*.npz')))
    with ProcessPoolExecutor(27, mp_context=mp.get_context('spawn')) as ex:
        out = [r for rs in ex.map(cell, paths) for r in rs]
    D = pd.DataFrame(out); D.to_parquet(os.path.join(H, 'out', 'p6b_e2_extra.parquet'))
    print(len(D)); print(D.groupby('model')[['n_lambda_clip_low', 'n_lambda_clip_high', 'n_tail_floor']].agg(lambda s: int((s > 0).sum())))
    print(D[D.model == 'M'].groupby(['arm', 'param'])[['cens_share_promo', 'cens_share_nonpromo']].median().round(3).to_string())
