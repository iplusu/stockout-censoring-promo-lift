"""P3: refit every frozen E2 dataset (reconstruction) with corrected M/NB2/MIX and legacy NB2/MIX.
Legacy naive/G/M on the same inputs already exist in out/e2_replay_*.csv (P1).
  python p3_fit.py --cell 13:6 [--limit N]            # one cell (limit = first N datasets, timing only)
  python p3_fit.py --all --workers W                   # production; resume = cells without out/p3/<cell>.csv
Outputs per cell: out/p3/fits_<city>_<cat>.csv (one row per dataset x model), out/p3/starts_<city>_<cat>.csv,
                  out/p3/coef_<city>_<cat>.npz (full coefficient vectors)."""
import argparse, glob, json, math, os, sys, time
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))


def legacy():
    sys.path.insert(0, os.path.join(H, 'legacy', 'main')); sys.path.insert(0, os.path.join(H, 'legacy', 'pilot'))
    import importlib
    return importlib.import_module('tobit'), importlib.import_module('07_pending_3_altmodels')


def run_cell(args):
    city, cat, limit, lo, hi = (args + (0, None))[:5] if len(args) == 3 else args
    import pandas as pd, mfit_corrected as mf
    tobit, m07 = legacy()
    z = np.load(os.path.join(H, 'frozen', f'e2_cell_{city}_{cat}.npz'))
    X = z['X']; p = X.shape[1]; n_ds = len(z['rep']) if not limit else min(limit, len(z['rep']))
    hi = n_ds if hi is None else min(hi, n_ds)
    rows, srows, coefs = [], [], {}
    t0 = time.time()
    for i in range(lo, hi):
        K = z['K'][i].astype(float); I = z['C'][i].astype(bool)
        key = dict(city=city, cat=cat, arm=str(z['arm'][i]), param=float(z['param'][i]), rep=int(z['rep'][i]),
                   n=len(K), cens_share=float(I.mean()))
        with np.errstate(all='ignore'):
            b0 = tobit.init_poisson_irls(X, K, I.astype(int))
        recs = {}
        e_p, e_d = np.zeros(p), np.zeros(p); e_p[1] = .1; e_d[2] = .1
        t = time.time(); fM, sM = mf.fit('M', X, K, I, [b0, b0 + e_p, b0 + e_d]); recs['M'] = (fM, sM, time.time() - t)
        bM = fM['x'] if fM['valid'] else None
        stN = [np.append(b0, math.log(5.0))] + ([np.append(bM, math.log(5.0))] if bM is not None else [])
        stZ = [np.append(b0, math.log(.1 / .9))] + ([np.append(bM, math.log(.1 / .9))] if bM is not None else [])
        t = time.time(); recs['NB2'] = (*mf.fit('NB2', X, K, I, stN), time.time() - t)
        t = time.time(); recs['MIX'] = (*mf.fit('MIX', X, K, I, stZ), time.time() - t)
        for m, (f, s, el) in recs.items():
            x = f.pop('x', None)
            if x is not None:
                coefs[f'{i}_{m}'] = x
            rows.append({**key, 'impl': 'corrected', **f, 'b_p': x[1] if x is not None else np.nan,
                         'b_d': x[2] if x is not None else np.nan, 'aux': x[-1] if (x is not None and m != 'M') else np.nan,
                         'seconds': el})
            srows += [{**key, 'impl': 'corrected', 'model': m, **r} for r in s]
        # legacy NB2/MIX on the same reconstructed inputs (same legacy starts as the original E2)
        for m, fn in [('NB2', m07.fit_nb2), ('MIX', m07.fit_zip)]:
            t = time.time()
            try:
                with np.errstate(all='ignore'):
                    g = fn(X, K, I.astype(int), b0)
                rows.append({**key, 'impl': 'legacy', 'model': m, 'status': 'legacy_conv' if g['converged'] else 'legacy_nonconv',
                             'valid': bool(g['converged']), 'b_p': g['b'][1], 'b_d': g['b'][2],
                             'legacy_grad_norm_sum_L2': g['grad_norm'], 'seconds': time.time() - t})
            except Exception as e:
                rows.append({**key, 'impl': 'legacy', 'model': m, 'status': f'legacy_exception:{type(e).__name__}',
                             'valid': False, 'b_p': np.nan, 'b_d': np.nan, 'seconds': time.time() - t})
    tag = f'{city}_{cat}' + (f'_lim{limit}' if limit else '') + (f'_c{lo:04d}' if (lo or hi != n_ds) else '')
    os.makedirs(os.path.join(H, 'out', 'p3'), exist_ok=True)
    pd.DataFrame(srows).to_csv(os.path.join(H, 'out', 'p3', f'starts_{tag}.csv'), index=False, float_format='%.17g')
    np.savez_compressed(os.path.join(H, 'out', 'p3', f'coef_{tag}.npz'), **coefs)
    pd.DataFrame(rows).to_csv(os.path.join(H, 'out', 'p3', f'fits_{tag}.csv'), index=False, float_format='%.17g')
    return city, cat, lo, hi, X.shape, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell'); ap.add_argument('--all', action='store_true')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=1); ap.add_argument('--chunk', type=int, default=100)
    a = ap.parse_args()
    if a.cell:
        c, k = map(int, a.cell.split(':')); print(run_cell((c, k, a.limit)), flush=True); return
    cells = sorted(tuple(map(int, os.path.basename(f)[8:-4].split('_'))) for f in glob.glob(os.path.join(H, 'frozen', 'e2_cell_*.npz')))
    tasks = []
    for c, k in cells:
        n_ds = len(np.load(os.path.join(H, 'frozen', f'e2_cell_{c}_{k}.npz'))['rep'])
        for lo in range(0, n_ds, a.chunk):   # ponytail: fixed chunks; the tag encodes lo so resume skips finished chunks
            if not os.path.exists(os.path.join(H, 'out', 'p3', f'fits_{c}_{k}_c{lo:04d}.csv')):
                tasks.append((c, k, 0, lo, lo + a.chunk))
    tasks.sort(key=lambda t: -np.load(os.path.join(H, 'frozen', f'e2_cell_{t[0]}_{t[1]}.npz'))['X'].shape[0])  # heaviest first
    print(f'{len(tasks)} chunks to fit', flush=True)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        for f in as_completed([ex.submit(run_cell, t) for t in tasks]):
            print(f.result(), time.strftime('%H:%M:%S'), flush=True)


if __name__ == '__main__':
    main()
