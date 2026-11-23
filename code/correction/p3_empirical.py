"""P3 (empirical): the same 254 cells, full rows, legacy vs corrected on identical inputs.
Per cell: legacy M (replay of 04_elasticity path; compared with stored elast_cells beta_rec), corrected M (3 starts),
corrected NB2/MIX (starts: corrected-M optimum, legacy b0), legacy NB2/MIX (current legacy code, init at legacy M
optimum, as 07_pending_3_altmodels does). Full coefficient vectors saved.
  python p3_empirical.py --cells 13:6,1:10      # test
  python p3_empirical.py --all --workers 4       # production; resume = cells with out/p3emp/<city>_<cat>.json
"""
import argparse, json, math, os, sys, time
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(os.path.dirname(H), 'main')


def legacy():
    sys.path.insert(0, os.path.join(H, 'legacy', 'main')); sys.path.insert(0, os.path.join(H, 'legacy', 'pilot'))
    import importlib
    return (importlib.import_module('tobit'), importlib.import_module('05_decomp'), importlib.import_module('03_main'),
            importlib.import_module('07_pending_3_altmodels'))


def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
    return o


def run_batch(cells):
    import mfit_corrected as mf
    tobit, m05, m3, m07 = legacy()
    groups = m05.build_groups(set(cells))
    done = []
    for (city, cat) in cells:
        t0 = time.time()
        d = m3.build_cell(groups[(city, cat)])
        X, K, C = d['X'], d['K'].astype(float), d['C']
        I = C.astype(bool); p = X.shape[1]
        out = dict(city=city, cat=cat, n=int(X.shape[0]), p=p, n_cens=int(I.sum()), n_promo=int((d['promo'] == 1).sum()))
        with np.errstate(all='ignore'):
            b0 = tobit.init_poisson_irls(X, K, C)
            t = time.time(); fl = tobit.fit(X, K, C, b0=b0)
        out['legacy_M'] = dict(b=fl['b'], converged=fl['converged'], success=bool(fl['success']), nit=fl['nit'],
                               grad_norm_sum_L2=fl['grad_norm'], seconds=time.time() - t)
        P = mf.Problem(X, K, I)
        out['legacy_M']['corrected_obj_at_legacy_b'] = float(mf.m_nll(fl['b'], P))
        out['legacy_M']['corrected_gsup_at_legacy_b'] = float(np.max(np.abs(mf.m_grad(fl['b'], P))))
        e_p, e_d = np.zeros(p), np.zeros(p); e_p[1] = .1; e_d[2] = .1
        t = time.time(); fM, sM = mf.fit('M', X, K, I, [b0, b0 + e_p, b0 + e_d]); fM['seconds'] = time.time() - t
        out['corrected_M'] = dict(fM, starts=sM)
        bM = fM.get('x')
        for m, aux in [('NB2', math.log(5.0)), ('MIX', math.log(.1 / .9))]:
            st = ([np.append(bM, aux)] if bM is not None else []) + [np.append(b0, aux)]
            t = time.time(); f, s = mf.fit(m, X, K, I, st); f['seconds'] = time.time() - t
            out[f'corrected_{m}'] = dict(f, starts=s)
        for m, fn, kw in [('NB2', m07.fit_nb2, dict(theta0=5.0)), ('MIX', m07.fit_zip, {})]:
            t = time.time()
            try:
                with np.errstate(all='ignore'):
                    g = fn(X, K, C, fl['b'], **kw)
                out[f'legacy_{m}'] = dict(b=g['b'], converged=g['converged'], nit=g['nit'], grad_norm_sum_L2=g['grad_norm'],
                                          aux=g.get('theta', g.get('pi')), seconds=time.time() - t)
            except Exception as e:
                out[f'legacy_{m}'] = dict(error=f'{type(e).__name__}: {e}', seconds=time.time() - t)
        out['seconds_total'] = time.time() - t0
        os.makedirs(os.path.join(H, 'out', 'p3emp'), exist_ok=True)
        tmp = os.path.join(H, 'out', 'p3emp', f'.{city}_{cat}.tmp')
        json.dump(jsonable(out), open(tmp, 'w'))
        os.replace(tmp, os.path.join(H, 'out', 'p3emp', f'{city}_{cat}.json'))
        done.append((city, cat, out['n'], round(out['seconds_total'], 1)))
    return done


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cells'); ap.add_argument('--all', action='store_true')
    ap.add_argument('--workers', type=int, default=1); a = ap.parse_args()
    import pandas as pd
    if a.cells:
        cells = [tuple(map(int, c.split(':'))) for c in a.cells.split(',')]
    else:
        e = pd.read_csv(os.path.join(LIVE, 'elast_cells.csv')); e = e[e.status == 'ok']
        cells = sorted(zip(e.city.astype(int), e['cat'].astype(int)))
    cells = [c for c in cells if not os.path.exists(os.path.join(H, 'out', 'p3emp', f'{c[0]}_{c[1]}.json'))]
    print(f'{len(cells)} cells to fit', flush=True)
    if a.workers == 1:
        print(run_batch(cells), flush=True); return
    # batches of ~8 cells, heaviest (by stored n_rows) spread round-robin
    e = pd.read_csv(os.path.join(LIVE, 'elast_cells.csv')).set_index(['city', 'cat'])['n_rows']
    cells.sort(key=lambda c: -e.get(c, 0))
    batches = [cells[i::max(1, len(cells) // 8)] for i in range(max(1, len(cells) // 8))]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        for f in as_completed([ex.submit(run_batch, b) for b in batches]):
            print(f.result(), time.strftime('%H:%M:%S'), flush=True)


if __name__ == '__main__':
    main()
