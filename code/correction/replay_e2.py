"""P1: E2 replay through the byte-frozen legacy code (correction_20261122/legacy/).

Calls legacy `run_cell_arm` UNCHANGED for every (cell, arm), so random-number consumption, data
generation, naive/G/legacy-M fits are the original code path. Two hooks only:
  * capture: wrap legacy `tobit.init_poisson_irls` (called exactly once per generated dataset inside
    run_cell_arm, before M) to record (K, C) of every dataset;
  * skip NB2/MIX (stub raises -> NaN): not part of the Q2-B certification and unaffected by M.
Neither hook consumes random numbers (verified: fitters do not use the generator).

  python replay_e2.py --cells all --workers 14     # -> frozen/e2_cell_<city>_<cat>.npz, out/e2_replay_<city>_<cat>.csv
"""
import argparse, os, sys, time, json, hashlib
import numpy as np

LEG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'legacy')


def load_legacy():
    sys.path.insert(0, os.path.join(LEG, 'main'))
    sys.path.insert(0, os.path.join(LEG, 'pilot'))
    import importlib
    m2 = importlib.import_module('E2_misspec_sim')
    assert os.path.dirname(m2.__file__) == os.path.join(LEG, 'main'), m2.__file__
    assert os.path.dirname(m2.tobit.__file__) == os.path.join(LEG, 'pilot'), m2.tobit.__file__
    # legacy reads its CSV inputs relative to its own dir -> point at the frozen originals
    live = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main')
    m2.DECOMP_CSV = os.path.join(live, 'decomp_cells.csv')
    m2.CELLS_CSV = os.path.join(live, 'main_cells.csv')
    m2.SAMPLED_CELLS_CSV = os.path.join(os.path.dirname(live), 'experiments', 'E2_sampled_cells.csv')
    return m2


def run_cell(args):
    city, cat, outdir, frozendir = args
    import pandas as pd
    m2 = load_legacy()
    m05, m3, tobit = m2.m05, m2.m3, m2.tobit
    cells = m2.select_30_cells()
    anchor_keys = m2.anchor_assignment(cells)
    decomp = pd.read_csv(m2.DECOMP_CSV)
    groups = m05.build_groups({(city, cat)})
    full = m3.build_cell(groups[(city, cat)])
    d = m2.subsample_cell(full)
    X, real_K, real_C = d['X'], d['K'], d['C']
    beta_vec, beta_p, beta_d, is_anchor = m2.true_beta_vec_for_cell(X, real_K, real_C, city, cat, decomp, anchor_keys)
    # hooks (installed AFTER the generating-beta fit, which also calls init_poisson_irls)
    cap = []
    real_init = tobit.init_poisson_irls

    def init_capture(X_, K_, C_):
        cap.append((np.asarray(K_, np.int64).copy(), np.asarray(C_, np.int8).copy()))
        return real_init(X_, K_, C_)
    tobit.init_poisson_irls = init_capture

    def _skip(*a, **k):
        raise RuntimeError('NB2/MIX skipped in replay (not in certification scope)')
    m2.m07.fit_nb2 = _skip
    m2.m07.fit_zip = _skip
    arm_param_map = {'A0_poisson': [None], 'A1_nb2': m2.NB2_ALPHAS, 'A2_zip': m2.ZIP_PSIS,
                     'A3_label_noise': m2.LABEL_PI, 'A4_nonignorable': m2.RHOS, 'A5_combined': [None]}
    rows, keys, Ks, Cs = [], [], [], []
    t0 = time.time()
    for arm, params in arm_param_map.items():
        n0 = len(cap)
        recs = m2.run_cell_arm(X, real_K, real_C, beta_vec, arm, m2.N_REP, seed_base=city * 100 + cat, pi_names=params)
        assert len(cap) - n0 == len(recs), (arm, len(cap) - n0, len(recs))
        for r, (K, C) in zip(recs, cap[n0:]):
            r.update(city=city, cat=cat, beta_true_p=beta_p, beta_true_d=beta_d, is_anchor=is_anchor)
            rows.append(r)
            keys.append((arm, -1.0 if r['param'] is None else float(r['param']), int(r['rep'])))
            Ks.append(K); Cs.append(C)
        cap.clear()
    el = time.time() - t0
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    np.savez_compressed(os.path.join(frozendir, f'e2_cell_{city}_{cat}.npz'),
                        X=X, real_K=real_K, real_C=real_C, beta_vec=beta_vec, n_full=full['X'].shape[0],
                        arm=np.array([k[0] for k in keys]), param=np.array([k[1] for k in keys]),
                        rep=np.array([k[2] for k in keys]), K=np.stack(Ks).astype(np.int32), C=np.stack(Cs).astype(bool))
    df = pd.DataFrame(rows)
    df['K_sha16'] = [sha(k)[:16] for k in Ks]
    df['C_sha16'] = [sha(c)[:16] for c in Cs]
    df.to_csv(os.path.join(outdir, f'e2_replay_{city}_{cat}.csv'), index=False, float_format='%.17g')
    return city, cat, len(rows), X.shape, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='all')       # 'all' or 'city:cat,city:cat'
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--nrep', type=int, default=None)  # benchmarking only
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir, frozendir = os.path.join(here, 'out'), os.path.join(here, 'frozen')
    os.makedirs(outdir, exist_ok=True); os.makedirs(frozendir, exist_ok=True)
    if a.cells == 'all':
        import pandas as pd
        s = pd.concat([pd.read_csv(os.path.join(os.path.dirname(here), 'main', f'E2_misspec_cells_s{i}.csv'),
                                   usecols=['city', 'cat']) for i in range(16)]).drop_duplicates()
        cells = sorted(map(tuple, s.values.tolist()))
    else:
        cells = [tuple(map(int, c.split(':'))) for c in a.cells.split(',')]
    cells = [c for c in cells if not os.path.exists(os.path.join(frozendir, f'e2_cell_{c[0]}_{c[1]}.npz'))]
    print(f'{len(cells)} cells to replay', flush=True)
    if a.nrep:
        load_legacy().N_REP = a.nrep  # noqa (benchmark path; workers=1 only)
    if a.workers == 1:
        for c in cells:
            if a.nrep:
                m2 = load_legacy(); m2.N_REP = a.nrep
            print(run_cell((c[0], c[1], outdir, frozendir)), flush=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
            for f in as_completed([ex.submit(run_cell, (c[0], c[1], outdir, frozendir)) for c in cells]):
                print(f.result(), time.strftime('%H:%M:%S'), flush=True)


if __name__ == '__main__':
    main()
