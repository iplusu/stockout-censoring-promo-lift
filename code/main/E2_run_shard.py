"""E2 shard runner: same loop as E2_misspec_sim.main() but over a round-robin cell subset
(by estimated per-rep cost, so the 7 giant 20k cells spread across shards).
Shard i of 8 -> E2_misspec_cells_s{i}.csv. Launch via systemd-run (survives gateway restart).
Ponytail: duplicates main()'s ~30-line loop instead of refactoring the shipped script.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
m2 = importlib.import_module('E2_misspec_sim')
m05 = importlib.import_module('05_decomp')
m3 = importlib.import_module('03_main')

N_SHARD = int(os.environ.get('E2_SHARD', '0'))
N_SHARDS = int(os.environ.get('E2_NSHARDS', '8'))


def est_hours(n_rows):
    # from E2_bench.py measurements (2026-11-07): 1350->9.6s, 20000->87.4s per rep x13 params
    return max(0.00417 * n_rows + 3.97, 9.6) * 100 / 3600.0


def main():
    cells = m2.select_30_cells()
    decomp = pd.read_csv(m2.DECOMP_CSV)
    eligible = set(zip(cells.city.astype(int), cells['cat'].astype(int)))
    groups = m05.build_groups(eligible)
    # estimate cost per cell (row count after cap), deal round-robin heaviest-first
    costs = []
    for (city, cat) in sorted(eligible):
        d = m2.subsample_cell(m3.build_cell(groups[(city, cat)]))
        costs.append((city, cat, d['X'].shape[0]))
    costs.sort(key=lambda c: c[2], reverse=True)
    my = [c for i, c in enumerate(costs) if i % N_SHARDS == N_SHARD]
    print(f'shard {N_SHARD}/{N_SHARDS}: {len(my)} cells: {[(c, k) for c, k, _ in my]}', flush=True)

    anchor_keys = m2.anchor_assignment(cells)
    arm_param_map = {
        'A0_poisson': [None], 'A1_nb2': m2.NB2_ALPHAS, 'A2_zip': m2.ZIP_PSIS,
        'A3_label_noise': m2.LABEL_PI, 'A4_nonignorable': m2.RHOS, 'A5_combined': [None],
    }
    import json
    manifest = m2.build_manifest(arm_param_map, len(cells), anchor_keys,
                                  extra={'n_cells_this_shard': len(my), 'shard': f'{N_SHARD}/{N_SHARDS}'})
    mpath = os.path.join(os.path.dirname(HERE), 'experiments', f'E2_test_manifest_s{N_SHARD}.json')
    if not os.path.exists(mpath):
        with open(mpath, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

    all_rows = []
    for (city, cat, _) in my:
        d = m2.subsample_cell(m3.build_cell(groups[(city, cat)]))
        X, real_K, real_C = d['X'], d['K'], d['C']
        beta_vec, beta_p, beta_d, is_anchor = m2.true_beta_vec_for_cell(
            X, real_K, real_C, city, cat, decomp, anchor_keys)
        for arm, params in arm_param_map.items():
            recs = m2.run_cell_arm(X, real_K, real_C, beta_vec, arm, m2.N_REP,
                                   seed_base=city * 100 + cat, pi_names=params)
            for r in recs:
                r.update(city=city, cat=cat, beta_true_p=beta_p, beta_true_d=beta_d,
                         is_anchor=is_anchor)
            all_rows.extend(recs)
        print(f'shard{N_SHARD} cell ({city},{cat}) done, rows={len(all_rows)}', flush=True)

    df = pd.DataFrame(all_rows)
    # ponytail: E2_OUT_SUFFIX env lets a rerun (e.g. v2 SE capture) write alongside the original
    # shard output instead of overwriting it; empty by default so shipped behavior is unchanged.
    out = os.path.join(HERE, f'E2_misspec_cells_s{N_SHARD}{os.environ.get("E2_OUT_SUFFIX", "")}.csv')
    df.to_csv(out, index=False)
    print(f'SHARD {N_SHARD} DONE: {len(df)} rows -> {out}')


if __name__ == '__main__':
    main()
