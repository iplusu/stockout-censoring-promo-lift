"""P1 gate (Q2-B): replayed naive/G/legacy-M vs stored canonical shards, key (city,cat,arm,param,rep).
exact replay  <=> every key present once and |diff| <= 1e-10 for naive_p/d, g_p/d, m_p/d and identical g_conv/m_conv,
                  beta_true_p/d and is_anchor identical.
  python compare_replay.py [--cells city:cat,...]   -> out/replay_compare.csv + printed verdict
"""
import argparse, glob, os
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
NUM = ['naive_p', 'naive_d', 'g_p', 'g_d', 'm_p', 'm_d', 'beta_true_p', 'beta_true_d']
FLG = ['g_conv', 'm_conv', 'is_anchor']
KEY = ['city', 'cat', 'arm', 'param', 'rep']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cells', default=None); a = ap.parse_args()
    st = pd.concat([pd.read_csv(os.path.join(os.path.dirname(H), 'main', f'E2_misspec_cells_s{i}.csv')) for i in range(16)])
    rp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(H, 'out', 'e2_replay_*.csv')))])
    for d in (st, rp):
        d['param'] = d['param'].fillna(-1.0).astype(float)
    if a.cells:
        sel = {tuple(map(int, c.split(':'))) for c in a.cells.split(',')}
        st = st[[(c, k) in sel for c, k in zip(st.city, st.cat)]]
    done = set(zip(rp.city, rp.cat))
    st = st[[(c, k) in done for c, k in zip(st.city, st.cat)]]
    j = st.merge(rp, on=KEY, how='outer', suffixes=('_s', '_r'), indicator=True)
    out = dict(cells=len(done), stored_rows=len(st), replay_rows=len(rp), unmatched=int((j._merge != 'both').sum()))
    b = j[j._merge == 'both']
    for c in NUM:
        d = (b[c + '_s'] - b[c + '_r']).abs()
        both_nan = b[c + '_s'].isna() & b[c + '_r'].isna()
        out[f'max|d| {c}'] = float(d[~both_nan].max()) if (~both_nan).any() else 0.0
        out[f'n>1e-10 {c}'] = int(((d > 1e-10) | (b[c + '_s'].isna() ^ b[c + '_r'].isna())).sum())
    for c in FLG:
        out[f'n_mismatch {c}'] = int((b[c + '_s'].astype(bool) != b[c + '_r'].astype(bool)).sum())
    bad = out['unmatched'] + sum(v for k, v in out.items() if k.startswith('n>') or k.startswith('n_mismatch'))
    out['verdict'] = 'EXACT_REPLAY' if bad == 0 else 'MISMATCH'
    pd.Series(out).to_csv(os.path.join(H, 'out', 'replay_compare.csv'), header=False)
    for k, v in out.items():
        print(f'{k:24s} {v}')


if __name__ == '__main__':
    main()
