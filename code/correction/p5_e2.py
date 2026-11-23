"""P5 E2: rerun the original E2 analysis chain (e2_analysis2 / e2_boot2 / e2_extra2, unchanged logic, seed 42,
4,000 city-cluster draws, both Holm families) on five shard sets:
  orig      : the 16 original shards (historical result)
  replay    : legacy estimators on the reconstructed inputs (naive/G/M from replay, NB2/MIX legacy from P3)
  corrected : corrected M/NB2/MIX (P3); validity = solver contract; non-valid -> excluded (NaN), naive/G unchanged
  legacy_on_common    : replay coefficients restricted to IDs valid under BOTH legacy and corrected (per estimator)
  corrected_on_common : corrected coefficients on the same common IDs
replay -> legacy_on_common isolates the validity-exclusion change; legacy_on_common -> corrected_on_common isolates the
estimate change on identical repeat IDs. Outputs: out/p5_e2/<set>/{headline,f1,f2,stdout}.
"""
import glob, os, re, subprocess, sys
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
KEY = ['city', 'cat', 'arm', 'param', 'rep']


def load_orig():
    sh = sorted(f for f in glob.glob(os.path.join(H, '..', 'main', 'E2_misspec_cells_s*.csv')) if re.search(r'_s\d+\.csv$', f))
    assert len(sh) == 16
    return pd.concat([pd.read_csv(f) for f in sh], ignore_index=True)


def base_and_fits():
    rp = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(H, 'out', 'e2_replay_*.csv'))], ignore_index=True)
    rp['param'] = rp['param'].fillna(-1.0)
    f = pd.read_parquet(os.path.join(H, 'out', 'p3_e2_fits.parquet'))
    return rp, f


def pivot(f, impl):
    s = f[f.impl == impl].copy()
    s['ok'] = s.valid if impl == 'corrected' else (s.status == 'legacy_conv')
    out = {}
    for m, pre in [('M', 'm'), ('NB2', 'nb2'), ('MIX', 'mix')]:
        t = s[s.model == m].set_index(KEY)
        if len(t):
            out[pre] = t[['b_p', 'b_d', 'ok']].rename(columns={'b_p': f'{pre}_p', 'b_d': f'{pre}_d', 'ok': f'{pre}_conv'})
    return out


def build_sets():
    orig = load_orig(); rp, f = base_and_fits()
    base = rp[KEY + ['naive_p', 'naive_d', 'g_p', 'g_d', 'g_conv', 'beta_true_p', 'beta_true_d', 'is_anchor']].set_index(KEY)
    legM = rp.set_index(KEY)[['m_p', 'm_d', 'm_conv']]
    L = pivot(f, 'legacy'); C = pivot(f, 'corrected')
    replay = base.join(legM).join(L['nb2']).join(L['mix'])
    corr = base.join(C['m']).join(C['nb2']).join(C['mix'])
    for pre in ['m', 'nb2', 'mix']:
        corr.loc[~corr[f'{pre}_conv'].astype(bool), [f'{pre}_p', f'{pre}_d']] = np.nan
    lc, cc = replay.copy(), corr.copy()
    for pre in ['m', 'nb2', 'mix']:
        both = replay[f'{pre}_conv'].astype(bool) & corr[f'{pre}_conv'].astype(bool)
        lc[f'{pre}_conv'] = both; cc[f'{pre}_conv'] = both
    sets = dict(orig=orig, replay=replay.reset_index(), corrected=corr.reset_index(),
                legacy_on_common=lc.reset_index(), corrected_on_common=cc.reset_index())
    for k, v in sets.items():
        v['param'] = v['param'].replace(-1.0, np.nan); assert len(v) == 35100, (k, len(v))
        for pre in ['g', 'm', 'nb2', 'mix']:
            v[f'{pre}_conv'] = v[f'{pre}_conv'].fillna(False).astype(bool)
    return sets


def run(name, df):
    d = os.path.join(H, 'out', 'p5_e2', name); os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, 'e2_all2.parquet'))
    log = open(os.path.join(d, 'stdout.txt'), 'w')
    for sc in ['e2_analysis2.py', 'e2_boot2.py', 'e2_extra2.py']:
        src = open(os.path.join(H, 'legacy', 'main', sc)).read().replace('/tmp/', d + '/')
        p = os.path.join(d, sc); open(p, 'w').write(src)
        log.write(f'##### {sc}\n'); log.flush()
        subprocess.run([PY, p], check=True, stdout=log, stderr=subprocess.STDOUT, cwd=d)
    return d


if __name__ == '__main__':
    sets = build_sets()
    for k in (sys.argv[1:] or sets):
        print(k, run(k, sets[k]), flush=True)
