"""Production / engineering runner. Checkpoint = one JSON per replicate ID; resume runs only missing IDs.

  python run.py --stream engineering --workers 5          # 5 smoke IDs (engineering children 1..5)
  python run.py --stream evaluation --workers 30          # IDs 1..1000 (evaluation child r-1)
Outputs: out/<stream>/rep_XXXX.json ; out/<stream>/timing.json
"""
import argparse, json, os, resource, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np


def seeds(stream):
    root = np.random.SeedSequence(20261122).spawn(3)
    if stream == 'evaluation':
        return {r: c for r, c in enumerate(root[0].spawn(1000), start=1)}
    return {i: c for i, c in enumerate(root[2].spawn(6)) if i >= 1}   # child 0 reserved for calibration MC check


def work(args):
    rid, ss, stream, outdir = args
    import sim
    t = time.time()
    res = sim.run_replicate(rid, ss, sim.load_calibration(), stream=stream)
    res['elapsed_s'] = time.time() - t
    res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    tmp = os.path.join(outdir, f'.rep_{rid:04d}.tmp')
    with open(tmp, 'w') as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(tmp, os.path.join(outdir, f'rep_{rid:04d}.json'))
    return rid, res['elapsed_s'], res['peak_rss_mb']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream', choices=['evaluation', 'engineering'], required=True)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    outdir = os.path.join('out', a.stream)
    os.makedirs(outdir, exist_ok=True)
    sd = seeds(a.stream)
    todo = [r for r in sd if not os.path.exists(os.path.join(outdir, f'rep_{r:04d}.json'))]
    print(f'{a.stream}: {len(sd)} scheduled, {len(sd) - len(todo)} done, {len(todo)} to run', flush=True)
    t0, times = time.time(), []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        futs = [ex.submit(work, (r, sd[r], a.stream, outdir)) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rid, el, rss = f.result()
            times.append((rid, el, rss))
            if i % 50 == 0 or i == len(todo):
                print(f'{i}/{len(todo)} done, wall {time.time() - t0:.0f}s', flush=True)
    json.dump(dict(stream=a.stream, workers=a.workers, n_run=len(todo), wall_s=time.time() - t0,
                   per_id=[dict(rid=r, elapsed_s=e, peak_rss_mb=m) for r, e, m in times]),
              open(os.path.join(outdir, 'timing.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
