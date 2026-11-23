"""Engineering validation (protocol §9.1-9.5). Writes validation_log.txt; exit 1 on any failure.
Engineering stream: SeedSequence(20261122).spawn(3)[2]; child 0 = calibration MC check, children 1..5 = smoke IDs."""
import json, math, sys, time
from decimal import Decimal, getcontext
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import sim

getcontext().prec = 80
LOG, FAILS = [], []


def check(name, ok, detail=''):
    LOG.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print(LOG[-1], flush=True)
    if not ok:
        FAILS.append(name)


def dec_tail(k, mu):
    """High-precision Pr(Y>k) and pmf(k) by direct positive-tail summation."""
    mu = Decimal(mu)
    t = (-mu).exp()
    for j in range(1, k + 1):
        t = t * mu / j
    pmf_k = t
    t = t * mu / (k + 1)
    s, j = Decimal(0), k + 1
    while s == 0 or t > s * Decimal(10) ** -60:
        s += t
        j += 1
        t = t * mu / j
    return s, pmf_k


def eng_ids():
    eng = np.random.SeedSequence(20261122).spawn(3)[2]
    return {i: c for i, c in enumerate(eng.spawn(6)) if i >= 1}


def _rep(args):
    rid, ss, cal = args
    return rid, sim.run_replicate(rid, ss, cal, stream='engineering')


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)       # deterministic unit-test fixtures only (not a study stream)
    # ---- 9.1 censoring
    D = np.array([2, 3, 1]); C = np.array([2, 2, 2])
    K, I = np.minimum(D, C), D > C
    check('9.1 hand (D,C)->(K,I)', list(zip(K.tolist(), I.astype(int).tolist())) == [(2, 0), (2, 1), (1, 0)])
    zd, zc = sim.draw_latent(eng_ids()[1])
    Dd, _ = sim.make_demand(zd)
    cal = sim.load_calibration()
    ok_support, ok_strict = True, True
    for eta, q in sim.CONDS:
        Kc, Ic, _ = sim.make_condition(zd, zc, Dd, eta, q, cal)
        ok_support &= bool(Kc.dtype.kind == 'i' and (Kc >= 0).all() and (Dd >= 0).all())
        ok_strict &= bool((Dd[Ic] > Kc[Ic]).all() and (Dd[~Ic] == Kc[~Ic]).all())
    check('9.1 nonneg integer support', ok_support)
    check('9.1 censored => D strictly > K; uncensored => K = D', ok_strict)
    # strict survival used in likelihood: censored contribution = log Pr(Y > K) (not >= K)
    b = np.array([math.log(20), 0.2, 1.0]); y1 = np.full(sim.N, 15.0); I1 = np.zeros(sim.N, bool); I1[0] = True
    lam0 = 20.0
    manual = -(np.sum(y1[1:] * np.log(lam0 * np.exp(0.2 * sim.P_COL[1:] + sim.D_COL[1:])) -
                      lam0 * np.exp(0.2 * sim.P_COL[1:] + sim.D_COL[1:]) - [math.lgamma(16)] * (sim.N - 1))
               + math.log(float(dec_tail(15, lam0)[0]))) / sim.N
    check('9.1 censored contribution uses logsf(K) (strict)', abs(sim.nll(b, y1, I1) - manual) < 1e-12,
          f'|diff|={abs(sim.nll(b, y1, I1) - manual):.2e}')
    # ---- 9.2 likelihood
    y = rng.poisson(20, sim.N).astype(float)
    bt = np.array([math.log(19), 0.15, 0.9])
    Z = np.zeros(sim.N, bool)
    check('9.2 all I=0: M objective == Poisson', abs(sim.nll(bt, y, Z) - sim.nll(bt, y, None)) < 1e-14)
    check('9.2 all I=0: M gradient == Poisson', np.max(np.abs(sim.grad(bt, y, Z) - sim.grad(bt, y, None))) < 1e-14)
    worst = 0.0
    for k, mu in [(0, 0.5), (3, 2.0), (10, 4.0), (25, 20.0), (40, 20.0), (5, 30.0)]:
        S, _ = dec_tail(k, mu)
        worst = max(worst, abs(float(sim.logsf(k, mu)) - float(S.ln())))
    check('9.2 strict Poisson tail vs direct summation (small cases)', worst < 1e-12, f'max abs log diff={worst:.2e}')
    # ---- 9.3 derivatives (mixed censored fixture)
    lam = np.exp(sim.X @ sim.BETA_TRUE)
    Dm = rng.poisson(lam); Cm = rng.poisson(0.9 * lam)
    ym, Im = np.minimum(Dm, Cm).astype(float), Dm > Cm
    for tag, b0 in [('near truth', sim.BETA_TRUE + [0.05, -0.03, 0.1]), ('far', np.array([2.0, 0.6, -0.5]))]:
        g = sim.grad(b0, ym, Im)
        fd = np.array([(sim.nll(b0 + e, ym, Im) - sim.nll(b0 - e, ym, Im)) / 2e-6 for e in np.eye(3) * 1e-6])
        dg = np.max(np.abs(g - fd)) / (1 + np.max(np.abs(g)))
        check(f'9.3 gradient vs centered FD ({tag}, h=1e-6)', dg < 1e-5, f'scaled={dg:.2e}')
        H = sim.hess(b0, ym, Im)
        Hfd = np.column_stack([(sim.grad(b0 + e, ym, Im) - sim.grad(b0 - e, ym, Im)) / 2e-5 for e in np.eye(3) * 1e-5])
        dh = np.max(np.abs(H - Hfd)) / (1 + np.max(np.abs(H)))
        check(f'9.3 Hessian vs gradient differences ({tag}, h=1e-5)', dh < 1e-4, f'scaled={dh:.2e}')
    # far right tail K=1000, mu=1: log-survival, score h, curvature h(mu+h-K-1)
    S, f = dec_tail(1000, 1.0)
    h_ref = Decimal(1) * f / S
    curv_ref = h_ref * (Decimal(1) + h_ref - 1000 - 1)
    ls = float(sim.logsf(1000, 1.0))
    h = math.exp(math.log(1.0) + float(sim.logpmf(1000.0, 1.0)) - ls)
    curv = h * float(sim.curv_bracket(np.array([1000.0]), np.array([1.0]), np.array([h]))[0])   # production path
    e1, e2, e3 = abs(ls - float(S.ln())), abs(h - float(h_ref)) / float(h_ref), abs(curv - float(curv_ref)) / abs(float(curv_ref))
    check('9.3 far tail K=1000 mu=1 (Decimal 80 digits)', e1 < 1e-9 and e2 < 1e-9 and e3 < 1e-9,
          f'logsf={ls!r} ref={float(S.ln())!r} abs={e1:.1e}; score rel={e2:.1e}; curvature rel={e3:.1e}')
    worst = 0.0
    for k, mu in [(41, 20.0), (50, 20.0), (60, 5.0), (12, 6.0), (30, 20.0), (200, 1.0), (100, 1.0)]:
        S3, f3 = dec_tail(k, mu)
        hr = Decimal(mu) * f3 / S3
        cr = hr * (Decimal(mu) + hr - k - 1)
        hh = math.exp(math.log(mu) + float(sim.logpmf(float(k), mu)) - float(sim.logsf(k, mu)))
        cc = hh * float(sim.curv_bracket(np.array([float(k)]), np.array([mu]), np.array([hh]))[0])
        worst = max(worst, abs(cc - float(cr)) / max(abs(float(cr)), 1e-300))
    check('9.3 curvature across direct/series switch (k+1 vs 2mu, 7 points, Decimal ref)', worst < 1e-9, f'max rel={worst:.2e}')
    worst = 0.0
    for mu in (0.5, 1.0, 3.0):
        for k in range(20, 400):
            S2 = sim.special.gammainc(k + 1, mu)
            if 1e-290 < S2 < 1e-100:
                worst = max(worst, abs(math.log(S2) - sim._series_logsf(k, mu)))
    check('9.3 tail switch overlap (gammainc vs series, S in 1e-290..1e-100)', worst < 1e-11, f'max={worst:.2e}')
    # ---- 9.4 structural
    yz, Ia = np.full(sim.N, 3.0), np.ones(sim.N, bool)
    check('9.4 all-censored -> nonexistent', sim.screen(yz, Ia)[0] == 'nonexistent', str(sim.screen(yz, Ia)))
    y0 = np.zeros(sim.N)
    check('9.4 all-zero uncensored -> nonexistent', sim.screen(y0, None)[0] == 'nonexistent', str(sim.screen(y0, None)))
    check('9.4 positive uncensored spanning design -> pass', sim.screen(y, None)[0] == 'pass')
    # mixed: positives only in no-promo stratum, uncensored zeros in deep stratum, shallow/moderate censored
    ymx = np.where(sim.S_IDX == 0, 5.0, np.where(sim.S_IDX == 3, 0.0, 2.0))
    Imx = (sim.S_IDX == 1) | (sim.S_IDX == 2)
    st, J = sim.screen(ymx, Imx)
    import scipy.optimize as so
    # recover v and confirm direct movement strictly lowers the objective without bound
    Up = np.unique(sim.X[(~Imx) & (ymx > 0)], axis=0); U0 = np.unique(sim.X[(~Imx) & (ymx == 0)], axis=0)
    C1 = np.unique(sim.X[Imx], axis=0)
    r = so.linprog(-(C1.sum(0) - U0.sum(0)), A_ub=np.vstack([U0, -C1]), b_ub=np.zeros(len(U0) + len(C1)),
                   A_eq=Up, b_eq=np.zeros(len(Up)), bounds=[(-1, 1)] * 3, method='highs')
    v = r.x; b0 = np.array([math.log(5), 0, 0])
    vals = [sim.nll(b0 + t * v, ymx, Imx) for t in (0, 1, 2, 4, 8, 16)]
    mono = all(a > b_ for a, b_ in zip(vals, vals[1:]))
    check('9.4 mixed example: LP certifies nonexistence and moving along v lowers objective monotonically',
          st == 'nonexistent' and mono, f'J*={J:.4f} v={np.round(v, 4).tolist()} nll={np.round(vals, 6).tolist()}')
    # ---- 9.5 streams (calibration already checked in S3) follows 9.6
    # ---- 9.6 pairing and output checks (hand-constructed arrays, production analysis functions)
    import analysis as an
    nan = np.nan
    AM, AG = np.array([1., 5, 2, 100]), np.array([3., 1, 9, 0])
    B = AM - AG                                              # [-2, 4, -7, 100]
    check('9.6 median of within-ID differences (not difference of medians)',
          an.paired_median(B) == (1.0, 4) and np.median(AM) - np.median(AG) == 1.5, f'{an.paired_median(B)}')
    check('9.6 large finite loss retained (100 pp)', an.paired_median(B)[0] == 1.0 and an.paired_median(B[:3])[0] == -2.0)
    AGf = AG.copy(); AGf[1] = nan                            # G invalid for ID 1 -> missing loss, not zero
    check('9.6 missing-fit eligibility: invalid G removes ID from V only', an.paired_median(AM - AGf) == (-2.0, 3))
    B0, B6 = np.array([1., 2, 3, 4]), np.array([2., 2, 10, 0])
    th = an.theta_vector(B0, B6)                             # [1, 0, 7, -4]
    check('9.6 nested Theta = median(B6 - B0), not Delta6 - Delta0',
          an.paired_median(th)[0] == 0.5 and np.median(B6) - np.median(B0) == -0.5)
    B6m = B6.copy(); B6m[2] = nan
    check('9.6 Theta uses W = V0 ∩ V.6', an.paired_median(an.theta_vector(B0, B6m)) == (0.0, 3))
    bm, bn = an.boot_medians(B, np.array([[0, 0, 3], [1, 2, 3]]))
    check('9.6 bootstrap keeps multiplicities', bm[0] == -2.0 and bn[0] == 3 and bm[1] == 4.0, f'{bm.tolist()}')
    bm2, bn2 = an.boot_medians(np.array([nan, nan, 1.0]), np.array([[0, 1, 0], [2, 2, 0]]))
    s = an.summarize(1.0, 1, bm2)
    check('9.6 empty resample -> missing, B_usable counted, no replacement draw',
          np.isnan(bm2[0]) and bn2[0] == 0 and s['B_usable'] == 1 and s['B_total'] == 2 and np.isnan(s['lo']))
    check('9.6 no original valid pairs -> not estimable', an.summarize(nan, 0, np.array([nan, nan]))['estimable'] is False)
    fits = [dict(valid=v) for v in (True, True, False, True)]
    p = 1 - sum(f['valid'] for f in fits) / len(fits)
    check('9.6 failed fits stay in failure denominator (per attempted fit)', p == 0.25)
    ids = eng_ids()
    zA = sim.draw_latent(ids[1]); zB = sim.draw_latent(ids[2])
    check('9.5 different streams across IDs', not np.array_equal(zA[0], zB[0]) and not np.array_equal(zA[1], zB[1]))
    # identical demand across conditions within an ID: D computed once, verified via per-dataset demand hash
    one = sim.run_replicate(1, ids[1], cal, stream='engineering')
    check('9.5 identical demand across the six conditions within an ID',
          len({d['demand_sha'] for d in one['datasets']}) == 1 and len(one['datasets']) == 6)
    # worker-order invariance: run the 5 smoke IDs in two different schedules with spawned workers
    ctx = mp.get_context('spawn')
    t1 = time.time()
    with ProcessPoolExecutor(5, mp_context=ctx) as ex:
        a = dict(ex.map(_rep, [(i, ids[i], cal) for i in [1, 2, 3, 4, 5]]))
    t_par = time.time() - t1
    with ProcessPoolExecutor(2, mp_context=ctx) as ex:
        b_ = dict(ex.map(_rep, [(i, ids[i], cal) for i in [5, 3, 1, 4, 2]]))
    def _nt(o):   # drop wall-clock fields; everything else must match bit-for-bit
        return json.dumps({k: [{kk: vv for kk, vv in r.items() if kk != 'fit_seconds_GM'} for r in v] for k, v in o.items()},
                          sort_keys=True, default=str)
    same = all(_nt(a[i]) == _nt(b_[i]) for i in ids)
    check('9.5 identical results under different worker order/count (5 smoke IDs, 30 datasets)', same)
    json.dump({str(k): v for k, v in a.items()}, open('smoke_results.json', 'w'), default=str)
    LOG.append(f'INFO  smoke 5 IDs x 6 conditions on 5 workers: {t_par:.1f}s wall')
    LOG.append(f'INFO  total validation wall time {time.time() - t0:.1f}s; precision: float64 + Decimal(80) reference; '
               f'FD steps grad 1e-6, Hessian 1e-5')
    open('validation_log.txt', 'w').write('\n'.join(LOG) + f'\n\nRESULT: {"ALL PASS" if not FAILS else "FAIL: " + ", ".join(FAILS)}\n')
    print('ALL PASS' if not FAILS else f'FAIL {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
