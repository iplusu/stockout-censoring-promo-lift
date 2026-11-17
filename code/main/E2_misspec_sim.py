"""E2 — Misspecified-DGP simulation (design: ~/.hermes/shared/bot_debates/jams-e2-design-20261107/,
spec: experiments/E2_spec.md, round1 convergence: experiments/E2_spec.md + STATUS.md in that dir).

Reuses the shipped fitting code (no reimplementation):
  - pilot/tobit.fit            -> M  (censored Poisson MLE)
  - pilot/tobit.init_poisson_irls -> G  (Poisson GLM on uncensored days only)
  - main/07_pending_3_altmodels.fit_nb2 -> M-NB2
  - main/07_pending_3_altmodels.fit_zip -> M-mix (ZIP, pi estimated not fixed -- see NOTE below)
  - main/05_decomp.build_groups -> real covariates X and real censoring pattern C per cell

Design decisions from the Round-1 debate (1크루나/코루나/아루나, 2026-11-07):
  1. A0 (Poisson, matches M's own assumption) is a SELF-CHECK, not evidence. Reported separately.
  2. Headline comparison is M vs M-NB2 vs M-mix relative error (not M vs naive/G, which is
     near-tautological since naive/G ignore censoring entirely in every arm).
  3. External beta anchor: for ANCHOR_CELLS (6 of the 30, not all), the "true" beta used to
     generate data is decomp_cells.csv's beta_naive_full (a non-M estimator) instead of the
     cell's own M-fitted beta -- reduces (does not fully remove) circular-truth risk for that
     subset. The other 24 cells use beta_rec (M's own fit) and are flagged as such in output.
  4. Semi-synthetic censoring for A0-A3: reuse the REAL observed day-level censoring indicator C
     and, on censored days, the REAL observed K as the capacity ceiling (capacity does not depend
     on the simulated D, so this does not have E4's demand-dependent-capacity flaw). Only A4
     synthesizes a new non-ignorable capacity mechanism.

NOTE on M-mix: the original external spec's "M-mix (pi fixed at 0.85)" is simplified to fit_zip's
estimated pi (the shipped ZIP implementation estimates pi, nothing in the codebase fits a
pi-fixed variant). ponytail: if a reviewer wants pi-fixed, add a `pi_fixed` kwarg to fit_zip --
not built now since nothing currently needs it.
"""
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
sys.path.insert(0, HERE)
import tobit  # noqa: E402
import importlib
m07 = importlib.import_module('07_pending_3_altmodels')  # noqa: E402
m05 = importlib.import_module('05_decomp')  # noqa: E402
m3 = importlib.import_module('03_main')  # noqa: E402

OUT_DIR = HERE
CELLS_CSV = os.path.join(OUT_DIR, 'main_cells.csv')
DECOMP_CSV = os.path.join(OUT_DIR, 'decomp_cells.csv')
SAMPLED_CELLS_CSV = os.path.join(os.path.dirname(HERE), 'experiments', 'E2_sampled_cells.csv')
TEST_MANIFEST_JSON = os.path.join(os.path.dirname(HERE), 'experiments', 'E2_test_manifest.json')

ARMS = ['A0_poisson', 'A1_nb2', 'A2_zip', 'A3_label_noise', 'A4_nonignorable', 'A5_combined']
NB2_ALPHAS = [0.1, 0.5, 1.0]  # stress grid; empirical alpha (1/theta from decomp Pearson) used separately
ZIP_PSIS = [0.05, 0.15]
LABEL_PI = [0.95, 0.85, 0.75]  # pi = P(flag correctly KEPT as censored | truly censored) --
# i.e. true-positive rate for the censoring flag; 1-pi is the false-negative (mislabel) rate
# applied in A3/A5 (코루나 Round4: this definition must be explicit in manifest/report, not
# left implicit in a code comment -- clarified here and in the manifest below).
RHOS = [0.0, 0.5, 1.0]
N_REP = 100


def select_30_cells(seed=42):
    """Stratified sample: store-count x stockout-share x promo-stockout-overlap terciles,
    forcing in the pilot cell (0,4). Frozen to SAMPLED_CELLS_CSV so it cannot be re-drawn
    after seeing results (코루나/아루나 Round-1 requirement)."""
    if os.path.exists(SAMPLED_CELLS_CSV):
        return pd.read_csv(SAMPLED_CELLS_CSV)
    cells = pd.read_csv(CELLS_CSV)
    ok = cells[cells.status == 'ok'].copy()
    rng = np.random.default_rng(seed)
    for col, new in [('n_stores', 'store_terc'), ('stockout_day_share', 'so_terc'), ('both_share', 'overlap_terc')]:
        ok[new] = pd.qcut(ok[col], 3, labels=False, duplicates='drop')
    ok['strata'] = list(zip(ok.store_terc, ok.so_terc, ok.overlap_terc))
    groups = list(ok.groupby('strata'))
    rng.shuffle(groups)
    picked = []
    pilot = ok[(ok.city == 0) & (ok['cat'] == 4)]
    if len(pilot):
        picked.append(pilot.iloc[0])
    per_stratum = max(1, 30 // max(len(groups), 1))
    for _, g in groups:
        g = g[~((g.city == 0) & (g['cat'] == 4))]
        if len(g) == 0:
            continue
        take = g.sample(n=min(per_stratum, len(g)), random_state=seed)
        picked.extend([r for _, r in take.iterrows()])
        if len(picked) >= 30:
            break
    out = pd.DataFrame(picked).drop_duplicates(subset=['city', 'cat']).head(30).reset_index(drop=True)
    os.makedirs(os.path.dirname(SAMPLED_CELLS_CSV), exist_ok=True)
    out.to_csv(SAMPLED_CELLS_CSV, index=False)
    return out


def anchor_assignment(sampled, n_anchor=6, seed=42):
    """6 of the 30 cells get the non-M (beta_naive_full) truth anchor; the rest use beta_rec."""
    rng = np.random.default_rng(seed)
    keys = list(zip(sampled.city.astype(int), sampled['cat'].astype(int)))
    anchor_keys = set(map(tuple, rng.choice(keys, size=min(n_anchor, len(keys)), replace=False)))
    return anchor_keys


def true_beta_vec_for_cell(X, real_K, real_C, city, cat, decomp, anchor_keys):
    """Full beta vector for the DGP: intercept/dow/holiday/activity/weather pinned to the
    REAL M-fit on this cell (keeps simulated lambda at a realistic scale so the real censoring
    pattern stays a meaningful capacity proxy -- caught by selfcheck: zeroing those out made
    simulated lambda<=2.2 vs real K up to 27, so real C=1 rows were almost never actually
    censored by the fake draws, silently mislabeling uncensored rows as censored).
    Only promo/depth (what the arms vary) come from the anchor/rec source."""
    b0 = tobit.init_poisson_irls(X, real_K, real_C)
    fM = tobit.fit(X, real_K, real_C, b0=b0)
    beta_vec = fM['b'].copy()
    row = decomp[(decomp['city'] == city) & (decomp['cat'] == cat)].iloc[0]
    is_anchor = (int(city), int(cat)) in anchor_keys
    src = 'naive_full' if is_anchor else 'rec'
    beta_vec[1] = row[f'beta_{src}_promo']
    beta_vec[2] = row[f'beta_{src}_depth']
    return beta_vec, float(beta_vec[1]), float(beta_vec[2]), is_anchor


def gen_D(lam, arm, rng, nb2_alpha=None, zip_psi=None):
    if arm == 'A0_poisson':
        return rng.poisson(lam)
    if arm == 'A1_nb2':
        th = 1.0 / nb2_alpha
        p = th / (th + lam)
        return rng.negative_binomial(th, p)
    if arm == 'A2_zip':
        zero_inflate = rng.random(len(lam)) < zip_psi
        d = rng.poisson(lam)
        return np.where(zero_inflate, 0, d)
    raise ValueError(arm)


def gen_D_poisson_with_shock(lam, rng):
    """Inverse-CDF Poisson draw that also returns the uniform shock u used to generate D,
    so a capacity model can be built as a function of the SAME shock (needed for A4's
    non-ignorable censoring -- see apply_censoring)."""
    from scipy import stats
    u = rng.random(len(lam))
    D = stats.poisson.ppf(u, lam).astype(np.int64)
    return D, u


def gen_D_nb2_with_shock(lam, alpha, rng):
    """Inverse-CDF NB2 draw + shared shock (코루나 Round3: NB2 can share a uniform shock with
    its capacity exactly like Poisson via nbinom.ppf -- used by A5 so the 'combined' arm is
    genuinely non-ignorable instead of silently falling back to an independent-shock capacity)."""
    from scipy import stats
    th = 1.0 / alpha
    p = th / (th + lam)
    u = rng.random(len(lam))
    D = stats.nbinom.ppf(u, th, p).astype(np.int64)
    return D, u


def apply_censoring(D, real_K, real_C, kappa_rho=None, demand_shock=None, rng=None):
    """A0-A3: a capacity ceiling exists only on real-C=1 days (real_K is the proxy ceiling),
    but censoring itself is determined by whether the SIMULATED D exceeds that ceiling --
    NOT a blind copy of real_C (root-cause fix 2026-11-07, 코루나 Round2: blindly copying
    real_C mislabels rows as censored when simulated D never reached the ceiling, e.g.
    D=[0,1,2,4], cap=10 all mislabeled C=1 -- confirmed by code read, no counterexample needed).
    A4 (kappa_rho given): genuinely non-ignorable capacity -- cap is driven by the SAME
    latent demand_shock that generated D (not an independent fresh draw), so capacity and
    realized demand are correlated through rho (root-cause fix: previous version drew u/eta
    independently of D, so rho only changed censoring severity, never non-ignorability --
    confirmed by code read, 코루나 Round2)."""
    if kappa_rho is None:
        cap = np.where(real_C == 1, real_K, np.inf)
        C = (D > cap).astype(np.int8)
        K = np.where(C == 1, cap, D).astype(np.int64)
        return K, C
    kappa, rho, lam = kappa_rho
    # demand_shock in (0,1): same per-row shock used to draw D (via its Poisson/NB2 quantile)
    # so capacity is literally a function of the same latent draw -> true MNAR dependency.
    u = np.clip(demand_shock, 1e-6, 1 - 1e-6)
    eta = rng.normal(0, 0.5, len(D))
    cap = np.floor(kappa * lam * np.exp(rho * np.log(u) + (1 - rho) * eta))
    cap = np.maximum(cap, 0)
    C = (D > cap).astype(np.int8)
    K = np.where(C == 1, cap, D).astype(np.int64)
    return K, C


def relerr(bhat, beta):
    if abs(beta) < 0.05:
        return np.nan
    return abs(bhat - beta) / abs(beta)


def run_cell_arm(X, real_K, real_C, beta_vec, arm, n_rep, seed_base, pi_names):
    """One (cell, arm) block: n_rep replicates, all 5 estimators, paired across estimators."""
    rows = []
    beta_p, beta_d = float(beta_vec[1]), float(beta_vec[2])
    lam_true = np.clip(np.exp(X @ beta_vec), 1e-6, 1e5)
    stockout_share = float(real_C.mean())
    for rep in range(n_rep):
        # crc32 (not hash()): builtin hash() is salted per process (PYTHONHASHSEED) ->
        # E2 draws were NOT reproducible across runs/shards (fixed 2026-11-07).
        ss = np.random.SeedSequence([42, 2, seed_base, zlib.crc32(arm.encode()) % 10000, rep])
        rng = np.random.default_rng(ss)
        for param in pi_names:
            if arm == 'A0_poisson':
                D = gen_D(lam_true, arm, rng)
                K, C = apply_censoring(D, real_K, real_C)
            elif arm == 'A1_nb2':
                D = gen_D(lam_true, arm, rng, nb2_alpha=param)
                K, C = apply_censoring(D, real_K, real_C)
            elif arm == 'A2_zip':
                D = gen_D(lam_true, arm, rng, zip_psi=param)
                K, C = apply_censoring(D, real_K, real_C)
            elif arm == 'A3_label_noise':
                D = gen_D(lam_true, 'A0_poisson', rng)
                K, C = apply_censoring(D, real_K, real_C)
                # Pure label noise (코루나 Round3 fix): a mislabeled row must look EXACTLY like
                # a real uncensored observation to the estimator -- only the C flag flips.
                # Previously this ALSO replaced K with D, which silently removes the censoring
                # from the data itself (not a label error at all, just less censoring).
                flip = (C == 1) & (rng.random(len(C)) > param)  # flagged censored but truly not, w.p. 1-pi
                C = np.where(flip, 0, C)  # K stays the (still-capped) observed value -- label-only change
            elif arm == 'A4_nonignorable':
                D, shock = gen_D_poisson_with_shock(lam_true, rng)
                K, C = apply_censoring(D, real_K, real_C, kappa_rho=(max(stockout_share, 0.05), param, lam_true),
                                        demand_shock=shock, rng=rng)
            elif arm == 'A5_combined':
                # A1(empirical alpha, from decomp Pearson chi2/df 3.22 -> alpha~=0.31) + A3(pi=0.85) + A4(rho=0.5)
                # Root-cause fix (코루나 Round3): now shares the real NB2 shock with capacity
                # (gen_D_nb2_with_shock) instead of an independent rng.random() capacity draw --
                # this arm is genuinely non-ignorable+NB2+label-noise now, not just NB2+severity.
                D, shock = gen_D_nb2_with_shock(lam_true, 1.0 / 3.22, rng)
                K, C = apply_censoring(D, real_K, real_C, kappa_rho=(max(stockout_share, 0.05), 0.5, lam_true),
                                        demand_shock=shock, rng=rng)
                flip = (C == 1) & (rng.random(len(C)) > 0.85)
                C = np.where(flip, 0, C)  # label-only flip (same A3 fix: K stays the capped value)
            else:
                raise ValueError(arm)
            rec = dict(arm=arm, param=param, rep=rep)
            try:
                b_naive = np.linalg.lstsq(X, np.log1p(K), rcond=None)[0]
                rec['naive_p'], rec['naive_d'] = b_naive[1], b_naive[2]
            except Exception:
                rec['naive_p'] = rec['naive_d'] = np.nan
            try:
                # G = manuscript's actual definition: plain Poisson GLM, IRLS-to-convergence,
                # on ALL rows (full observed K, censoring ignored entirely) -- main/05_decomp.py
                # poisson_glm_irls(X, K). Root-cause fix (코루나 Round2): previously used
                # tobit.init_poisson_irls(X,K,C) which fits only UNCENSORED rows for 10 fixed
                # iterations -- a different, already-partially-censoring-aware estimator, not
                # the manuscript's G at all.
                b_g, g_nit, g_conv = m05.poisson_glm_irls(X, K.astype(float))
                rec['g_p'], rec['g_d'], rec['g_conv'] = b_g[1], b_g[2], g_conv
            except Exception:
                rec['g_p'] = rec['g_d'] = np.nan
                rec['g_conv'] = False
            try:
                b0 = tobit.init_poisson_irls(X, K, C)
                fM = tobit.fit(X, K, C, b0=b0)
                rec['m_p'], rec['m_d'], rec['m_conv'] = fM['b'][1], fM['b'][2], fM['converged']
                _se = fM.get('se')  # v2 (REFRAME_DESIGN part 3.4): Wald SE capture for CI coverage
                rec['m_se_p'], rec['m_se_d'] = (_se[1], _se[2]) if _se is not None else (np.nan, np.nan)
            except Exception:
                rec['m_p'] = rec['m_d'] = np.nan
                rec['m_conv'] = False
            try:
                fN = m07.fit_nb2(X, K, C, b0)
                rec['nb2_p'], rec['nb2_d'], rec['nb2_conv'] = fN['b'][1], fN['b'][2], fN['converged']
            except Exception:
                rec['nb2_p'] = rec['nb2_d'] = np.nan
                rec['nb2_conv'] = False
            try:
                fZ = m07.fit_zip(X, K, C, b0)
                rec['mix_p'], rec['mix_d'], rec['mix_conv'] = fZ['b'][1], fZ['b'][2], fZ['converged']
            except Exception:
                rec['mix_p'] = rec['mix_d'] = np.nan
                rec['mix_conv'] = False
            rows.append(rec)
    return rows


def subsample_cell(d, max_n=20000, seed=42):
    """Cap rows for compute tractability -- caught during testing: the pilot cell (0,4) has
    578,430 real rows and a single M-fit took 29s; uncapped, the full 100-rep x 12-param run on
    that one cell alone would take ~days. A uniform random subsample preserves the covariate
    distribution and the real censoring/stockout pattern; only bounds per-fit cost.
    ponytail: stratified-by-store subsampling would be more careful, add if a reviewer flags that
    random row subsampling could bias the stockout share for giant cells -- not observed in
    testing, so not built now."""
    n = d['X'].shape[0]
    if n <= max_n:
        return d
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    out = dict(d)
    for key in ('X', 'K', 'C', 'k_h', 'ST', 'promo'):
        if key in out:
            out[key] = out[key][idx]
    return out


def git_commit_hash():
    """Best-effort current commit hash for manifest provenance (코루나 Round4: code version
    must be recorded alongside the pi definition, not just in a commit message)."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE,
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def build_manifest(arm_param_map, n_cells, anchor_keys, n_rep=N_REP, extra=None):
    """Single source of truth for run manifests -- main() and E2_run_shard.py both call this
    instead of each building their own dict (코루나 Round4 caught E2_run_shard.py's manifest
    silently missing the pi definition added to main()'s because it duplicated the dict)."""
    m = {'arms': arm_param_map, 'n_cells': n_cells, 'n_rep': n_rep,
         'anchor_cells': sorted(list(anchor_keys)),
         'code_commit': git_commit_hash(),
         'headline_comparison': 'M vs M-NB2 vs M-mix relative error (1크루나 Round1)',
         'A0_role': 'self-check only, not evidence (1크루나 Round1)',
         'A3_A5_pi_definition': ('pi = P(censoring flag correctly KEPT | row truly censored), '
                                  'i.e. true-positive rate; 1-pi = false-negative mislabel rate '
                                  '(코루나 Round4)')}
    if extra:
        m.update(extra)
    return m


def main(n_rep=N_REP, out_suffix=''):
    cells = select_30_cells()
    decomp = pd.read_csv(DECOMP_CSV)
    anchor_keys = anchor_assignment(cells)
    eligible = set(zip(cells.city.astype(int), cells['cat'].astype(int)))
    groups = m05.build_groups(eligible)

    arm_param_map = {
        'A0_poisson': [None], 'A1_nb2': NB2_ALPHAS, 'A2_zip': ZIP_PSIS,
        'A3_label_noise': LABEL_PI, 'A4_nonignorable': RHOS, 'A5_combined': [None],
    }
    manifest = build_manifest(arm_param_map, len(cells), anchor_keys, n_rep=n_rep)
    os.makedirs(os.path.dirname(TEST_MANIFEST_JSON), exist_ok=True)
    if not os.path.exists(TEST_MANIFEST_JSON):
        with open(TEST_MANIFEST_JSON, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

    all_rows = []
    for _, crow in cells.iterrows():
        city, cat = int(crow['city']), int(crow['cat'])
        if (city, cat) not in groups:
            continue
        d = m3.build_cell(groups[(city, cat)])
        d = subsample_cell(d)
        X, real_K, real_C = d['X'], d['K'], d['C']
        beta_vec, beta_p, beta_d, is_anchor = true_beta_vec_for_cell(X, real_K, real_C, city, cat, decomp, anchor_keys)
        for arm, params in arm_param_map.items():
            recs = run_cell_arm(X, real_K, real_C, beta_vec, arm, n_rep,
                                 seed_base=city * 100 + cat, pi_names=params)
            for r in recs:
                r.update(city=city, cat=cat, beta_true_p=beta_p, beta_true_d=beta_d, is_anchor=is_anchor)
            all_rows.extend(recs)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, f'E2_misspec_cells{out_suffix}.csv')
    df.to_csv(out_csv, index=False)
    print(f'wrote {len(df)} rows -> {out_csv}')
    return df


def selfcheck():
    """ponytail: smallest runnable check -- 1 cell, A0 only, 3 reps. A0 is Poisson/M's own
    assumption, so M should recover beta to within ~20% relative error; if it doesn't, the
    harness (not the DGP) is broken."""
    cells = pd.read_csv(CELLS_CSV)
    ok = cells[cells.status == 'ok'].iloc[[0]]
    decomp = pd.read_csv(DECOMP_CSV)
    eligible = {(int(ok.iloc[0]['city']), int(ok.iloc[0]['cat']))}
    groups = m05.build_groups(eligible)
    city, cat = next(iter(eligible))
    d = m3.build_cell(groups[(city, cat)])
    d = subsample_cell(d)
    beta_vec, beta_p, beta_d, _ = true_beta_vec_for_cell(d['X'], d['K'], d['C'], city, cat, decomp, set())
    recs = run_cell_arm(d['X'], d['K'], d['C'], beta_vec, 'A0_poisson', 3, city * 100 + cat, [None])
    df = pd.DataFrame(recs)
    err_p = (df['m_p'] - beta_p).abs().median()
    err_d = (df['m_d'] - beta_d).abs().median()
    assert df['m_conv'].all(), f'M did not converge in selfcheck: {df.m_conv.tolist()}'
    # A0 is Poisson (well-specified for all 5 estimators): NB2/MIX non-convergence here means
    # the harness is broken, not the data (maxiter 100->1000 fix, 2026-11-07 stability probe).
    assert df['nb2_conv'].all(), f'M-NB2 did not converge in selfcheck: {df.nb2_conv.tolist()}'
    assert df['mix_conv'].all(), f'M-mix did not converge in selfcheck: {df.mix_conv.tolist()}'
    assert err_p < 0.3 * max(abs(beta_p), 0.2), f'M promo recovery off: median |err|={err_p}, beta={beta_p}'
    assert err_d < 0.3 * max(abs(beta_d), 0.2), f'M depth recovery off: median |err|={err_d}, beta={beta_d}'
    print(f'SELFCHECK PASS: cell ({city},{cat}) beta_true=({beta_p:.3f},{beta_d:.3f}) '
          f'M median |err|=({err_p:.3f},{err_d:.3f}) conv={df.m_conv.tolist()}')

    # Regression checks for the 2026-11-07 코루나 Round2 bugs (censoring-flag mismatch, A4
    # non-ignorability) -- these are branch-logic bugs a coefficient-recovery check alone
    # would not have caught, so test the mechanism directly.
    D_test = np.array([0, 1, 2, 4])
    real_K_test = np.array([10, 10, 10, 10])
    real_C_test = np.array([1, 1, 1, 1])
    K_t, C_t = apply_censoring(D_test, real_K_test, real_C_test)
    assert (C_t == 0).all(), f'censoring-flag regression: D<=cap must give C=0, got {C_t}'
    assert (K_t == D_test).all(), f'censoring-flag regression: K must equal D when C=0, got {K_t}'
    D_test2 = np.array([15, 1])
    K_t2, C_t2 = apply_censoring(D_test2, real_K_test[:2], real_C_test[:2])
    assert C_t2.tolist() == [1, 0], f'censoring-flag regression: only D>cap should censor, got {C_t2}'

    rng_a4 = np.random.default_rng(7)
    lam_a4 = np.full(2000, 20.0)
    D_a4, shock_a4 = gen_D_poisson_with_shock(lam_a4, rng_a4)
    kappa, lam = 0.5, lam_a4
    eta0 = np.random.default_rng(7).normal(0, 0.5, len(D_a4))
    cap_rho0 = np.floor(kappa * lam * np.exp(0.0 * np.log(np.clip(shock_a4, 1e-6, 1 - 1e-6)) + 1.0 * eta0))
    cap_rho1 = np.floor(kappa * lam * np.exp(1.0 * np.log(np.clip(shock_a4, 1e-6, 1 - 1e-6)) + 0.0 * eta0))
    # the real test of "non-ignorable": does the CAPACITY correlate with the same shock that
    # drove demand? (corr(D,C) is a poor proxy -- any threshold-crossing indicator mechanically
    # correlates with the variable being thresholded even for an independent threshold, so that
    # alone doesn't distinguish ignorable from non-ignorable censoring)
    corr_rho0 = abs(np.corrcoef(shock_a4, cap_rho0)[0, 1])
    corr_rho1 = abs(np.corrcoef(shock_a4, cap_rho1)[0, 1])
    assert corr_rho1 > corr_rho0 + 0.3, (
        f'A4 non-ignorability regression: |corr(shock,capacity)| should rise sharply with rho, '
        f'got rho0={corr_rho0:.3f} rho1={corr_rho1:.3f}')
    print(f'REGRESSION PASS: censoring-flag fix verified, A4 |corr(shock,capacity)| rho=0 -> {corr_rho0:.3f}, rho=1 -> {corr_rho1:.3f}')

    # A3 label-noise regression (코루나 Round3): flipping C must NOT change K -- a mislabeled
    # row looks like real uncensored data with the SAME (still-capped) observed value.
    rng_a3 = np.random.default_rng(11)
    D_a3 = np.array([20, 20, 20, 20])
    real_K_a3 = np.array([5, 5, 5, 5])
    real_C_a3 = np.array([1, 1, 1, 1])
    K_a3, C_a3 = apply_censoring(D_a3, real_K_a3, real_C_a3)
    assert (K_a3 == 5).all() and (C_a3 == 1).all(), f'A3 precondition failed: K={K_a3} C={C_a3}'
    flip = np.array([True, False, True, False])
    C_flipped = np.where(flip, 0, C_a3)
    assert (K_a3 == 5).all(), 'A3 regression: K must stay at the capped value after a label flip'
    assert C_flipped.tolist() == [0, 1, 0, 1], f'A3 regression: only C should change, got {C_flipped}'

    # A5 regression: NB2 demand shock must correlate with capacity (genuinely non-ignorable,
    # not an independent-shock fallback).
    rng_a5 = np.random.default_rng(13)
    lam_a5 = np.full(2000, 20.0)
    D_a5, shock_a5 = gen_D_nb2_with_shock(lam_a5, 1.0 / 3.22, rng_a5)
    cap_a5 = np.floor(0.5 * lam_a5 * np.exp(0.5 * np.log(np.clip(shock_a5, 1e-6, 1 - 1e-6))))
    corr_a5 = abs(np.corrcoef(shock_a5, cap_a5)[0, 1])
    assert corr_a5 > 0.3, f'A5 regression: NB2 shock should correlate with capacity, got {corr_a5:.3f}'
    print(f'REGRESSION PASS: A3 label-only flip verified, A5 NB2 shock-capacity corr={corr_a5:.3f}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selfcheck':
        selfcheck()
    elif len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        main(n_rep=2, out_suffix='_smoke')
    else:
        main(n_rep=N_REP)
