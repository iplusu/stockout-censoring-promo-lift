"""P7: city-block bootstrap (5,000 resamples, seed 42) for the Sec. 3.8 statistics and the
Bonferroni-8 lower bounds of Table bonf8, on the ORIGINAL and the CORRECTED cell files.

Why: the sandbox copy of 06_bootstrap_ci.py hard-codes ~/papers/jams-frn50k/main/ paths, so
sandbox/logs_06_bootstrap_ci.py.log printed the ORIGINAL numbers. This script reuses the sandbox
(NaN-safe) 06 functions verbatim and only points them at the corrected files.
Gate: on the original files it must reproduce the published 06 log values."""
import importlib.util, os, random

ROOT = os.path.expanduser("~/papers/jams-frn50k")
spec = importlib.util.spec_from_file_location("m06", f"{ROOT}/correction_20261122/sandbox/main/06_bootstrap_ci.py")
m06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m06)
A8 = 0.05 / 8


def boot(fn, by_city, cities, n_boot=5000, seed=42):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        rows = []
        for c in [rng.choice(cities) for _ in cities]:
            rows.extend(by_city[c])
        vals.append(fn(rows))
    vals.sort()
    point = fn(sum((by_city[c] for c in cities), []))
    return point, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1], vals[int(A8 * n_boot)]


def run(main_path, decomp_path, label):
    main = [r for r in m06.load(main_path) if r["status"] == "ok"]
    dec = m06.load(decomp_path)
    cities = sorted(set(r["city"] for r in main))
    bm = {c: [r for r in main if r["city"] == c] for c in cities}
    bd = {c: [r for r in dec if r["city"] == c] for c in cities}
    specs = [
        ("P1a_frac_ge20", bm, lambda R: m06.frac_ge(R, "bias_ratio")),
        ("P1a_median", bm, lambda R: m06.median_abs(R, "bias_ratio")),
        ("pure_frac_depth", bd, lambda R: m06.frac_ge(R, "bias_pct_pure_censoring_depth")),
        ("pure_frac_promo", bd, lambda R: m06.frac_ge(R, "bias_pct_pure_censoring_promo")),
        ("pure_median_depth", bd, lambda R: m06.median_abs(R, "bias_pct_pure_censoring_depth")),
        ("pure_median_promo", bd, lambda R: m06.median_abs(R, "bias_pct_pure_censoring_promo")),
        ("flip_pure_depth", bd, lambda R: m06.sign_flip_frac(R, "beta_glm_depth", "beta_rec_depth")),
        ("flip_pure_promo", bd, lambda R: m06.sign_flip_frac(R, "beta_glm_promo", "beta_rec_promo")),
        ("flip_orig_depth", bd, lambda R: m06.sign_flip_frac(R, "beta_naive_reduced_depth", "beta_rec_depth")),
        ("flip_orig_promo", bd, lambda R: m06.sign_flip_frac(R, "beta_naive_reduced_promo", "beta_rec_promo")),
    ]
    print(f"### {label}: {len(main)} main rows, {len(dec)} decomp rows, {len(cities)} cities")
    out = {}
    for name, bc, fn in specs:
        out[name] = boot(fn, bc, cities)
        p, lo, hi, b8 = out[name]
        print(f"{name}: point={p:.4f} CI=[{lo:.4f}, {hi:.4f}] bonf8_lo={b8:.4f}")
    return out


if __name__ == "__main__":
    o = run(f"{ROOT}/main/main_cells.csv", f"{ROOT}/main/decomp_cells.csv", "ORIGINAL (gate)")
    # published 06 log / Table bonf8 values
    assert round(o["P1a_frac_ge20"][1], 4) == 0.7597 and round(o["flip_pure_depth"][2], 4) == 0.1502
    assert round(o["P1a_median"][3], 3) == 0.403 and round(o["flip_pure_promo"][3], 3) == 0.094
    print("GATE PASS: original reproduces published 06 CIs and Table bonf8 lower bounds")
    run(f"{ROOT}/correction_20261122/sandbox/main/main_cells.csv",
        f"{ROOT}/correction_20261122/sandbox/main/decomp_cells.csv", "CORRECTED")
