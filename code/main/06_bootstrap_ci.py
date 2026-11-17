"""City-block bootstrap CIs for Study 2 headline statistics.

Addresses 1크루나's reviewer-debate finding (2026-10-30): all headline point estimates in
04_study2_results.md §3.8 had zero uncertainty quantification, and the 254/397 cells are not
independent (they share city-level day-of-week/holiday/weather covariates), so a naive per-cell
CI would understate variance. Fix: resample CITIES (18, with replacement), not cells, pool all
cells belonging to the resampled cities (a city can appear 0 or multiple times), recompute each
statistic on the pooled resample, repeat 5000x, report the 2.5/97.5 percentile interval.

Data: main/main_cells.csv (397 rows, filtered to status=='ok' -> 254), main/decomp_cells.csv
(254 rows, already the 'ok' subset). Deterministic, seed 42.

Ceiling (ponytail): percentile bootstrap, not BCa — fine for these monotonic fraction/median
statistics with 18 clusters; add BCa if a referee specifically asks for bias-corrected intervals.
No multiple-comparison adjustment across the 10 statistics tested here — each CI is a marginal
95% interval, not a joint one (noted explicitly in the paper, 07_limitations.md).
"""
import csv
import random
import sys


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def frac_ge(rows, col, thresh=0.20):
    vals = [abs(float(r[col])) for r in rows]
    return sum(1 for v in vals if v >= thresh) / len(vals) if vals else float("nan")


def median_abs(rows, col):
    vals = sorted(abs(float(r[col])) for r in rows)
    n = len(vals)
    if n == 0:
        return float("nan")
    mid = n // 2
    return vals[mid] if n % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2


def sign_flip_frac(rows, col_a, col_b):
    flips = sum(1 for r in rows if sgn(float(r[col_a])) != sgn(float(r[col_b])))
    return flips / len(rows) if rows else float("nan")


def bootstrap_ci(stat_fn, by_city, cities, n_boot=5000, seed=42):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample_cities = [rng.choice(cities) for _ in cities]
        rows = []
        for c in sample_cities:
            rows.extend(by_city[c])
        vals.append(stat_fn(rows))
    vals.sort()
    lo = vals[int(0.025 * n_boot)]
    hi = vals[int(0.975 * n_boot) - 1]
    point = stat_fn(sum((by_city[c] for c in cities), []))
    return point, lo, hi


def run(main_path, decomp_path, n_boot=5000, seed=42):
    main_cells = [r for r in load(main_path) if r["status"] == "ok"]
    decomp_cells = load(decomp_path)
    cities = sorted(set(r["city"] for r in main_cells))
    by_city_main = {c: [r for r in main_cells if r["city"] == c] for c in cities}
    by_city_decomp = {c: [r for r in decomp_cells if r["city"] == c] for c in cities}

    specs = [
        ("P1a_frac_ge20", by_city_main, lambda rows: frac_ge(rows, "bias_ratio", 0.20)),
        ("P1a_median_bias_ratio", by_city_main, lambda rows: median_abs(rows, "bias_ratio")),
        ("pure_censoring_frac_ge20_depth", by_city_decomp,
         lambda rows: frac_ge(rows, "bias_pct_pure_censoring_depth", 0.20)),
        ("pure_censoring_frac_ge20_promo", by_city_decomp,
         lambda rows: frac_ge(rows, "bias_pct_pure_censoring_promo", 0.20)),
        ("pure_censoring_median_depth", by_city_decomp,
         lambda rows: median_abs(rows, "bias_pct_pure_censoring_depth")),
        ("pure_censoring_median_promo", by_city_decomp,
         lambda rows: median_abs(rows, "bias_pct_pure_censoring_promo")),
        ("sign_flip_pure_depth", by_city_decomp,
         lambda rows: sign_flip_frac(rows, "beta_glm_depth", "beta_rec_depth")),
        ("sign_flip_pure_promo", by_city_decomp,
         lambda rows: sign_flip_frac(rows, "beta_glm_promo", "beta_rec_promo")),
        ("sign_flip_orig_depth", by_city_decomp,
         lambda rows: sign_flip_frac(rows, "beta_naive_reduced_depth", "beta_rec_depth")),
        ("sign_flip_orig_promo", by_city_decomp,
         lambda rows: sign_flip_frac(rows, "beta_naive_reduced_promo", "beta_rec_promo")),
    ]
    out = {}
    for name, by_city, fn in specs:
        out[name] = bootstrap_ci(fn, by_city, cities, n_boot=n_boot, seed=seed)
    return out, len(cities), len(main_cells), len(decomp_cells)


def _selftest():
    """Smallest runnable check: CI must bracket the point estimate, and re-running with the
    same seed must be exactly reproducible (deterministic, no hidden global RNG state)."""
    out1, n_city, n_main, n_decomp = run(
        "/home/ubuntu/papers/jams-frn50k/main/main_cells.csv",
        "/home/ubuntu/papers/jams-frn50k/main/decomp_cells.csv",
        n_boot=500, seed=1,
    )
    for name, (point, lo, hi) in out1.items():
        assert lo <= point <= hi, f"{name}: point {point} not inside CI [{lo},{hi}]"
    out2, *_ = run(
        "/home/ubuntu/papers/jams-frn50k/main/main_cells.csv",
        "/home/ubuntu/papers/jams-frn50k/main/decomp_cells.csv",
        n_boot=500, seed=1,
    )
    assert out1 == out2, "not deterministic under a fixed seed"
    assert n_city == 18
    print("selftest OK:", n_city, "cities,", n_main, "main cells,", n_decomp, "decomp cells")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    out, n_city, n_main, n_decomp = run(
        "/home/ubuntu/papers/jams-frn50k/main/main_cells.csv",
        "/home/ubuntu/papers/jams-frn50k/main/decomp_cells.csv",
    )
    print(f"# City-block bootstrap (n_boot=5000, seed=42, {n_city} cities, "
          f"{n_main} main cells, {n_decomp} decomp cells)\n")
    for name, (point, lo, hi) in out.items():
        print(f"{name}: point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
