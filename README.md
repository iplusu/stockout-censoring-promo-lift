# Reproducible Artifact — "Out of Stock, Out of Sight: Stockout Censoring and Model-Relative Gaps in Promotion-Lift Estimates"

Anonymized for double-blind review. No author names, affiliations, or
acknowledgments appear anywhere in this package (verified by grep over the
whole tree before packaging).

## Data source (not redistributed here)

This paper analyzes **FreshRetailNet-50K** (arXiv:2505.16319v1), a public
dataset: 898 stores x 863 SKUs, 90 days, ~50,000 store-product series, daily
rows with 24-hour vectors (`hours_sale` / `hours_stock_status`) and a
`discount` price ratio (1.0 = no discount). Download the dataset from its
public release (arXiv / the dataset's own distribution channel) and point
`code/pilot/01_select_cell.py` and `code/main/03_main.py` /
`code/main/09_dose_wildci.py` at the local `train.parquet` (see the `HERE`
and dataset-path constants near the top of each script). The raw dataset is
not bundled here, consistent with its own distribution terms; only our
processing code and the frozen intermediate outputs we computed from it are
included.

Note: the paper's text also references a separately released, structurally
different dataset, FreshRetailNet-LT (1,057 stores / 576 SKUs / ~20,000
series / 2+ year span) -- see Section 2.3 of the manuscript for why this
paper uses the 50K/v1 panel instead. LT is not used by any script here.

## Pipeline (run in order)

1. `code/pilot/01_select_cell.py`, `02_pilot.py` (uses `tobit.py`) -- pilot
   cell selection and model-fitting criteria, fixed before the main
   analysis (unregistered). Produces `cell_selection.csv`.
2. `code/main/03_main.py` -- main 254 city x category panel audit (naive
   vs. recovery-model estimators). Produces `main_cells.csv`.
3. `code/main/04_elasticity.py` -- produces `elast_cells.csv`.
4. `code/main/05_decomp.py` -- produces `decomp_cells.csv`.
5. `code/main/06_bootstrap_ci.py`, `06_risk_coverage_sim.py` -- bootstrap
   CIs and the risk-coverage simulation validation.
6. `code/main/07_pending_1_denom.py` .. `07_pending_5_fe.py` -- robustness
   checks (common-denominator decomposition, ordering, alternative
   families, joint CIs, fixed-effect sensitivity).
7. `code/main/08_pending_final.py` -- **only `run_pA()` is this paper's
   reproduction path** (Section 3.9: real-data recovery-model sensitivity
   check). `run_pB()` and `run_pC()` inside the same file compute an
   unrelated exploratory analysis (Study 1, Yelp-based activity matching)
   that is **not part of this paper** and requires an external Yelp
   dataset not included here; `main()` calls all three unconditionally, so
   running the file as-is will fail at the P-B stage unless you either
   supply that external data or call `run_pA()` directly instead of
   `main()`.
8. `code/main/09_dose_wildci.py`, `10_fix_posthoc.py` -- wild-cluster
   bootstrap CIs and the final convergence-restricted figures. Produces
   **`fix_posthoc_pA.csv`**, the canonical Section 3.9 output table.
9. `code/main/11_e2_run_all.py` -- **deterministic entry point for the E2
   misspecified-DGP simulation** (Section 3.10; seed 42, city-cluster
   bootstrap n_boot=4000). Reads the 16 frozen shards in `data/main/`
   (`E2_misspec_cells_s0.csv` .. `s15.csv`, strictly excluding any `_v2`
   variant) and calls `e2_analysis2.py`, `e2_boot2.py`, `e2_extra2.py` in
   sequence, reproducing the Section 3.10 headline tables and tests.
   `E2_misspec_sim.py` / `E2_run_shard.py` are the original shard-generation
   scripts (included for provenance; the shards in `data/main/` are
   already the frozen output, so these do not need to be re-run for exact
   reproduction).
10. `code/main/12_discount_scenario.py` -- combined predicted promotion
   effect L_m(d) for matched G and M at five discount depths, and the paired
   18-city block bootstrap (seed 42, 5,000 resamples; reported difference is
   G minus M in percentage points). Needs the raw parquet (`FRN_PARQUET`) and
   `data/main/decomp_cells.csv`; writes `cell_max_depth.csv`,
   `discount_scenario_L.csv`, `discount_scenario_jointboot.json` (frozen
   copies included, rerun is bit-identical).

## What's deliberately excluded

- Raw FreshRetailNet-50K data (public, cited, not redistributed).
- `E1_*` scripts/files: an earlier benchmark design never cited in the
  published paper (superseded by E2).
- `*_v2.csv` shard variants, selfcheck files, run logs, `.bak` files,
  `__pycache__`, and `pending_final_report.md` (explicitly superseded by
  the convergence-filter-corrected run; kept only as a historical note in
  the manuscript's own appendix, not part of the reproduction path).
- The Study 1 (Yelp) directory and its activity-matching code (`run_pB`
  above): no part of the submitted paper's claims depends on it.

## Known non-reproducibility (disclosed in the manuscript)

Per the manuscript's Appendix ("Reproducibility"): the E2 simulation's
Naive/G columns reproduce exactly; the correction models (M, MIX, NB2) are
not bit-reproducible run-to-run at the individual-fit level (max ~0.08
absolute difference across all 60 `tab:e2descriptive` entries on a fresh
re-run), because their iterative MLE fits are solver-sensitive. This does
not affect the formally tested claim (M's error exceeding Naive/G for
rho>0 in arm A4), which is robust across both realizations. The
74.1%/89.3%/34.8%/40.8% fit-level error-rate statistics are a separate,
not-yet-scripted computation on an `e2_err2.parquet` intermediate and are
not reproduced by `11_e2_run_all.py` alone.

## Requirements

Python 3, `pandas`, `numpy`, `scipy`, `pyarrow`. No GPU required.
