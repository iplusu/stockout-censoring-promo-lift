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

## Corrected pipeline (source of every number in the current manuscript)

A code audit found implementation defects in the original recovery-model code. All manuscript results now come from
the corrected pipeline in `code/correction/`. The original scripts in `code/pilot` and `code/main` are kept unchanged
as the historical record; their outputs in `data/main` are the pre-correction values.

Defects corrected (details: `data/correction/repair_log.md`):
1. The censored tail log Pr(D > K) was floored at 1e-300 and lambda was clipped. It is now a strict, stable log-domain
   tail with no floor or clip (`mfit_corrected.py`).
2. The convergence flag was solver success only. Validity now requires stationarity <= 1e-6 on both the original and
   the standardized scale, a recession-LP screen for structural non-existence of the MLE, and an identification check.
3. The censored-row curvature in the store-day fixed-effect (F) model had the wrong sign. F is now a joint Newton fit
   with a joint convergence check.
4. The hourly profile used `k_h[~C]` on an integer array, which selects rows -1/-2 instead of uncensored days. It now
   uses `k_h[C == 0]`.
5. The same tail floor in the NB2, ZIP and mixture alternatives is replaced (`altmodels_corrected.py`).

Run order (`OMP_NUM_THREADS=1`; set `FRN_PARQUET` to the raw `train.parquet`):
1. `python validate_corrected.py` writes `numeric_validation.log` (25/25 checks). `python altmodels_corrected.py`
   runs the alternative-model self-check.
2. `python replay_e2.py` then `python compare_replay.py`: freeze the E2 inputs with the unmodified legacy generator.
   This is a documented reconstruction: keys and true coefficients are identical for all 35,100 datasets, and the
   generated data are identical for 28,985 of them.
3. `python p3_fit.py --all --workers 28` (E2, all 35,100 datasets) and `python p3_empirical.py --all` (254 cells).
4. `python p4_hourly_f.py --all` (hourly mask fix for 254 cells, corrected F for the original 20 cells).
5. `python p5_e2.py` reruns the original E2 analysis chain (seed 42, 4,000-draw city bootstrap, both Holm families)
   on the corrected fits.
6. `run_p5_stage1.sh` and `run_p5_stage2.sh` rerun the original empirical scripts in a sandbox copy whose
   `pilot/tobit.py` is `tobit_corrected_shim.py`. The other edits to the sandbox copies are in
   `code/correction/sandbox_patches/` (they exclude cells without a valid fit and state the denominator).
7. `python p5_before_after.py`, `python p6_package.py` and `python p6c_manuscript_map.py` write `before_after.csv`,
   `fit_diagnostics.csv` and `manuscript_changes.csv`.

Paths: the scripts use the absolute workspace paths where they were run (`.../correction_20261122`, the raw-data path). Set `JAMS_CORR_DIR` to `code/correction` and `FRN_PARQUET` to the raw file. The sandbox is `code/main` and `code/pilot` with the patches in `code/correction/sandbox_patches/` applied and `pilot/tobit.py` replaced by `tobit_corrected_shim.py`.

Outputs in `data/correction/`:
- `before_after.csv`: old value, corrected value, absolute change, denominator and comparison basis for every
  affected statistic.
- `fit_diagnostics.csv.gz`: one row per attempted dataset and model, 106,082 rows. Failed fits are kept.
- `manuscript_changes.csv`: every changed manuscript sentence, table and figure.
- `sandbox_main/`: corrected versions of the `data/main` tables.
- `p5_e2/`: E2 tables and tests.

Controlled simulation (manuscript §3.4, Figure 5, Online Resource 1): `code/controlled_sim/` and
`data/controlled_sim/` (see the README there; MIT licence).

## Original pipeline (historical, pre-correction values)

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

## Reproducibility notes

- E2 inputs: the original shards could not be replayed bit for bit, because the generator calls the legacy M fit and
  that fit is sensitive to floating-point behaviour. The corrected analysis therefore uses a frozen, hashed
  reconstruction (`replay_manifest.json`). The original shards stay in `data/main` as the historical result.
- The corrected fits are deterministic under the frozen solver contract (`solver_contract.json`).
- Under arm A4 with rho > 0, the censored-Poisson MLE does not exist for most datasets. These fits are reported as
  failures, not dropped (`fit_diagnostics.csv.gz`).

## Requirements

Python 3, `pandas`, `numpy`, `scipy`, `pyarrow`. No GPU required.
