# repair_log.md — JAMS existing-analysis correction (instructions dated 22 Nov 2026)

Workspace: `~/papers/jams-frn50k/correction_20261122/` (git; legacy originals are byte copies in `legacy/`, never edited).
Tag `p2-frozen` = corrected estimator + solver contract frozen before production fits.

## 1. Identities

- Raw data: FreshRetailNet-50K `train.parquet`, 106,436,287 bytes, sha256 `6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d` (not redistributed).
- Legacy code: `legacy/main`, `legacy/pilot` (byte copies of the live `main/`, `pilot/` at P0, commit `99d8088`; hashes in `input_manifest.json`).
- Original E2 results: the 16 shards `main/E2_misspec_cells_s{0..15}.csv` (35,100 rows). `main/E2_misspec_cells.tar.gz` is an earlier, different run with the same keys and is not the historical baseline.
- Environment: Python 3.14.9, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, 32-core Linux host.
- Corrected code (sha256 prefix): `mfit_corrected.py` adb926aa…, `altmodels_corrected.py` 3fe99613…, `tobit_corrected_shim.py` 7de38de1…, `p4_hourly_f.py` 3c583d27…, `solver_contract.json` db713789…, `numeric_validation.log` cccc9eb6….

- Inventory at P0 (outside the audit ZIP): the 16 E2 shards store only two fitted coefficients (promo, depth) and the convergence flag per attempt. Full fitted vectors, full generating `beta_vec`, the X/K/C datasets and optimizer logs were **not** saved, so the E2 inputs were rebuilt with the isolated legacy code (`frozen/*.npz`, which now store X, the full `beta_vec`, K and C for every dataset). Empirical: `elast_cells.csv` stores the promo/depth coefficients, `m_converged`, `m_grad` and `m_nit`, but no full vectors. `pilot/fit_state.npz` stores the pilot inputs but not X. Every corrected fit now stores the full coefficient vector (`full_fits/`).

## 2. Defects found and exact changes

| Legacy location | Defect | Correction (where) |
|---|---|---|
| `pilot/tobit.py:16–19` `_logpbar` (and the same floor in `07_pending_3` NB2/ZIP, `08_pending_final` mixtures) | Upper tail clipped at 1e-300; λ clipped to [1e-8, 1e6]. K=200, λ=1 gives −690.776 instead of −869.530. | Strict, stable upper tails with no floor or clip: Poisson series/regularized gamma in log domain, NB2 series in log domain (`mfit_corrected.pois_logsf`, `nb_logsf`). |
| `pilot/tobit.py:36–66` `fit` | `converged = res.success and finite ll`. All 254 empirical cells had a gradient above 1e-6 (max 7.08). | Average-NLL objective, L-BFGS-B, stationarity checked on the original and the standardized scale ≤ 1e-6 (`mfit_corrected.fit`, `_minimize`); recession LP screen for structural nonexistence; rank check for identification. Non-valid fits are reported, never used. |
| `pilot/tobit.py:98–106` `_scores_at` | Wrong sign of the censored-row curvature (K=0, λ=1: legacy +0.3387, exact −0.3386968873). | Analytic curvature `−h(K+1−μ−h)` with a series for the far tail (`mfit_corrected.m_hess`, `p4_hourly_f.obs_terms`). |
| `pilot/tobit.py:109–141` `fit_fe` | Alternating loop with no joint convergence check; 6 of 20 original cells stopped at f_nit=1. | Joint damped Newton on slopes and store-day offsets (Schur complement), Armijo backtracking, joint stationarity (slopes and per-group offset scores ≤ 1e-6) plus objective stability; columns absorbed by offsets are reported as absorbed; separated groups (all censored → +∞; all zero → −∞) are counted, not fitted (`p4_hourly_f.fit_f`). |
| `main/03_main.py:32`, `pilot/02_pilot.py:35` | `k_h[~C]` on an integer array indexes rows −1/−2 instead of selecting uncensored days. | `k_h[C == 0]`; a cell with no uncensored row → hourly result unavailable (occurs in 0 of 254 cells) (`p4_hourly_f.apportion_fixed`, sandbox `03_main.py`, `02_pilot.py`). |
| `main/07_pending_3_altmodels.py` NB2/ZIP | Same tail floor; maxiter-100-era output (NB2 16/254, ZIP 8/254 converged). | Shared corrected kernel (`mfit_corrected.fit('NB2'/'MIX')`); NB2 log-pmf rewritten to avoid cancellation at large θ (P2 defect fix, commit `0ccac6a`). |
| `main/08_pending_final.py` `fit_cell` (ZIP2, mixture Poisson / mixture NB2) | Same floors; `_fit` maxiter 100. | `altmodels_corrected.py` (log-domain mixtures, same model definitions and bounds, frozen solver). |
| `main/10_fix_posthoc.py:48–52` | Forced sign flip of `n2_ll`/`z1_ll` (legacy stored −ll). | Corrected fits store ll correctly; flip removed; NaN-safe wild CI (`wild_ci` drops non-finite cells with their city weights). |
| Consumers (`06_bootstrap_ci`, `07_pending_4`, `09_dose_wildci`, `12_discount_scenario`, `06_risk_coverage_sim`) | Assumed every cell has a coefficient. | Cells without a valid corrected fit are excluded and the eligible denominator is printed or recorded. |
| Operational (not scientific) | `03_main` checkpoint has no header; several 07/08 scripts require `pending_report.md` to exist. | Header restored from the live file; report files copied into the sandbox. |

Unchanged by design: data generation (A0–A5 generator, κ, seeds), cells, repeats, covariates, the |true coefficient| ≥ 0.05 rule, aggregation definitions, the 4,000-draw city bootstrap, the two Holm families, seed 42 / 5,000-draw combined-lift bootstrap. No |coefficient| cutoff was applied.

## 3. Solver contract (frozen before results, `solver_contract.json`)

Average NLL; scipy L-BFGS-B, maxiter 5000, maxfun 50000, maxls 50, internal gtol 1e-9, ftol 0. Valid = solver success, finite objective/coefficients/gradient, and sup-norm projected gradient ≤ 1e-6 on both the original and the column-standardized scale. Statuses: valid / structural_nonexistence (recession LP J* > 1e-8) / unresolved_screen / unidentified / nonconvergence / nonfinite. NB2 log θ ≤ log 1e4, MIX logit π ∈ [−8, 8]. M starts: legacy `init_poisson_irls` b0, b0 + 0.1 on promo, b0 + 0.1 on depth; NB2/MIX also start from the corrected M optimum. Lowest objective among valid starts; every start is logged.

## 4. Numerical checks

`numeric_validation.log` 25/25 PASS (instruction tests a–d, legacy-floor demonstration, FD gradients/Hessians, recovery on known truth, multistart agreement, NB2 large-θ log-pmf). `out/altmodels_selfcheck.log` ALL PASS (ZIP2/mixture FD gradients ≤ 3.3e-10, π=1 reductions exact, K=200 μ=1 = −869.530, recovery formulas vs brute force ≤ 1.8e-15). P4 allocation checks pass in all 254 cells (non-negative, additions only in flagged hours, uncensored days unchanged, daily totals ≤ 4.4e-16).

## 5. Replay status

- **E2: documented reconstruction.** Replaying the unmodified legacy generator gives keys, true coefficients, and anchors identical for 35,100/35,100. Generated data are identical for 28,985 datasets and differ for 6,115 (the generator calls legacy M for the true coefficients, and that M is sensitive to floating-point behaviour). Comparisons are therefore reported on the frozen reconstructed inputs (`frozen/*.npz`, `replay_manifest.json`), with the original-shard result kept as a separate column.
- **Empirical: near-exact replay.** Legacy M rerun vs stored `elast_cells.csv`: 191/254 identical to the stored 5 decimals; 63 differ by at most 3.1e-4 (same solver sensitivity). Hourly replay vs stored `main_hourly.parquet`: max 4.8e-5. F replay vs `main_topF.csv`: recF identical to 4 decimals except 0,5 / 0,16 / 0,28 (≤ 0.33). "Before" values in `before_after.csv` are the published files.

## 6. Accounting of every attempted fit (`fit_diagnostics.csv`, 106,082 rows)

- E2 M (35,100): valid 29,870, structural_nonexistence 4,526, nonconvergence 693, nonfinite 10, unresolved 1. NB2 valid 29,210, MIX valid 29,216. Of the 4,526 nonexistence datasets, 4,525 are A4 (ρ=0.5: 2,080; ρ=1.0: 2,442; ρ=0: 3), where the median censoring share is 99.9% and 100%.
- Empirical (254): M valid 249 (nonconvergence 3, unidentified 1, nonexistence 1); NB2 252; MIX 238.
- F (20): 20 jointly valid.

### Added diagnostics and pilot (review follow-up)
- `fit_diagnostics.csv` E2 rows also carry realized censoring by promotion status, uncensored counts (overall, promotion, positive), and the number of rows where the legacy safeguards would activate (λ clip low/high, 1e-300 tail floor), evaluated at the corrected coefficients (`p6b_diagnostics.py`; legacy full coefficient vectors were never saved). The tail floor activates in 0 rows; the λ upper clip is reached in 7 (M), 191 (NB2) and 168 (MIX) datasets. Empirical rows carry objective, screen, linear-predictor range, eigenvalues and condition number.
- Pilot (`02_pilot.py`, cell 0×4) rerun in the sandbox with the corrected fitter and row mask (`sandbox/logs_02_pilot.log`): A-route values are unchanged (naive −0.09542/hr, recovered −0.14998/hr, gap +0.05455, 57.2%; gates 12/14 and 9/14 weeks). M is valid after 181 iterations (legacy 167, non-stationary). F is jointly valid (10 iterations, 26,010 store-day groups, β_promo −0.47646). The legacy weekly B-route summary in `02_pilot.py` cannot combine the corrected F with the infinite offsets of the all-censored store-days, so it prints NaN. The manuscript's pilot B row therefore uses the corrected cell-level F from P4 (naive −2.8160, recovered −4.3786 on rows with finite recovery).
- `manuscript_changes.csv`: every changed manuscript sentence, table and figure (git diff d0acc5d → corrected commit), with the numbers removed and added. Those numbers are also appended to `before_after.csv` as `section = manuscript_location`.

## 7. Unresolved issues / limits

- Empirical legacy replay differs from stored values in 63 cells at ≤ 3.1e-4. This does not affect the corrected results, which are refitted from raw data.
- 5 empirical cells have no valid corrected M: (0,8) and (4,28) nonconvergence, (10,4) nonconvergence, (14,31) unidentified, (17,21) structural nonexistence. They are excluded from cell-level aggregates, and the denominator 249/254 is stated.
- F recovery is unavailable for the rows of all-censored store-days (offset +∞); these are counted per cell in `p4_F_summary.csv`.

## 8. Reproduce

```
cd correction_20261122; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FRN_PARQUET=<train.parquet>
PY=../.venv/bin/python
$PY validate_corrected.py                      # numeric_validation.log
$PY altmodels_corrected.py                     # alt-model self-check
$PY replay_e2.py ... ; $PY compare_replay.py   # P1 (see replay_manifest.json)
$PY p3_fit.py --all --workers 28               # E2 refits (≈5 h)
$PY p3_empirical.py --all --workers 4          # empirical refits
$PY p4_hourly_f.py --all --workers 8; $PY p4_summary.py
$PY p5_e2.py                                   # original E2 analysis chain on 5 input sets
./run_p5_stage1.sh; ./run_p5_stage2.sh         # sandbox consumers via corrected tobit shim
$PY p5_before_after.py; $PY p6_package.py      # before_after.csv, fit_diagnostics.csv
```

Git history of `correction_20261122/`: P0 `99d8088` → P1 `24006fa` → P2 `092d944`/`0ccac6a` (tag p2-frozen) → P3 `b426d62` → P4 `00db14c` → P5 `c3ae000`.
