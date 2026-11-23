# JAMS controlled simulation — completion note (protocol v1.0 §11)

Run date: 2026-11-22. Freeze `freeze-v1` (commit a4c6402). Results commit: see `git log`.

1. **Validation and calibration passed.** `validation_log.txt` is 30/30 PASS (§9.1–9.6). Calibration passed 24/24 cells: max |p−q| = 2.73e-11 (tolerance 1e-6), quadrature error ≤ 1.7e-12, and an independent Monte Carlo check gave max |z| = 1.77.
2. **All 1,000 replicate IDs were attempted in all six conditions:** 6,000 datasets, 13,000 fits (6,000 G, 6,000 M, 1,000 O), and 25,000 scheduled starts, all attempted (0 skipped). The run was not resumed. Wall time 355 s on 30 workers.
3. **Primary estimate.** Θ₀.₆₀ = **2.560 pp**, 95% Monte Carlo interval [2.450, 2.701], MCSE 0.068. |W₀.₆₀| = 1,000 and B_usable = 2,000/2,000.
   An independent recomputation directly from `losses.csv`, without `analysis.py`, matches all three Θ_q values.
4. **Model failures by condition: 0 in every condition.** G 0/1,000, M 0/1,000, and Oracle 0/1,000. There were no structural nonexistence, unresolved screen, nonconvergence, or nonfinite cases. All 6,000 M information diagnostics are `ok`; median condition number is 233.6–236.6. No dataset was all-censored or rank-deficient. The median minimum uncensored stratum count is 313 (q=.20), 152 (q=.60), and 16 (q=.95).
5. **Direction.** The primary contrast is **positive and its interval excludes zero.** Under this design, introducing capacity–demand dependence (η=0.6) makes the M − G lift-error difference less favorable to M.
   - This does not show that M is worse than G. Δ is negative, with intervals excluding zero, in all six conditions; that is, M has the smaller lift error in each condition:
     - q=.20: −0.50 → −0.16 pp (η=0 → 0.6)
     - q=.60: −4.45 → −2.09 pp
     - q=.95: −17.88 → −8.00 pp
   - Secondary: Θ₀.₂₀ = 0.225 [0.171, 0.261]; Θ₀.₉₅ = 10.05 [9.79, 10.26].
   - Adverse or mixed coefficient result: at q=.20 the paired depth relative error is slightly larger for M. ΔE_d = +0.0018 [−0.0000, 0.0039] at η=0 and +0.0030 [0.0011, 0.0053] at η=0.6. ΔE_P spans zero at q=.20. At q=.60 and .95 both coefficient differences are negative.
   - The agreement categories are descriptive only (e.g. at q=.20, η=.6: 42.4% improve neither criterion). See `summaries.csv`.
   - Oracle: median lift loss 1.03 pp [p10 0.41, p90 2.02].
6. **Interpretation limits (§11).** This is a comparison under six specified synthetic conditions. It does not establish the availability process in FreshRetailNet, prove that equal censoring implies equal identification, identify causal promotion effects, or establish optimal promotion policies. It is new evidence, kept distinct from the earlier E2 simulation.
7. **Deviations and implementation notes** (also in `run_manifest.json`):
   - No change to the design or estimands.
   - Two defects were found by validation and fixed before freeze and before any evaluation fit:
     - The tail function dropped series values for scalar inputs, returning −inf.
     - The far-tail Hessian weight had a subtractive cancellation (relative error 3.5e-7). It was replaced with an algebraically identical series form.
   - Performance only: with 30 parallel workers, per-ID time was 10.6 s versus 0.55 s serial in S6 (probably BLAS thread oversubscription). Results are unaffected, since worker-order invariance was validated.
   - Figure 5 legibility was checked only with `pdftotext -bbox` (186 text items inside the page). The automated visual check timed out, and no human visual review has been done yet.
   - Manuscript integration (S11) is not done. The target draft (user decision 1A) has not been received, so the target locations in `xx_mapping.csv` are pending.
