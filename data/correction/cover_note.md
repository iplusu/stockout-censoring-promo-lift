# Cover note — existing-analysis correction

**1. Did the main empirical coefficient gaps, review classifications, or combined-lift conclusions change?**
No material change. The corrected M is valid in 249 of 254 cells. On those cells, the corrected and legacy coefficients differ by a median of 2.1e-6 (promo) and 1.5e-5 (depth). The small differences that do appear come from excluding the 5 cells without a valid fit; on the same cells, legacy and corrected values coincide.
- Censoring gap (depth/promo): 25.3%/52.4% → 25.9%/52.5%.
- Unmatched gap: 50.2%/84.1% → 50.0%/83.4%.
- Cells reclassified at the 20% threshold: depth 90 (66/24) → 86 (63/23); promo 64 (39/25), unchanged.
- Combined lift G−M, in percentage points: p10 7.00 → 6.98, p25 4.19 → 3.87, p50 3.09 → 2.96, p75 1.90 → 1.66, p90 0.47 → 0.16. The city-block intervals are nearly unchanged, and the same intervals exclude 0 (p10–p50) or include it (p75, p90).
- The §3.9 alternative models now converge (NB2 16 → 252, ZIP 8 → 238), and their gaps move by up to 0.10. The qualitative conclusion holds.

**2. Did the existing A4 performance conclusions change?**
Yes. Under A4 with ρ=0.5 and ρ=1.0, the generator censors almost every day (median censoring share 99.9% and 100%). The censored-Poisson MLE does not exist in 2,080/2,700 and 2,442/2,700 datasets. Legacy M flagged 4,352 of these 4,526 datasets as "converged" and returned arbitrary stopping points (|b_promo| median 1.11, 90th percentile 49.9). After correct failure accounting:
- Valid-fit cells drop from 23–27 to 6–13.
- Holm-significant comparisons among the 12 retained F2 tests fall from 10 to 5.
- ρ=1.0 M−naive (promo) changes sign: +3.54 → −0.14.

On identical repeat IDs, legacy and corrected estimates agree. The change therefore comes from validity accounting, not from different estimates. The published claim that M's error exceeds naive/G under non-ignorable censoring is not supported for ρ>0 as stated. It must be restated as follows: the M estimator does not exist for most strongly non-ignorable datasets under this generator, and on the minority with a finite MLE the comparison is mixed. Family F1 (M vs NB2/MIX, 10 tests) is unchanged at 5 significant.

**3. Which hourly and F results must be replaced or left unreported?**
- Hourly: replace. The mask fix changes the peak hour of recovered demand in 160/254 cells and the peak of the bias in 178/254. Daily totals and recA are unchanged. No cell is unavailable.
- F: replace. All 20 cells are now jointly valid. Only promo, depth, and activity are identified within store-days. Store-days that are entirely censored (111–9,653 per cell) carry no recoverable demand and are reported as such. recF changes substantially in several cells (e.g., 0,28 −528.6 → −14.1; 0,5 217.8 → 10.5).
- Nothing is left unresolved.

**4. Does the running controlled experiment use the validated fitter?**
Yes. `controlled_sim/` imports only its own `sim.py`, which has strict tails, analytic score and curvature, an LP screen, and passes validation 30/30. It never imports the legacy `pilot/tobit.py`. All 13,000 stored fits are valid under that contract, and none requires correction.
