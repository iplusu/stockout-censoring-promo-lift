# Controlled simulation: observed-sales (G) vs censored Poisson (M) promotion-lift recovery

A synthetic experiment that follows a fixed protocol (`protocol.txt`, v1.0): 1,000 replicate IDs × 6 conditions (η ∈ {0, 0.6} × q ∈ {0.20, 0.60, 0.95}), n = 2,400 per dataset. Outcomes are in `COMPLETION_NOTE.md`.

## Reproduce
```
python calibrate.py                                     # -> calibration.csv (24 cells), compare to frozen file
python check_calibration_mc.py                          # independent Monte Carlo check of calibration
python validate.py                                      # -> validation_log.txt (protocol §9.1-9.6)
python run.py --stream evaluation --workers 30          # -> out/evaluation/rep_0001..1000.json (resume = missing only)
python analysis.py --stream evaluation --outdir results # -> results/*.csv (tables, 2,000 bootstrap iterations)
python figure5.py                                       # -> results/figure5.pdf / .png
```
Environment: `env_versions.txt` (Python 3.14.9, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0). Seeds and conventions: `freeze_manifest.json`, `run_manifest.json`. All random draws come from `SeedSequence(20261122)`, so any fit can be regenerated without a new draw (`sim.run_replicate(rid, seeds('evaluation')[rid], calibration)`).

## Files
- `config.json`, `calibration.csv`: frozen design and capacity calibration.
- `sim.py`: data generator, likelihoods (stable strict Poisson tail with no probability floor), finite-maximum LP screen, fitting, losses, and information diagnostics.
- `results/`: `datasets.csv` (6,000), `fits.csv` (13,000), `starts.csv` (25,000), `losses.csv` (13,000), `information.csv` (6,000), `paired.csv` (6,000), `dependence.csv` (3,000), `bootstrap.csv`, `summaries.csv`, `failure_summary.csv`, `oracle_summary.csv`, `figure5.pdf/png`.
- `xx_mapping.csv`: maps each manuscript quantity to its statistic, file, column, unit, and denominator.
- `out/evaluation/`: per-ID raw records. These are not in git; they are in the compressed archive `controlled_sim_results.tar.gz`.

License: MIT (`LICENSE`).
