"""Rerun the full E2 analysis pipeline (reproducibility entry point, v2 / post-4dae0dd).
Reads E2_misspec_cells_s*.csv, writes /tmp/e2_headline_tbl2.csv / e2_f1_tests2.csv /
e2_f2_tests2.csv. Deterministic (seed=42 throughout, city-cluster bootstrap n_boot=4000).
"""
import subprocess, sys, glob, re, pandas as pd
# Strict: original 16 shards only -- excludes *_v2.csv (SE-capture rerun) and smoke-test files,
# which an unfiltered 'E2_misspec_cells_s*.csv' glob also matches (코루나/아루나, jams-final-review-20261113).
shard_re = re.compile(r'E2_misspec_cells_s\d+\.csv$')
shards = sorted(f for f in glob.glob('E2_misspec_cells_s*.csv') if shard_re.search(f))
assert shards, "run from main/ after E2 shards are present"
dfs = [pd.read_csv(f) for f in shards]
df = pd.concat(dfs, ignore_index=True)
assert len(df) == 35100, f"expected 35100 rows, got {len(df)}"
df.to_parquet('/tmp/e2_all2.parquet')
for script in ['e2_analysis2.py', 'e2_boot2.py', 'e2_extra2.py']:
    subprocess.run([sys.executable, script], check=True)
