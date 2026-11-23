#!/bin/bash
# P5 stage 1: rerun M-consuming empirical pipeline in sandbox with the corrected tobit shim. Stops on first error.
set -e
cd /home/ubuntu/papers/jams-frn50k/correction_20261122/sandbox/main
PY=/home/ubuntu/papers/jams-frn50k/.venv/bin/python
export FRN_PARQUET=/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet
export OUT_DIR=.
for s in 03_main.py 04_elasticity.py 05_decomp.py 06_bootstrap_ci.py 07_pending_1_denom.py 07_pending_2_order.py 07_pending_4_jointci.py 12_discount_scenario.py; do
  echo "=== $s start $(date +%T)"; $PY $s > ../logs_$s.log 2>&1; echo "=== $s done $(date +%T)"
done
echo STAGE1_DONE
