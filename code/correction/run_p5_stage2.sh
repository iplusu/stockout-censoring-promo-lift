#!/bin/bash
# P5 stage 2 (after stage 1): alt models, product FE, P-A sensitivity, dose wild CI, posthoc, risk-coverage sim.
set -e
cd /home/ubuntu/papers/jams-frn50k/correction_20261122
grep -q STAGE1_DONE out/p5_s1.log || { echo "stage1 not done"; exit 1; }
cd sandbox/main
PY=/home/ubuntu/papers/jams-frn50k/.venv/bin/python
export FRN_PARQUET=/home/ubuntu/dataset/sigmetrics27-artifact-led/raw/frn50k/train.parquet
run() { echo "=== $* start $(date +%T)"; "$@"; echo "=== $1 $2 done $(date +%T)"; }
pids=()
( run $PY 08_pending_final.py --pA-only --nproc 20 > ../logs_08.log 2>&1 ) & pids+=($!)
( run $PY 07_pending_3_altmodels.py > ../logs_07_3.log 2>&1 ) & pids+=($!)
for s in 0 1 2 3; do ( $PY 07_pending_5_fe.py --nshards 4 --shard $s > ../logs_07_5_s$s.log 2>&1 ) & pids+=($!); done
( run $PY 09_dose_wildci.py > ../logs_09.log 2>&1 ) & pids+=($!)
( run $PY 06_risk_coverage_sim.py > ../logs_06_risk.log 2>&1 ) & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || { echo "a stage-2 job failed"; exit 1; }
run $PY 07_pending_5_fe.py --merge --nshards 4 > ../logs_07_5_merge.log 2>&1
run $PY 10_fix_posthoc.py > ../logs_10.log 2>&1
echo STAGE2_DONE
