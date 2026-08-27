#!/bin/bash
# Sequential launcher for the M3 grokking sweep (wd 0.1, 0.3, 1.0).
# Measured: ~8.3h/run incl. eval+checkpoint overhead -> ~25h total, under the
# 48h threshold in the task instructions, so sequential (not concurrent).
set -e
cd /mnt/beastmode/lottai/phase3
source /mnt/beastmode/lottai/.venv/bin/activate

for run_id in m3_wd0.1 m3_wd0.3 m3_wd1.0; do
  mkdir -p "runs/${run_id}"
  echo "=== starting ${run_id} at $(date) ===" >> "runs/${run_id}/stdout.log"
  python3 train_grok.py "configs/${run_id}.json" >> "runs/${run_id}/stdout.log" 2>&1
  echo "=== finished ${run_id} at $(date) ===" >> "runs/${run_id}/stdout.log"
done
