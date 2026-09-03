#!/bin/bash
# Waits for the original m3_wd1.0 run (PID 1959356, launched by
# run_m3_sequential.sh, finishing its original 1M-step run) to exit, then
# resumes it from checkpoint_latest.pt to 10M total steps.
set -e
cd /mnt/beastmode/lottai/phase3
source /mnt/beastmode/lottai/.venv/bin/activate

ORIG_PID=1959356
echo "=== waiting for original wd1.0 PID ${ORIG_PID} to exit at $(date) ===" >> runs/m3_wd1.0/stdout.log
while kill -0 "$ORIG_PID" 2>/dev/null; do
  sleep 30
done
echo "=== original wd1.0 PID ${ORIG_PID} exited at $(date); resuming to 10M ===" >> runs/m3_wd1.0/stdout.log

python3 train_grok.py configs/m3_wd1.0.json --resume runs/m3_wd1.0/checkpoint_latest.pt --max-steps 10000000 >> runs/m3_wd1.0/stdout.log 2>&1
echo "=== finished wd1.0 extend at $(date) ===" >> runs/m3_wd1.0/stdout.log
