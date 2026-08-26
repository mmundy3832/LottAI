#!/bin/bash
# Wrapper for scheduled prediction pipeline — sources env and logs output.

ENV_FILE="/mnt/beastmode/lottai/.env"
set -a; source "$ENV_FILE"; set +a

LOG_DIR="/mnt/beastmode/lottai/autoresearch/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_prediction.log"

cd /mnt/beastmode/lottai/autoresearch

source /mnt/beastmode/lottai/.venv/bin/activate && \
    python3 orchestrate.py 2>&1 | tee "$LOG_FILE"
