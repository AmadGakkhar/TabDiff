#!/bin/bash
# Approach 2 for paysim temporal splits 0,1,2 — run SEQUENTIALLY.
# Each split trains its 4 doses (d1..d4) in parallel, one per GPU, then generates
# 100k fraud rows per dose (reject; low-yield doses capped at 4h). Splits run one
# after another because each split needs all 4 GPUs.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff

MASTER_LOG=logs/paysim_app2_splits012.log
mkdir -p logs
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $1" | tee -a "$MASTER_LOG"; }

say "=== paysim Approach 2: splits 0,1,2 (sequential) START ==="
for s in 0 1 2; do
  say ">>> SPLIT $s: prepare"
  python -u prepare_paysim_app2.py "$s" >> "logs/prepare_ps${s}.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && { say "!!! SPLIT $s prepare FAILED rc=$rc — aborting"; exit 1; }

  say ">>> SPLIT $s: train+generate (4 GPUs)"
  python -u run_paysim_app2.py "$s" >> "logs/run_ps${s}.log" 2>&1
  rc=$?; say "<<< SPLIT $s done (rc=$rc)"
done
say "=== ALL SPLITS DONE ==="
