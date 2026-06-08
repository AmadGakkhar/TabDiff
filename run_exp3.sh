#!/usr/bin/env bash
# Orchestrate exp_3: train one TabDiff model on the decoded car-insurance-fraud
# dataset (5500 steps), then generate a fully-synthetic version matching the
# original class distribution + size, re-encoded to the original 68-col layout.
set -u

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff

NAME=car_fraud_exp3
LOGDIR=logs/exp_3
OUTDIR=synthetic_outputs/exp_3
mkdir -p "$LOGDIR" "$OUTDIR"
RUNLOG="$LOGDIR/run.log"
OUT="$OUTDIR/car_insurance_fraud_synthetic.csv"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

log "=== exp_3 run started (5500 steps, one model) ==="

log ">>> [$NAME] TRAIN start"
python -u main.py --dataname "$NAME" --mode train --no_wandb --exp_name "$NAME" --resume \
    >> "$LOGDIR/train_${NAME}.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "!!! [$NAME] TRAIN FAILED (exit $rc) — aborting"
  exit 1
fi
log "<<< [$NAME] TRAIN done"

log ">>> [$NAME] GENERATE (original class distribution + size)"
python -u gen_exp3.py --dataname "$NAME" --exp_name "$NAME" --out "$OUT" \
    > "$LOGDIR/gen_${NAME}.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "!!! [$NAME] GENERATE FAILED (exit $rc)"
  exit 1
fi
log "<<< [$NAME] GENERATE done -> $OUT"
log "=== exp_3 run finished ==="
