#!/usr/bin/env bash
# Orchestrate exp_2: for each partition, train an independent TabDiff model
# (5500 epochs) then generate a fully-synthetic, class-balanced version.
# Logs stream to logs/exp_2/ for real-time viewing (tail -f).
set -u

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff

LOGDIR=logs/exp_2
OUTDIR=synthetic_outputs/exp_2
mkdir -p "$LOGDIR" "$OUTDIR"
RUNLOG="$LOGDIR/run.log"

# partition -> synthetic_fraud_rows (from data/exp_2/generation_manifest.json).
# Balanced output => generate this many fraud AND this many non-fraud rows.
PARTS=(full_train fold0_train fold1_train fold2_train fold3_train fold4_train)
declare -A NROWS=(
  [full_train]=11947
  [fold0_train]=9557
  [fold1_train]=9556
  [fold2_train]=9556
  [fold3_train]=9558
  [fold4_train]=9558
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

log "=== exp_2 run started (5500 epochs/model, one model per partition) ==="

for part in "${PARTS[@]}"; do
  n=${NROWS[$part]}
  if [ -f "$OUTDIR/${part}_balanced.csv" ]; then
    log "=== [$part] already complete (output exists) — skipping"
    continue
  fi
  log ">>> [$part] TRAIN start (5500 epochs)"
  python -u main.py --dataname "$part" --mode train --no_wandb --exp_name "$part" --resume \
      >> "$LOGDIR/train_${part}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "!!! [$part] TRAIN FAILED (exit $rc) — skipping generation, continuing"
    continue
  fi
  log "<<< [$part] TRAIN done"

  log ">>> [$part] GENERATE balanced (fraud=$n, non-fraud=$n)"
  python -u gen_balanced.py --dataname "$part" --exp_name "$part" \
      --n_fraud "$n" --n_nonfraud "$n" \
      --out "$OUTDIR/${part}_balanced.csv" \
      > "$LOGDIR/gen_${part}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "!!! [$part] GENERATE FAILED (exit $rc) — continuing"
    continue
  fi
  log "<<< [$part] GENERATE done -> $OUTDIR/${part}_balanced.csv"
done

log "=== exp_2 run finished ==="
