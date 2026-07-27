#!/usr/bin/env bash
# Scale the car_fraud Approach-2 pool to 20,000 fraud rows (5,000 per model).
#
# Generation-only: the 4 dosage models (car_d1..car_d4) are already trained, so
# this just samples more fraud from each checkpoint and re-pools. No retraining,
# no config swap (generation restores the arch from the checkpoint).

set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff

N_PER_MODEL=5000                # 4 x 5000 = 20000
TARGET_COL=fraud_reported
OUTDIR=synthetic_outputs/car_app2/pool20k
LOGDIR=logs/car_app2
POOL=synthetic_outputs/car_app2/pooled_fraud_20k.csv

mkdir -p "$OUTDIR" "$LOGDIR"
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# Per-model sampling sized to fraud yield (d1 100%, d2 ~34%, d3 ~18.5%, d4 ~11.5%).
declare -A CHUNK=(    [car_d1]=6000  [car_d2]=8192  [car_d3]=8192  [car_d4]=50000 )
declare -A MAXITERS=( [car_d1]=4     [car_d2]=12    [car_d3]=30    [car_d4]=8     )

log "=== Generate ${N_PER_MODEL} fraud/model from trained car_d* checkpoints ==="
for d in car_d1 car_d2 car_d3 car_d4; do
    OUT="${OUTDIR}/${d}_fraud.csv"
    if [[ -f "$OUT" ]]; then
        have=$(($(wc -l < "$OUT") - 1))
        if [[ $have -ge $N_PER_MODEL ]]; then
            log "[$d] already has $have rows — skipping"
            continue
        fi
    fi
    log ">>> [$d] generate ${N_PER_MODEL} (chunk=${CHUNK[$d]}, max_iters=${MAXITERS[$d]})"
    python -u gen_expv01_app1.py \
        --dataname   "$d" \
        --exp_name   "$d" \
        --n_fraud    "$N_PER_MODEL" \
        --out        "$OUT" \
        --target_col "$TARGET_COL" \
        --chunk      "${CHUNK[$d]}" \
        --max_iters  "${MAXITERS[$d]}" \
        --seed       42 \
        > "${LOGDIR}/gen20k_${d}.log" 2>&1
    if [[ $? -ne 0 ]]; then
        log "!!! [$d] GENERATE FAILED — see ${LOGDIR}/gen20k_${d}.log — aborting"
        exit 1
    fi
    log "<<< [$d] done -> $OUT ($(($(wc -l < "$OUT") - 1)) rows)"
done

log "=== pool -> $POOL ==="
python - "$OUTDIR" "$POOL" "$TARGET_COL" <<'PY'
import sys, pandas as pd, glob
outdir, pool_path, target = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(f"{outdir}/car_d*_fraud.csv"))
parts = [pd.read_csv(f) for f in files]
pool  = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
pool.to_csv(pool_path, index=False)
print(f"Pooled {len(pool)} rows from {len(parts)} models | class counts: {pool[target].value_counts().to_dict()}")
for i, (f, p) in enumerate(zip(files, parts), 1):
    print(f"  model {i}: {len(p)} fraud rows  <-  {f}")
PY

log "=== done -> $POOL ==="
