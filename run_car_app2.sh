#!/usr/bin/env bash
# Approach 2 — Majority-Dosage Sweep Ensemble (M=4) for car_fraud.
#
# Stage 1  prepare_car_app2.py     — clean, build 4 dosage partitions, register,
#                                     write per-model TOML configs
# Stage 2  main.py (x4)            — train one TabDiff per dosage model
# Stage 3  gen_expv01_app1.py (x4) — generate 625 fraud rows per model
# Stage 4  pool                    — concat + shuffle -> pooled_fraud.csv (2500)
#
# Same config-swap-with-trap pattern as run_expv01_app2.sh.
#
# Backend: by default the train/generate stages run locally as plain Python (unchanged).
# Set BACKEND=local or BACKEND=cloud to route those stages through cloud/launch.py
# (SageMaker local mode / managed cloud). See docs/sagemaker.md.
#   BACKEND=cloud ./run_car_app2.sh

set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff

N_FRAUD_PER_MODEL=625
TARGET_COL=fraud_reported
LOGDIR=logs/car_app2
OUTDIR=synthetic_outputs/car_app2
ORIG_CFG=tabdiff/configs/tabdiff_configs.toml
BAK_CFG="${ORIG_CFG}.bak"
RUNLOG="${LOGDIR}/run.log"

mkdir -p "$LOGDIR" "$OUTDIR"
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

_restore_cfg() {
    if [[ -f "$BAK_CFG" ]]; then
        cp "$BAK_CFG" "$ORIG_CFG"
        rm -f "$BAK_CFG"
        echo "[$(ts)] Original TabDiff config restored."
    fi
}
trap _restore_cfg EXIT

# ── Stage 1: prepare ────────────────────────────────────────────────────────
log "=== Stage 1: clean + build dosage partitions + register ==="
python -u prepare_car_app2.py 2>&1 | tee -a "${LOGDIR}/prepare.log"

cp "$ORIG_CFG" "$BAK_CFG"
log "Original config backed up -> $BAK_CFG"

# Per-model generation params, sized to each model's fraud yield.
# Base fraud rate 11.5%, so yields run roughly: d1 high, d2 ~34%, d3 ~21%, d4 ~11.5%.
declare -A CHUNK=(    [car_d1]=4096 [car_d2]=4096 [car_d3]=4096 [car_d4]=8192 )
declare -A MAXITERS=( [car_d1]=6    [car_d2]=6    [car_d3]=10   [car_d4]=15   )
declare -A DOSE_LABEL=(
    [car_d1]="fraud only"
    [car_d2]="fraud + 25% majority"
    [car_d3]="fraud + 50% majority"
    [car_d4]="fraud + 100% majority"
)

# ── Stage 2 + 3: train then generate, model by model ─────────────────────────
log "=== Stage 2+3: train + generate (M=4 dosage models) ==="
for d in car_d1 car_d2 car_d3 car_d4; do
    OUT="${OUTDIR}/${d}_fraud.csv"
    MODEL_CFG="tabdiff/configs/tabdiff_configs_${d}.toml"
    TRAIN_LOG="${LOGDIR}/train_${d}.log"
    GEN_LOG="${LOGDIR}/gen_${d}.log"
    LABEL="${DOSE_LABEL[$d]}"

    if [[ -f "$OUT" ]]; then
        log "[$d] output already exists — skipping"
        continue
    fi

    log ">>> [$d] TRAIN start ($LABEL)"
    if [[ -z "${BACKEND:-}" ]]; then
        # Original local behavior: swap the default config in place, run, restore.
        cp "$MODEL_CFG" "$ORIG_CFG"
        python -u main.py --dataname "$d" --mode train --no_wandb --exp_name "$d" --resume \
            >> "$TRAIN_LOG" 2>&1
        rc=$?
        cp "$BAK_CFG" "$ORIG_CFG"
    else
        # SageMaker (local or cloud): pass the per-model config via --config_path; no swap.
        python -u cloud/launch.py train --backend "$BACKEND" \
            --dataname "$d" --exp_name "$d" --resume --config_path "$MODEL_CFG" \
            >> "$TRAIN_LOG" 2>&1
        rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        log "!!! [$d] TRAIN FAILED (exit $rc) — see $TRAIN_LOG — aborting"
        exit 1
    fi
    log "<<< [$d] TRAIN done"

    log ">>> [$d] GENERATE ($N_FRAUD_PER_MODEL fraud rows, chunk=${CHUNK[$d]}, max_iters=${MAXITERS[$d]})"
    if [[ -z "${BACKEND:-}" ]]; then
        python -u gen_expv01_app1.py \
            --dataname   "$d" \
            --exp_name   "$d" \
            --n_fraud    "$N_FRAUD_PER_MODEL" \
            --out        "$OUT" \
            --target_col "$TARGET_COL" \
            --chunk      "${CHUNK[$d]}" \
            --max_iters  "${MAXITERS[$d]}" \
            --seed       42 \
            > "$GEN_LOG" 2>&1
        rc=$?
    else
        # --gpu is managed by the container; do not pass it here.
        python -u cloud/launch.py generate --backend "$BACKEND" \
            --gen-script gen_expv01_app1.py \
            --dataname   "$d" \
            --exp_name   "$d" \
            --n_fraud    "$N_FRAUD_PER_MODEL" \
            --out        "$OUT" \
            --target_col "$TARGET_COL" \
            --chunk      "${CHUNK[$d]}" \
            --max_iters  "${MAXITERS[$d]}" \
            --seed       42 \
            > "$GEN_LOG" 2>&1
        rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        log "!!! [$d] GENERATE FAILED (exit $rc) — see $GEN_LOG — aborting"
        exit 1
    fi
    log "<<< [$d] GENERATE done -> $OUT"
done

# ── Stage 4: pool ─────────────────────────────────────────────────────────────
POOL="${OUTDIR}/pooled_fraud.csv"
log "=== Stage 4: pool dosage outputs -> $POOL ==="
python - "$OUTDIR" "$POOL" "$TARGET_COL" <<'PY'
import sys, pandas as pd, glob
outdir, pool_path, target = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(f"{outdir}/car_d*_fraud.csv"))
if not files:
    print("ERROR: no dosage fraud files found", file=sys.stderr); sys.exit(1)
parts = [pd.read_csv(f) for f in files]
pool  = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
pool.to_csv(pool_path, index=False)
vc = pool[target].value_counts().to_dict()
print(f"Pooled {len(pool)} rows from {len(parts)} dosage models | class counts: {vc}")
for i, (f, p) in enumerate(zip(files, parts), 1):
    print(f"  model {i}: {len(p)} fraud rows  <-  {f}")
PY

log "=== Approach 2 (car_fraud) run finished ==="
log "    Pooled synthetic fraud  -> $POOL"
log "    Logs                    -> $LOGDIR/"
log "    Per-model fraud CSVs    -> $OUTDIR/car_d*_fraud.csv"
