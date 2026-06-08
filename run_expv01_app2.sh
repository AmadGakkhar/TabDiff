#!/usr/bin/env bash
# Approach 2 — Majority-Dosage Sweep Ensemble (M=4) for exp-v01
#
# Stage 1  prepare_expv01_app2.py  — build 4 dosage partitions, register
#                                    datasets, write per-model TOML configs
# Stage 2  main.py (x4)            — train one TabDiff per dosage model
# Stage 3  gen_expv01_app1.py (x4) — generate 625 fraud rows per model
# Stage 4  pool                    — concat + shuffle -> pooled_fraud.csv (2500 rows)
#
# Per-model configs are sized to partition size (see prepare_expv01_app2.py).
# Model d4 (~6% yield) uses large chunk + more iterations to reach target.
# Config swap uses the same trap-based guard as run_expv01_app1.sh.

set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff

# ── Configuration ─────────────────────────────────────────────────────────────
N_FRAUD_PER_MODEL=625           # 4 × 625 = 2500 pooled fraud rows

LOGDIR=logs/expv01_app2
OUTDIR=synthetic_outputs/expv01_app2
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

# ── Stage 1: Prepare ──────────────────────────────────────────────────────────
log "=== Stage 1: build dosage partitions + register datasets ==="
python -u prepare_expv01_app2.py 2>&1 | tee -a "${LOGDIR}/prepare.log"

cp "$ORIG_CFG" "$BAK_CFG"
log "Original config backed up -> $BAK_CFG"

# ── Per-model generation parameters ──────────────────────────────────────────
# d1 (fraud only, ~100% yield): small chunk, few iters
# d2 (+25% majority, ~20% yield): moderate
# d3 (+50% majority, ~11% yield): more iters
# d4 (+100% majority, ~6% yield): large chunk, many iters
declare -A CHUNK=(    [expv01_d1]=4096  [expv01_d2]=4096  [expv01_d3]=4096  [expv01_d4]=50000 )
declare -A MAXITERS=( [expv01_d1]=10   [expv01_d2]=20    [expv01_d3]=30    [expv01_d4]=60    )
declare -A DOSE_LABEL=(
    [expv01_d1]="fraud only"
    [expv01_d2]="fraud + 25% majority"
    [expv01_d3]="fraud + 50% majority"
    [expv01_d4]="fraud + 100% majority"
)

# ── Stage 2 + 3: Train then generate, model by model ─────────────────────────
log "=== Stage 2+3: train + generate (M=4 dosage models) ==="

for d in expv01_d1 expv01_d2 expv01_d3 expv01_d4; do
    OUT="${OUTDIR}/${d}_fraud.csv"
    MODEL_CFG="tabdiff/configs/tabdiff_configs_${d}.toml"
    TRAIN_LOG="${LOGDIR}/train_${d}.log"
    GEN_LOG="${LOGDIR}/gen_${d}.log"
    LABEL="${DOSE_LABEL[$d]}"

    if [[ -f "$OUT" ]]; then
        log "[$d] output already exists — skipping"
        continue
    fi

    # ── Train ──────────────────────────────────────────────────────────────
    log ">>> [$d] TRAIN start ($LABEL)"
    cp "$MODEL_CFG" "$ORIG_CFG"

    python -u main.py \
        --dataname  "$d" \
        --mode      train \
        --no_wandb        \
        --exp_name  "$d"  \
        --resume          \
        >> "$TRAIN_LOG" 2>&1
    rc=$?

    cp "$BAK_CFG" "$ORIG_CFG"

    if [[ $rc -ne 0 ]]; then
        log "!!! [$d] TRAIN FAILED (exit $rc) — see $TRAIN_LOG — aborting"
        exit 1
    fi
    log "<<< [$d] TRAIN done"

    # ── Generate ───────────────────────────────────────────────────────────
    log ">>> [$d] GENERATE ($N_FRAUD_PER_MODEL fraud rows, chunk=${CHUNK[$d]}, max_iters=${MAXITERS[$d]})"

    python -u gen_expv01_app1.py \
        --dataname  "$d"                     \
        --exp_name  "$d"                     \
        --n_fraud   "$N_FRAUD_PER_MODEL"     \
        --out       "$OUT"                   \
        --chunk     "${CHUNK[$d]}"           \
        --max_iters "${MAXITERS[$d]}"        \
        --seed      42                       \
        > "$GEN_LOG" 2>&1
    rc=$?

    if [[ $rc -ne 0 ]]; then
        log "!!! [$d] GENERATE FAILED (exit $rc) — see $GEN_LOG — aborting"
        exit 1
    fi
    log "<<< [$d] GENERATE done -> $OUT"
done

# ── Stage 4: Pool ─────────────────────────────────────────────────────────────
POOL="${OUTDIR}/pooled_fraud.csv"
log "=== Stage 4: pool dosage outputs -> $POOL ==="

python - "$OUTDIR" "$POOL" <<'PY'
import sys, pandas as pd, glob

outdir, pool_path = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{outdir}/expv01_d*_fraud.csv"))
if not files:
    print("ERROR: no dosage fraud files found", file=sys.stderr)
    sys.exit(1)

parts = [pd.read_csv(f) for f in files]
pool  = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
pool.to_csv(pool_path, index=False)

vc = pool["FraudFound_P"].value_counts().to_dict()
print(f"Pooled {len(pool)} rows from {len(parts)} dosage models | class counts: {vc}")
for i, (f, p) in enumerate(zip(files, parts), 1):
    print(f"  model {i}: {len(p)} fraud rows  <-  {f}")
PY

log "=== Approach 2 run finished ==="
log "    Pooled synthetic fraud  -> $POOL"
log "    Logs                    -> $LOGDIR/"
log "    Per-model fraud CSVs    -> $OUTDIR/expv01_d*_fraud.csv"
