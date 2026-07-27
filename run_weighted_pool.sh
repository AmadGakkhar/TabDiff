#!/usr/bin/env bash
# Quality-weighted pooling for Approach 2.
#
# Stage 1  reservoirs  — generate up to N_RESERVOIR synthetic fraud rows per
#                        trained dosage model (so any mixing weight up to 100%
#                        is realisable; the 625-row run outputs cannot exceed 25%).
# Stage 2  pool        — score each model on data quality, convert to mixing
#                        weights, sample N_TOTAL rows proportionally.
#
# Models are already trained (Stage 2 of run_expv01_app2.sh); this only samples.
# Generation does not need the per-model TOML (arch is restored from the
# checkpoint), so no config swap is required here.

set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff

N_RESERVOIR=2500
N_TOTAL=2500
RES_DIR=synthetic_outputs/expv01_app2/reservoirs
LOGDIR=logs/expv01_app2
OUT=synthetic_outputs/expv01_app2/pooled_fraud_weighted.csv
REPORT=synthetic_outputs/expv01_app2/weight_report.json

mkdir -p "$RES_DIR" "$LOGDIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# Per-model sampling params, sized to each model's fraud yield so the reservoir
# fills in a few iterations. d4 (~6% yield) needs large chunks.
declare -A CHUNK=(    [expv01_d1]=6000  [expv01_d2]=8192  [expv01_d3]=8192  [expv01_d4]=50000 )
declare -A MAXITERS=( [expv01_d1]=8     [expv01_d2]=15    [expv01_d3]=25    [expv01_d4]=8     )

# ── Stage 1: reservoirs ────────────────────────────────────────────────────────
log "=== Stage 1: build reservoirs (${N_RESERVOIR} fraud/model) ==="
for d in expv01_d1 expv01_d2 expv01_d3 expv01_d4; do
    OUT_RES="${RES_DIR}/${d}_fraud.csv"
    if [[ -f "$OUT_RES" ]]; then
        have=$(($(wc -l < "$OUT_RES") - 1))
        if [[ $have -ge $N_RESERVOIR ]]; then
            log "[$d] reservoir already has $have rows — skipping"
            continue
        fi
    fi
    log ">>> [$d] generate ${N_RESERVOIR} (chunk=${CHUNK[$d]}, max_iters=${MAXITERS[$d]})"
    python -u gen_expv01_app1.py \
        --dataname  "$d" --exp_name "$d" \
        --n_fraud   "$N_RESERVOIR" \
        --out       "$OUT_RES" \
        --chunk     "${CHUNK[$d]}" \
        --max_iters "${MAXITERS[$d]}" \
        --seed      42 \
        > "${LOGDIR}/reservoir_${d}.log" 2>&1
    if [[ $? -ne 0 ]]; then
        log "!!! [$d] reservoir generation FAILED — see ${LOGDIR}/reservoir_${d}.log"
        exit 1
    fi
    log "<<< [$d] reservoir done -> $OUT_RES ($(($(wc -l < "$OUT_RES") - 1)) rows)"
done

# ── Stage 2: quality-weighted pool ─────────────────────────────────────────────
log "=== Stage 2: score + weight + pool -> $OUT ==="
python -u pool_weighted.py \
    --reservoir_dir "$RES_DIR" \
    --out           "$OUT" \
    --report        "$REPORT" \
    --n_total       "$N_TOTAL" \
    2>&1 | tee "${LOGDIR}/pool_weighted.log"

log "=== done ==="
log "    Weighted pool -> $OUT"
log "    Score report  -> $REPORT"
