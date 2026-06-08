#!/usr/bin/env bash
# Approach 1 — Balanced-Bagging Ensemble (M=4) for exp-v01
#
# Stage 1  prepare_expv01_app1.py  — build 4 balanced partitions, register
#                                    datasets, write member TOML config
# Stage 2  main.py (x4)            — train one TabDiff per member
# Stage 3  gen_expv01_app1.py (x4) — generate 625 fraud rows per member
# Stage 4  pool                    — concat + shuffle -> pooled_fraud.csv (2500 rows)
#
# Logs stream to logs/expv01_app1/ for real-time monitoring (tail -f).
# Idempotent: completed members (output CSV exists) are skipped on re-run.
# Config swap: the member TOML overwrites tabdiff/configs/tabdiff_configs.toml
# before each training run; a trap restores the original on any exit.

set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabdiff

# ── Configuration ─────────────────────────────────────────────────────────────
M=4
N_FRAUD_PER_MEMBER=625          # 4 × 625 = 2500 pooled fraud rows

LOGDIR=logs/expv01_app1
OUTDIR=synthetic_outputs/expv01_app1
ORIG_CFG=tabdiff/configs/tabdiff_configs.toml
MEMBER_CFG=tabdiff/configs/tabdiff_configs_expv01_app1.toml
BAK_CFG="${ORIG_CFG}.bak"
RUNLOG="${LOGDIR}/run.log"

mkdir -p "$LOGDIR" "$OUTDIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

# ── Config guard ───────────────────────────────────────────────────────────────
# Restore the original TabDiff config on any exit (Ctrl-C, error, or normal
# finish) so other experiments are not affected.
_restore_cfg() {
    if [[ -f "$BAK_CFG" ]]; then
        cp "$BAK_CFG" "$ORIG_CFG"
        rm -f "$BAK_CFG"
        echo "[$(ts)] Original TabDiff config restored."
    fi
}
trap _restore_cfg EXIT

# ── Stage 1: Prepare ──────────────────────────────────────────────────────────
log "=== Stage 1: build partitions + register datasets ==="
python -u prepare_expv01_app1.py 2>&1 | tee -a "${LOGDIR}/prepare.log"

# Backup original config once (after prepare writes the member config).
cp "$ORIG_CFG" "$BAK_CFG"
log "Original config backed up -> $BAK_CFG"

# ── Stage 2 + 3: Train then generate, member by member ───────────────────────
log "=== Stage 2+3: train + generate (M=$M, $N_FRAUD_PER_MEMBER fraud/member) ==="

for m in $(seq 1 $M); do
    NAME="expv01_m${m}"
    OUT="${OUTDIR}/member_${m}_fraud.csv"
    TRAIN_LOG="${LOGDIR}/train_${NAME}.log"
    GEN_LOG="${LOGDIR}/gen_${NAME}.log"

    if [[ -f "$OUT" ]]; then
        log "[$NAME] output already exists — skipping"
        continue
    fi

    # ── Train ──────────────────────────────────────────────────────────────
    log ">>> [$NAME] TRAIN start (member config: dim_t=512, batch=256, steps=2000)"
    cp "$MEMBER_CFG" "$ORIG_CFG"

    python -u main.py \
        --dataname  "$NAME" \
        --mode      train   \
        --no_wandb          \
        --exp_name  "$NAME" \
        --resume            \
        >> "$TRAIN_LOG" 2>&1
    rc=$?

    # Restore config immediately after training regardless of outcome.
    cp "$BAK_CFG" "$ORIG_CFG"

    if [[ $rc -ne 0 ]]; then
        log "!!! [$NAME] TRAIN FAILED (exit $rc) — see $TRAIN_LOG — aborting"
        exit 1
    fi
    log "<<< [$NAME] TRAIN done"

    # ── Generate ───────────────────────────────────────────────────────────
    log ">>> [$NAME] GENERATE ($N_FRAUD_PER_MEMBER fraud rows, seed=$m)"

    python -u gen_expv01_app1.py \
        --dataname "$NAME"       \
        --exp_name "$NAME"       \
        --n_fraud  "$N_FRAUD_PER_MEMBER" \
        --out      "$OUT"        \
        --seed     "$m"          \
        > "$GEN_LOG" 2>&1
    rc=$?

    if [[ $rc -ne 0 ]]; then
        log "!!! [$NAME] GENERATE FAILED (exit $rc) — see $GEN_LOG — aborting"
        exit 1
    fi
    log "<<< [$NAME] GENERATE done -> $OUT"
done

# ── Stage 4: Pool ─────────────────────────────────────────────────────────────
POOL="${OUTDIR}/pooled_fraud.csv"
log "=== Stage 4: pool member outputs -> $POOL ==="

python - "$OUTDIR" "$POOL" <<'PY'
import sys, pandas as pd, glob

outdir, pool_path = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{outdir}/member_*_fraud.csv"))
if not files:
    print("ERROR: no member fraud files found", file=sys.stderr)
    sys.exit(1)

parts  = [pd.read_csv(f) for f in files]
pool   = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
pool.to_csv(pool_path, index=False)

vc = pool["FraudFound_P"].value_counts().to_dict()
print(f"Pooled {len(pool)} rows from {len(parts)} members | class counts: {vc}")
for i, (f, p) in enumerate(zip(files, parts), 1):
    print(f"  member {i}: {len(p)} fraud rows  <-  {f}")
PY

log "=== Approach 1 run finished ==="
log "    Pooled synthetic fraud  -> $POOL"
log "    Logs                    -> $LOGDIR/"
log "    Per-member fraud CSVs   -> $OUTDIR/member_*_fraud.csv"
