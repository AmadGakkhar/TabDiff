#!/bin/bash
# Diabetes d1 (positive-only) on all 5 splits. Train on diabetes==1 rows from
# splits/<id>/train.csv ONLY (test.csv untouched); reject-generate 100k positive
# synthetic rows per split. Moderate-novelty config (see prepare_diab_d1.py).
# 4 GPUs: splits 0-3 in parallel (one GPU each), then split 4 on GPU 0.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff
LOG=logs/diabetes_d1_only
mkdir -p "$LOG"
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $1" | tee -a "$LOG/run.log"; }

do_split(){  # split gpu
  local s=$1 g=$2 n=db$1_d1
  say ">>> [$n] prepare"
  python -u prepare_diab_d1.py "$s" > "$LOG/prepare_$n.log" 2>&1 || { say "!!! [$n] prepare FAILED"; return 1; }
  say ">>> [$n] TRAIN on GPU $g (resume-skips if already trained)"
  python -u main.py --dataname "$n" --mode train --gpu "$g" \
    --config_path "tabdiff/configs/tabdiff_configs_$n.toml" --exp_name "$n" --no_wandb --resume \
    > "$LOG/train_$n.log" 2>&1
  say "<<< [$n] TRAIN done (rc=$?); GENERATE 100000 (reject)"
  python -u generate_paysim_app2.py --dataname "$n" --gpu "$g" \
    --n_fraud 100000 --mode reject --batch 32768 \
    --out_dir tabdiff/synthetic_diabetes --label diabetes \
    > "$LOG/gen_$n.log" 2>&1
  say "<<< [$n] GENERATE done (rc=$?)"
}

say "=== diabetes d1: splits 0,1,2,3 START ==="
do_split 0 0 &
do_split 1 1 &
do_split 2 2 &
do_split 3 3 &
wait
say "=== diabetes d1: splits 0-3 done; split 4 START ==="
do_split 4 0
say "=== diabetes d1: ALL SPLITS DONE ==="
