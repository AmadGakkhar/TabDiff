#!/bin/bash
# Train ONLY d1 (fraud-only) for paysim splits 1 and 2, in parallel (one GPU each),
# then reject-generate 100k fraud rows per split. Split 0 d1 is already done.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff
LOG=logs/paysim_d1_only
mkdir -p "$LOG"
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $1" | tee -a "$LOG/run.log"; }

do_split(){  # split gpu
  local s=$1 g=$2 n=ps$1_d1
  say ">>> [$n] prepare"
  python -u prepare_d1.py "$s" > "$LOG/prepare_$n.log" 2>&1 || { say "!!! [$n] prepare FAILED"; return 1; }
  say ">>> [$n] TRAIN on GPU $g"
  python -u main.py --dataname "$n" --mode train --gpu "$g" \
    --config_path "tabdiff/configs/tabdiff_configs_$n.toml" --exp_name "$n" --no_wandb --resume \
    > "$LOG/train_$n.log" 2>&1
  say "<<< [$n] TRAIN done (rc=$?); GENERATE 100000 (reject)"
  python -u generate_paysim_app2.py --dataname "$n" --gpu "$g" \
    --n_fraud 100000 --mode reject --batch 32768 \
    > "$LOG/gen_$n.log" 2>&1
  say "<<< [$n] GENERATE done (rc=$?)"
}

say "=== d1-only: splits 1,2 START ==="
do_split 1 0 &
do_split 2 1 &
wait
say "=== d1-only: splits 1,2 ALL DONE ==="
