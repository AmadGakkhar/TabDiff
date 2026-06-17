#!/bin/bash
# Train one positive-only TabDiff model per GMSC split (level_0..level_4), in
# parallel across the 4 T4 GPUs, then reject-generate 5x the original positive
# count per split. Output CSVs are written beside each split:
#   /srv/datasets/GMSC/splits/level_<N>/synthetic_positives_5x.csv
# The held-out test.csv is never read.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff
LOG=logs/gmsc_splits
mkdir -p "$LOG"
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $1" | tee -a "$LOG/run.log"; }

# Original positive counts per split (from manifest.json); 5x is the generation target.
POS=(1604 3208 4813 6417 8021)

do_split(){  # split gpu
  local s=$1 g=$2
  local n=gmsc_l$s
  local five_x=$(( ${POS[$s]} * 5 ))
  local out="/srv/datasets/GMSC/splits/level_$s/synthetic_positives_5x.csv"

  say ">>> [$n] prepare"
  python -u prepare_gmsc.py "$s" > "$LOG/prepare_$n.log" 2>&1 || { say "!!! [$n] prepare FAILED"; return 1; }

  say ">>> [$n] TRAIN on GPU $g"
  python -u main.py --dataname "$n" --mode train --gpu "$g" \
    --config_path "tabdiff/configs/tabdiff_configs_$n.toml" --exp_name "$n" --no_wandb --resume \
    > "$LOG/train_$n.log" 2>&1 || { say "!!! [$n] TRAIN FAILED"; return 1; }

  say "<<< [$n] TRAIN done; GENERATE $five_x -> $out"
  python -u gen_gmsc.py --split "$s" --n "$five_x" --gpu "$g" --batch 8192 --out "$out" \
    > "$LOG/gen_$n.log" 2>&1 || { say "!!! [$n] GENERATE FAILED"; return 1; }

  say "<<< [$n] GENERATE done -> $out"
}

say "=== GMSC positive-only: 5 splits START ==="
# 4 GPUs (0..3); split 4 reuses GPU 0 (level_0 and level_4 are both small).
do_split 0 0 &
do_split 1 1 &
do_split 2 2 &
do_split 3 3 &
wait
do_split 4 0
say "=== GMSC positive-only: ALL DONE ==="
