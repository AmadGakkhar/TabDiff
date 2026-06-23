#!/bin/bash
# v3: train one positive-only TabDiff model per GMSC split from the NEW (reduced-positive)
# train.csv, tuned to avoid memorising the tiny positive sets (160-802 rows) while keeping
# variation, then generate 10x positives. All 4 GPUs used in parallel.
# Output: /srv/datasets/GMSC/splits/level_<N>/synthetic_positives_10x.csv
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff
LOG=logs/gmsc_v3
mkdir -p "$LOG"
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $1" | tee -a "$LOG/run.log"; }

# Positive counts in the NEW train.csv per split; 10x is the generation target.
POS=(160 321 481 642 802)

do_split(){  # split gpu
  local s=$1 g=$2
  local n=gmsc_l${s}_v3
  local ten_x=$(( ${POS[$s]} * 10 ))
  local out="/srv/datasets/GMSC/splits/level_$s/synthetic_positives_10x.csv"

  say ">>> [$n] prepare (train.csv, ${POS[$s]} pos)"
  python -u prepare_gmsc_v3.py "$s" > "$LOG/prepare_$n.log" 2>&1 || { say "!!! [$n] prepare FAILED"; return 1; }

  say ">>> [$n] TRAIN on GPU $g"
  python -u main.py --dataname "$n" --mode train --gpu "$g" \
    --config_path "tabdiff/configs/tabdiff_configs_$n.toml" --exp_name "$n" --no_wandb \
    > "$LOG/train_$n.log" 2>&1 || { say "!!! [$n] TRAIN FAILED"; return 1; }

  say "<<< [$n] TRAIN done; GENERATE $ten_x -> $out"
  python -u gen_gmsc_v3.py --split "$s" --n "$ten_x" --gpu "$g" --batch 8192 --out "$out" \
    > "$LOG/gen_$n.log" 2>&1 || { say "!!! [$n] GENERATE FAILED"; return 1; }

  say "<<< [$n] DONE -> $out"
}

say "=== GMSC v3 (10x, tiny-data tuning): 5 splits START ==="
do_split 0 0 &
do_split 1 1 &
do_split 2 2 &
do_split 3 3 &
wait
do_split 4 0
say "=== GMSC v3: ALL DONE ==="
