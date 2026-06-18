#!/bin/bash
# v2: retrain one improved positive-only TabDiff model per GMSC split (missingness
# indicators + integer dequant + dim_t=512), then generate 5x positives with NaN
# restore / integer rounding / tail clipping. Outputs written ALONGSIDE v1:
#   /srv/datasets/GMSC/splits/level_<N>/synthetic_positives_5x_v2.csv
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff
LOG=logs/gmsc_v2
mkdir -p "$LOG"
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $1" | tee -a "$LOG/run.log"; }

POS=(1604 3208 4813 6417 8021)

do_split(){  # split gpu
  local s=$1 g=$2
  local n=gmsc_l${s}_v2
  local five_x=$(( ${POS[$s]} * 5 ))
  local out="/srv/datasets/GMSC/splits/level_$s/synthetic_positives_5x_v2.csv"

  say ">>> [$n] prepare"
  python -u prepare_gmsc_v2.py "$s" > "$LOG/prepare_$n.log" 2>&1 || { say "!!! [$n] prepare FAILED"; return 1; }

  say ">>> [$n] TRAIN on GPU $g"
  python -u main.py --dataname "$n" --mode train --gpu "$g" \
    --config_path "tabdiff/configs/tabdiff_configs_$n.toml" --exp_name "$n" --no_wandb --resume \
    > "$LOG/train_$n.log" 2>&1 || { say "!!! [$n] TRAIN FAILED"; return 1; }

  say "<<< [$n] TRAIN done; GENERATE $five_x -> $out"
  python -u gen_gmsc_v2.py --split "$s" --n "$five_x" --gpu "$g" --batch 8192 --out "$out" \
    > "$LOG/gen_$n.log" 2>&1 || { say "!!! [$n] GENERATE FAILED"; return 1; }

  say "<<< [$n] DONE -> $out"
}

say "=== GMSC v2: 5 splits START ==="
do_split 0 0 &
do_split 1 1 &
do_split 2 2 &
do_split 3 3 &
wait
do_split 4 0
say "=== GMSC v2: ALL DONE ==="
