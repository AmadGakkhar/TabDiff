#!/bin/bash
# Train one positive-only TabDiff model per GMSC split from train_inner.csv (v2
# methodology), then generate 10x positives. val.csv / test.csv are NEVER read.
# Outputs:
#   /srv/datasets/GMSC/splits/level_<N>/synthetic_positives_10x_inner.csv
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff
LOG=logs/gmsc_inner
mkdir -p "$LOG"
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $1" | tee -a "$LOG/run.log"; }

# Positive counts in train_inner.csv per split; 10x is the generation target.
POS=(1283 2566 3850 5134 6417)

do_split(){  # split gpu
  local s=$1 g=$2
  local n=gmsc_l${s}_inner
  local ten_x=$(( ${POS[$s]} * 10 ))
  local out="/srv/datasets/GMSC/splits/level_$s/synthetic_positives_10x_inner.csv"

  say ">>> [$n] prepare (train_inner.csv)"
  python -u prepare_gmsc_inner.py "$s" > "$LOG/prepare_$n.log" 2>&1 || { say "!!! [$n] prepare FAILED"; return 1; }

  say ">>> [$n] TRAIN on GPU $g"
  python -u main.py --dataname "$n" --mode train --gpu "$g" \
    --config_path "tabdiff/configs/tabdiff_configs_$n.toml" --exp_name "$n" --no_wandb --resume \
    > "$LOG/train_$n.log" 2>&1 || { say "!!! [$n] TRAIN FAILED"; return 1; }

  say "<<< [$n] TRAIN done; GENERATE $ten_x -> $out"
  python -u gen_gmsc_inner.py --split "$s" --n "$ten_x" --gpu "$g" --batch 8192 --out "$out" \
    > "$LOG/gen_$n.log" 2>&1 || { say "!!! [$n] GENERATE FAILED"; return 1; }

  say "<<< [$n] DONE -> $out"
}

say "=== GMSC inner (10x): 5 splits START ==="
do_split 0 0 &
do_split 1 1 &
do_split 2 2 &
do_split 3 3 &
wait
do_split 4 0
say "=== GMSC inner (10x): ALL DONE ==="
