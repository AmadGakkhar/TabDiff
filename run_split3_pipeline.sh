#!/bin/bash
# Pipeline split 3 onto GPUs as they free from split 2.
# GPU0 is free now (ps2_d1 done) -> start ps3_d1 immediately.
# GPU1/2/3 free when ps2_d2/d3/d4 finish generating -> start matching ps3 dose then.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tabdiff
cd /home/amad/projects/TabDiff
LOG=logs/paysim_app2_ps3
mkdir -p $LOG
declare -A TGT=( [d1]=225000 [d2]=90000 [d3]=45000 [d4]=25000 )

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
runlog(){ echo "[$(stamp)] $1" | tee -a $LOG/run.log; }

run_dose(){  # dose gpu
  local d=$1 g=$2
  runlog ">>> [ps3_$d] TRAIN start on GPU $g"
  python -u main.py --dataname ps3_$d --mode train --gpu $g \
    --config_path tabdiff/configs/tabdiff_configs_ps3_$d.toml --exp_name ps3_$d --no_wandb \
    > $LOG/train_ps3_$d.log 2>&1
  runlog "<<< [ps3_$d] TRAIN done (rc=$?) on GPU $g; GENERATE ${TGT[$d]}"
  python -u generate_paysim_app2.py --dataname ps3_$d --gpu $g \
    --n_fraud ${TGT[$d]} --mode reject --batch 32768 --max_seconds 10800 \
    > $LOG/gen_ps3_$d.log 2>&1
  runlog "<<< [ps3_$d] GENERATE done (rc=$?) on GPU $g"
}

# wait until split-3 data+config for a dose is ready
wait_ready(){ while [ ! -f data/ps3_$1/info.json ] || [ ! -f tabdiff/configs/tabdiff_configs_ps3_$1.toml ]; do sleep 10; done; }

runlog "=== split-3 pipeline start (targets ${TGT[*]}) ==="

# GPU0: free now
( wait_ready d1; run_dose d1 0 ) &

# GPU1/2/3: launch when the matching split-2 dose finishes generating (GPU frees)
( while ! grep -q 'DONE:' logs/paysim_app2_ps2/gen_ps2_d2.log 2>/dev/null; do sleep 60; done
  runlog "GPU1 freed (ps2_d2 done)"; wait_ready d2; run_dose d2 1 ) &
( while ! grep -q 'DONE:' logs/paysim_app2_ps2/gen_ps2_d3.log 2>/dev/null; do sleep 60; done
  runlog "GPU2 freed (ps2_d3 done)"; wait_ready d3; run_dose d3 2 ) &
( while ! grep -q 'DONE:' logs/paysim_app2_ps2/gen_ps2_d4.log 2>/dev/null; do sleep 60; done
  runlog "GPU3 freed (ps2_d4 done)"; wait_ready d4; run_dose d4 3 ) &
wait
runlog "=== split-3 pipeline ALL DONE ==="
