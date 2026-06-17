"""Prepare the diabetes d1 (positive-only) dataset + config for a given split.

Positive-class-only TabDiff recipe (analogous to the paysim fraud `d1` recipe in
prepare_d1.py / prepare_paysim_app2.py), adapted to the diabetes_prediction
dataset and tuned for MODERATE NOVELTY (stronger weight decay + slightly fewer
epochs so the model generalizes rather than memorizing the ~6.8k positive rows).

Reads ONLY splits/<id>/train.csv (test.csv is never touched) and keeps only
diabetes == 1 rows. Registers as `db<SPLIT_ID>_d1` to avoid colliding with the
unrelated built-in `diabetes.json`.

  python prepare_diab_d1.py <SPLIT_ID>

Diabetes schema (9 cols, no drops):
  0 gender               cat
  1 age                  num
  2 hypertension (0/1)   cat
  3 heart_disease (0/1)  cat
  4 smoking_history      cat
  5 bmi                  num
  6 HbA1c_level          num
  7 blood_glucose_level  num
  8 diabetes  (target)   cat/target
"""
import os
import json
import copy
import shutil
import subprocess
import sys

import pandas as pd
import src

SPLIT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT_CSV = f"/home/amad/projects/datasets/diabetes/splits/{SPLIT_ID}/train.csv"
TARGET    = "diabetes"
INFO_DIR  = "data/Info"

N_COLS         = 9
NUM_COL_IDX    = [1, 5, 6, 7]
CAT_COL_IDX    = [0, 2, 3, 4]
TARGET_COL_IDX = [8]
# int-valued categorical/target columns cast to str for clean categorical handling
INT_CAT_IDX    = [2, 3]

name    = f"db{SPLIT_ID}_d1"
tmp_dir = "data/diabetes_d1_partitions"

# Base config (mirrors _BASE in prepare_paysim_app2.py) with the moderate-novelty
# d1 overrides applied: weight_decay 1e-3 (vs 1e-4), steps 1500 (vs 3500),
# best_ckpt_start_epoch 150 (vs 300). Stochastic sampler stays on for diversity.
_CFG = {
    "data": {"dequant_dist": "none", "int_dequant_factor": 0},
    "unimodmlp_params": {
        "num_layers": 2, "d_token": 4, "n_head": 1,
        "factor": 32, "bias": True, "use_mlp": True, "dim_t": 256,
    },
    "diffusion_params": {
        "num_timesteps": 100,
        "scheduler": "power_mean",
        "cat_scheduler": "log_linear",
        "noise_dist": "uniform_t",
        "sampler_params": {"stochastic_sampler": True, "second_order_correction": True},
        "edm_params": {"precond": True, "sigma_data": 1.0, "net_conditioning": "sigma"},
        "noise_dist_params": {"P_mean": -1.2, "P_std": 1.2},
        "noise_schedule_params": {
            "sigma_min": 0.002, "sigma_max": 80, "rho": 7,
            "eps_max": 1e-3, "eps_min": 1e-5,
            "rho_init": 7.0, "rho_offset": 5.0, "k_init": -6.0, "k_offset": 1.0,
        },
    },
    "train": {
        "main": {
            "lr": 0.001, "ema_decay": 0.997,
            "lr_scheduler": "reduce_lr_on_plateau", "factor": 0.90,
            "closs_weight_schedule": "anneal", "c_lambda": 1.0, "d_lambda": 1.0,
            "batch_size": 256, "steps": 1500, "weight_decay": 1e-3,
            "check_val_every": 250, "reduce_lr_patience": 25,
            "best_ckpt_start_epoch": 150, "early_stop_patience": 0,
            "val_sample_size": 5000,
        }
    },
    "sample": {"batch_size": 4096},
}


def register(name, csv_path):
    data_dir = f"data/{name}"
    os.makedirs(data_dir, exist_ok=True)
    dst = f"{data_dir}/{name}.csv"
    shutil.copyfile(csv_path, dst)
    info = {
        "name": name,
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": NUM_COL_IDX,
        "cat_col_idx": CAT_COL_IDX,
        "target_col_idx": TARGET_COL_IDX,
        "file_type": "csv",
        "data_path": dst,
        "val_path": None,
        "test_path": None,
    }
    with open(f"{INFO_DIR}/{name}.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"\n=== Registering {name} ===", flush=True)
    subprocess.run(["python", "process_dataset.py", "--dataname", name], check=True)


def main():
    os.makedirs(INFO_DIR, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
    src.dump_config(copy.deepcopy(_CFG), cfg_path)
    print(f"[{name}] config -> {cfg_path}", flush=True)

    if os.path.exists(f"data/{name}/info.json"):
        print(f"[{name}] already registered — skipping data prep", flush=True)
        return

    df = pd.read_csv(SPLIT_CSV)
    assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET, \
        f"col {TARGET_COL_IDX[0]} is '{df.columns[TARGET_COL_IDX[0]]}', expected '{TARGET}'"
    assert len(df.columns) == N_COLS, f"expected {N_COLS} cols, got {len(df.columns)}"

    for i in INT_CAT_IDX + TARGET_COL_IDX:
        df[df.columns[i]] = df[df.columns[i]].astype(int).astype(str)

    pos = df[df[TARGET] == "1"].reset_index(drop=True)
    pos = pos.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[{name}] split {SPLIT_ID}: {len(df)} rows | positives={len(pos)} "
          f"(positive share {len(pos)/len(df):.3%})", flush=True)

    csv_path = f"{tmp_dir}/{name}.csv"
    pos.to_csv(csv_path, index=False)
    print(f"[{name}] {len(pos)} positive rows -> registering", flush=True)
    register(name, csv_path)
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    main()
