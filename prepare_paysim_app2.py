"""Approach 2 (majority-dosage sweep) preparation for paysim split 0.

From datasets/paysim/splits/0/train.csv, build the four majority-dosage
partitions and register them as TabDiff datasets:

  ps0_d1 — fraud only            (~6.6k rows,  ~100%  fraud share)
  ps0_d2 — fraud + 25% majority  (~1.28M rows, ~0.51% fraud share)
  ps0_d3 — fraud + 50% majority  (~2.55M rows, ~0.26% fraud share)
  ps0_d4 — fraud + 100% majority (~5.09M rows, ~0.13% fraud share)

(All fraud rows are in every partition; majority subsets drawn independently.)

Also writes one TOML config per model, scaled to its partition size.

Schema notes (paysim):
  - target column is `isFraud` (0/1, cast to str for clean categorical handling)
  - dropped columns: nameOrig (~5.08M unique) and nameDest (~2.27M unique) —
    near-unique account identifiers that would blow up categorical handling
    without adding learnable signal.
  - isFlaggedFraud kept as a categorical flag (cast to str).

Run once before run_paysim_app2.py.
"""
import os
import json
import copy
import shutil
import subprocess

import sys
import pandas as pd
import src

# Split index from argv (default 0). Datasets/configs are named ps<SPLIT_ID>_d{1..4}
# so different splits never collide.
SPLIT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT_CSV = f"/home/amad/projects/datasets/paysim/splits/{SPLIT_ID}/train.csv"
DROP_COLS = ["nameOrig", "nameDest"]
TARGET    = "isFraud"
INFO_DIR  = "data/Info"

# Column schema AFTER dropping DROP_COLS (9 columns, indices 0..8):
#  0 step             num
#  1 type             cat
#  2 amount           num
#  3 oldbalanceOrg    num
#  4 newbalanceOrig   num
#  5 oldbalanceDest   num
#  6 newbalanceDest   num
#  7 isFraud          target
#  8 isFlaggedFraud   cat
N_COLS         = 9
NUM_COL_IDX    = [0, 2, 3, 4, 5, 6]
TARGET_COL_IDX = [7]
CAT_COL_IDX    = [1, 8]

_BASE = {
    "data": {"dequant_dist": "none", "int_dequant_factor": 0},
    "unimodmlp_params": {
        "num_layers": 2, "d_token": 4, "n_head": 1,
        "factor": 32, "bias": True, "use_mlp": True,
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
        }
    },
    "sample": {"batch_size": 4096},
}

# Per-dose overrides. IMPORTANT: `steps` is the number of EPOCHS, and each epoch is
# a full pass over ALL batches. For million-row partitions a few-hundred passes is
# already plenty; the small fraud-only set needs many more passes but strict early
# stopping to avoid memorization. We also cap in-training validation sampling
# (val_sample_size) because the default samples real_data_size rows (millions) per
# validation, which would otherwise dominate wall-clock for d2/d3/d4.
#
#   steps                 = max epochs (early stopping usually halts sooner)
#   best_ckpt_start_epoch = earliest epoch best_ema may be saved (so early stop works)
#   early_stop_patience   = stop after N epochs w/o held-out EMA-loss improvement
#   val_sample_size       = #rows generated during each in-training evaluation
# Batch sizes: the model is a tiny MLP, so even batch 16k uses ~2GB of the T4's
# 15GB (memory is NOT the constraint). Batches are therefore sized for GPU
# *throughput* while keeping >=~100 gradient updates/epoch, NOT to fill VRAM
# (huge batches = too few updates + worse generalization). d1 is kept small on
# purpose: small-batch SGD noise regularizes the memorization-prone fraud-only
# model. Max epochs are generous upper bounds — early stopping is the real
# terminator, so a high cap costs nothing when the model converges sooner.
DOSE_OVERRIDES = {
    # Early stopping DISABLED (early_stop_patience=0): every run completes all epochs
    # and the best checkpoint (min held-out EMA loss, tracked from best_ckpt_start_epoch
    # onward across the FULL run) is selected for generation.
    # fraud-only (~5.9k train): memorization-prone -> small batch (regularization), strong WD
    "d1": {"dim_t": 256,  "batch_size": 256,   "steps": 3500, "weight_decay": 1e-4, "check_val_every": 500, "reduce_lr_patience": 25,
           "best_ckpt_start_epoch": 300, "early_stop_patience": 0, "val_sample_size": 5000},
    # Restore full epochs (quality) and cut generation targets to fit ~6h/GPU.
    # d2/d3 get their prior epoch counts; d4 capped at 130 (500 ep = ~13h, can't fit 6h).
    # +25% majority (~1.15M train, ~14.5s/epoch): 800 ep ≈ 3.2h
    "d2": {"dim_t": 512,  "batch_size": 4096, "steps": 800,  "weight_decay": 1e-4, "check_val_every": 50,  "reduce_lr_patience": 20,
           "best_ckpt_start_epoch": 40,  "early_stop_patience": 0,  "val_sample_size": 50000},
    # +50% majority (~2.29M train, ~27.5s/epoch): 500 ep ≈ 3.8h
    "d3": {"dim_t": 512,  "batch_size": 4096, "steps": 500,  "weight_decay": 1e-5, "check_val_every": 50,  "reduce_lr_patience": 20,
           "best_ckpt_start_epoch": 40,  "early_stop_patience": 0,  "val_sample_size": 50000},
    # full data (~4.58M train, ~96s/epoch): 500 ep ≈ 13h (epoch cap removed — train to convergence)
    "d4": {"dim_t": 1024, "batch_size": 8192, "steps": 500,  "weight_decay": 0,    "check_val_every": 30,  "reduce_lr_patience": 15,
           "best_ckpt_start_epoch": 20,  "early_stop_patience": 0,  "val_sample_size": 50000},
}


def build_config(dose):
    cfg = copy.deepcopy(_BASE)
    ov  = DOSE_OVERRIDES[dose]
    cfg["unimodmlp_params"]["dim_t"]              = ov["dim_t"]
    cfg["train"]["main"]["batch_size"]            = ov["batch_size"]
    cfg["train"]["main"]["steps"]                 = ov["steps"]
    cfg["train"]["main"]["weight_decay"]          = ov["weight_decay"]
    cfg["train"]["main"]["check_val_every"]       = ov["check_val_every"]
    cfg["train"]["main"]["reduce_lr_patience"]    = ov["reduce_lr_patience"]
    cfg["train"]["main"]["best_ckpt_start_epoch"] = ov["best_ckpt_start_epoch"]
    cfg["train"]["main"]["early_stop_patience"]   = ov["early_stop_patience"]
    cfg["train"]["main"]["val_sample_size"]       = ov["val_sample_size"]
    return cfg


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
    tmp_dir = "data/paysim_app2_partitions"
    os.makedirs(tmp_dir, exist_ok=True)

    s = SPLIT_ID
    df = pd.read_csv(SPLIT_CSV).drop(columns=DROP_COLS)
    assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET, \
        f"col {TARGET_COL_IDX[0]} is '{df.columns[TARGET_COL_IDX[0]]}', expected '{TARGET}'"
    assert len(df.columns) == N_COLS, f"expected {N_COLS} cols, got {len(df.columns)}"

    # Cast the int-valued categorical/target columns (isFraud idx7, isFlaggedFraud idx8)
    # to str for clean categorical handling (avoids int<->float round-trip drift through
    # the numpy concat in process_dataset). `type` (idx1) is already a string — leave it.
    for i in [8] + TARGET_COL_IDX:
        df[df.columns[i]] = df[df.columns[i]].astype(int).astype(str)

    fraud    = df[df[TARGET] == "1"].reset_index(drop=True)
    nonfraud = df[df[TARGET] == "0"].reset_index(drop=True)
    n_nf     = len(nonfraud)
    print(f"\n##### Split {s}: {len(df)} rows | fraud={len(fraud)} non-fraud={n_nf} "
          f"(fraud share {len(fraud)/len(df):.3%})")

    doses = [
        ("d1", fraud),
        ("d2", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.25), random_state=100 + s)], ignore_index=True)),
        ("d3", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.50), random_state=200 + s)], ignore_index=True)),
        ("d4", df),
    ]

    for dose, partition in doses:
        name = f"ps{s}_{dose}"
        cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
        src.dump_config(build_config(dose), cfg_path)
        print(f"[{name}] config -> {cfg_path}")

        if os.path.exists(f"data/{name}/info.json"):
            print(f"[{name}] already registered — skipping data prep")
            continue

        partition = partition.sample(frac=1, random_state=42).reset_index(drop=True)
        csv_path  = f"{tmp_dir}/{name}.csv"
        partition.to_csv(csv_path, index=False)
        fraud_n   = (partition[TARGET] == "1").sum()
        print(f"[{name}] {fraud_n} fraud + {len(partition)-fraud_n} non-fraud "
              f"= {len(partition)} rows | fraud share={fraud_n/len(partition):.3%}")
        register(name, csv_path)

    print("\nPreparation complete. Run run_paysim_app2.py to train across all GPUs.")


if __name__ == "__main__":
    main()
