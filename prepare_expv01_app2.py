"""Build 4 majority-dosage partitions of the exp-v01 dataset for Approach 2.

Each partition contains all ~738 fraud rows plus a different DOSE of the
non-fraud majority:
  expv01_d1 — fraud only            (~738 rows,  ~100% fraud yield)
  expv01_d2 — fraud + 25% majority  (~3638 rows, ~20%  fraud yield)
  expv01_d3 — fraud + 50% majority  (~6537 rows, ~11%  fraud yield)
  expv01_d4 — fraud + 100% majority (~12336 rows, ~6%  fraud yield)

Also writes one TOML config per model, each sized to its partition.

Run once before run_expv01_app2.sh.
"""
import os
import json
import shutil
import subprocess

import pandas as pd
import src

SRC       = "data/exp-v01/experiment2_real_train.csv"
DROP_COLS = ["PolicyNumber"]
TARGET    = "FraudFound_P"
INFO_DIR  = "data/Info"

N_COLS         = 32
NUM_COL_IDX    = [1, 7, 10, 16, 17, 18, 30]
TARGET_COL_IDX = [15]
CAT_COL_IDX    = [i for i in range(N_COLS) if i not in NUM_COL_IDX + TARGET_COL_IDX]

# Per-model configs scaled to each partition size.
# Key changes vs. default (batch=2048, dim_t=1024, steps=5500, weight_decay=0):
#   d1 (~738 rows):   very small — max regularisation, shortest run
#   d2 (~3638 rows):  small
#   d3 (~6537 rows):  medium
#   d4 (~12336 rows): full dataset — stock defaults, just raise num_timesteps
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

MODEL_OVERRIDES = {
    "expv01_d1": {"dim_t": 256,  "batch_size": 128,  "steps": 1500, "weight_decay": 1e-4,  "check_val_every": 150,  "reduce_lr_patience": 15},
    "expv01_d2": {"dim_t": 512,  "batch_size": 256,  "steps": 2500, "weight_decay": 1e-4,  "check_val_every": 250,  "reduce_lr_patience": 25},
    "expv01_d3": {"dim_t": 512,  "batch_size": 512,  "steps": 3500, "weight_decay": 1e-5,  "check_val_every": 350,  "reduce_lr_patience": 35},
    "expv01_d4": {"dim_t": 1024, "batch_size": 2048, "steps": 5500, "weight_decay": 0,     "check_val_every": 2000, "reduce_lr_patience": 50},
}


def build_config(name):
    import copy
    cfg = copy.deepcopy(_BASE)
    ov  = MODEL_OVERRIDES[name]
    cfg["unimodmlp_params"]["dim_t"] = ov["dim_t"]
    cfg["train"]["main"]["batch_size"]         = ov["batch_size"]
    cfg["train"]["main"]["steps"]              = ov["steps"]
    cfg["train"]["main"]["weight_decay"]       = ov["weight_decay"]
    cfg["train"]["main"]["check_val_every"]    = ov["check_val_every"]
    cfg["train"]["main"]["reduce_lr_patience"] = ov["reduce_lr_patience"]
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
    df       = pd.read_csv(SRC).drop(columns=DROP_COLS)
    fraud    = df[df[TARGET] == 1].reset_index(drop=True)
    nonfraud = df[df[TARGET] == 0].reset_index(drop=True)
    n_nf     = len(nonfraud)

    assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET
    assert len(df.columns) == N_COLS

    print(f"Source: {len(df)} rows | fraud={len(fraud)} non-fraud={n_nf}")

    os.makedirs(INFO_DIR, exist_ok=True)
    tmp_dir = "data/expv01_app2_partitions"
    os.makedirs(tmp_dir, exist_ok=True)

    doses = [
        ("expv01_d1", fraud),
        ("expv01_d2", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.25), random_state=2)], ignore_index=True)),
        ("expv01_d3", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.50), random_state=3)], ignore_index=True)),
        ("expv01_d4", df),
    ]

    for name, partition in doses:
        cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
        src.dump_config(build_config(name), cfg_path)
        print(f"[{name}] config written -> {cfg_path}")

        if os.path.exists(f"data/{name}/info.json"):
            print(f"[{name}] already registered — skipping")
            continue

        partition = partition.sample(frac=1, random_state=42).reset_index(drop=True)
        csv_path  = f"{tmp_dir}/{name}.csv"
        partition.to_csv(csv_path, index=False)
        fraud_n  = (partition[TARGET] == 1).sum()
        print(f"[{name}] {fraud_n} fraud + {len(partition)-fraud_n} non-fraud = {len(partition)} rows | fraud share={fraud_n/len(partition):.1%}")
        register(name, csv_path)

    print("\nPreparation complete. Run run_expv01_app2.sh to train and generate.")


if __name__ == "__main__":
    main()
