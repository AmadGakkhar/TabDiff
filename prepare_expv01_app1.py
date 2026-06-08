"""Build M=4 balanced partitions of the exp-v01 dataset for Approach 1
(balanced-bagging ensemble of TabDiff generators).

Each partition: all ~738 fraud rows  +  a different random undersample of
~738 non-fraud rows  ->  ~50% fraud share  ->  ~50% yield at generation time.

Also writes a single shared TabDiff config tuned for the ~1476-row member size.

Run once before run_expv01_app1.sh.
"""
import os
import json
import shutil
import subprocess

import pandas as pd
import src

SRC        = "data/exp-v01/experiment2_real_train.csv"
DROP_COLS  = ["PolicyNumber"]   # row identifier, not a feature
TARGET     = "FraudFound_P"
M          = 4
INFO_DIR   = "data/Info"
CONFIG_OUT = "tabdiff/configs/tabdiff_configs_expv01_app1.toml"

# Schema after dropping PolicyNumber — identical to the registered 'fraud' dataset.
N_COLS         = 32
NUM_COL_IDX    = [1, 7, 10, 16, 17, 18, 30]   # WeekOfMonth, WeekOfMonthClaimed, Age, RepNumber, Deductible, DriverRating, Year
TARGET_COL_IDX = [15]                           # FraudFound_P
CAT_COL_IDX    = [i for i in range(N_COLS) if i not in NUM_COL_IDX + TARGET_COL_IDX]

# Config tuned for ~1476-row balanced members (vs. the default tuned for ~12k rows).
MEMBER_CONFIG = {
    "data": {"dequant_dist": "none", "int_dequant_factor": 0},
    "unimodmlp_params": {
        "num_layers": 2, "d_token": 4, "n_head": 1,
        "factor": 32, "bias": True,
        "dim_t": 512,       # reduced from 1024 — right-sized for ~1.5k rows
        "use_mlp": True,
    },
    "diffusion_params": {
        "num_timesteps": 100,           # increased from 50 — better per-sample fidelity
        "scheduler": "power_mean",
        "cat_scheduler": "log_linear",
        "noise_dist": "uniform_t",
        "sampler_params": {
            "stochastic_sampler": True,         # diversity; fights mode collapse on small sets
            "second_order_correction": True,
        },
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
            "steps": 2000,              # shorter than 5500 default — avoid memorising ~1.5k rows
            "lr": 0.001,
            "weight_decay": 1e-4,       # regularisation (default is 0)
            "ema_decay": 0.997,
            "batch_size": 256,          # well below subset size; default 2048 would give ~6 batches/epoch
            "check_val_every": 200,     # frequent enough for early-stop visibility
            "lr_scheduler": "reduce_lr_on_plateau",
            "factor": 0.90,
            "reduce_lr_patience": 20,   # reduced from 50 to match shorter run
            "closs_weight_schedule": "anneal",
            "c_lambda": 1.0,
            "d_lambda": 1.0,
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
    df = pd.read_csv(SRC).drop(columns=DROP_COLS)
    assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET, \
        f"Column at index {TARGET_COL_IDX[0]} is '{df.columns[TARGET_COL_IDX[0]]}', expected '{TARGET}'"
    assert len(df.columns) == N_COLS, \
        f"Expected {N_COLS} columns after dropping PolicyNumber, got {len(df.columns)}"

    fraud    = df[df[TARGET] == 1].reset_index(drop=True)
    nonfraud = df[df[TARGET] == 0].reset_index(drop=True)
    n_maj    = len(fraud)   # strict 1:1 balance — maximises fraud-mode capacity per member

    print(f"Source: {len(df)} rows | fraud={len(fraud)} non-fraud={len(nonfraud)}")
    print(f"Each member: {len(fraud)} fraud + {n_maj} non-fraud = {len(fraud)+n_maj} rows (~50% fraud)")

    os.makedirs(INFO_DIR, exist_ok=True)

    tmp_dir = "data/expv01_app1_partitions"
    os.makedirs(tmp_dir, exist_ok=True)

    for m in range(1, M + 1):
        name = f"expv01_m{m}"
        if os.path.exists(f"data/{name}/info.json"):
            print(f"[{name}] already registered — skipping")
            continue

        maj_sample = nonfraud.sample(n=n_maj, random_state=m)   # different seed per member
        partition  = pd.concat([fraud, maj_sample], ignore_index=True)
        partition  = partition.sample(frac=1, random_state=m).reset_index(drop=True)

        csv_path = f"{tmp_dir}/{name}.csv"
        partition.to_csv(csv_path, index=False)
        print(f"[{name}] {len(fraud)} fraud + {len(maj_sample)} non-fraud = {len(partition)} rows -> {csv_path}")
        register(name, csv_path)

    src.dump_config(MEMBER_CONFIG, CONFIG_OUT)
    print(f"\nMember config written -> {CONFIG_OUT}")
    print("\nPreparation complete. Run run_expv01_app1.sh to train and generate.")


if __name__ == "__main__":
    main()
