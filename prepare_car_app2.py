"""Build 4 majority-dosage partitions of the car_fraud dataset for Approach 2.

Cleans the raw car_baseline_train.csv, then mirrors prepare_expv01_app2.py:
each partition keeps all fraud rows plus a growing DOSE of the non-fraud
majority, and gets a TOML config sized to the partition.

Cleaning (raw 24 cols -> 22 cols):
  - drop policy_id      (24k-unique identifier)
  - drop incident_city  (15k-unique, identifier-like)
  - incident_date       -> integer days since the earliest date (numeric)
  - authorities_contacted NaN -> "None"
  - fraud_reported Y/N  -> 1/0  (so the generator can filter target == 1)

Run once before run_car_app2.sh.
"""
import os
import json
import shutil
import subprocess

import pandas as pd
import src

SRC       = "data/car_fraud/car_baseline_train.csv"
DROP_COLS = ["policy_id", "incident_city"]
TARGET    = "fraud_reported"
INFO_DIR  = "data/Info"

# schema AFTER cleaning (22 cols) — see module docstring for the column order
N_COLS         = 22
NUM_COL_IDX    = [1, 2, 3, 8, 14, 15, 16, 17, 19, 20]
TARGET_COL_IDX = [21]
CAT_COL_IDX    = [i for i in range(N_COLS) if i not in NUM_COL_IDX + TARGET_COL_IDX]

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

# partitions are ~2x exp-v01 sizes (2752 fraud, 21248 non-fraud)
MODEL_OVERRIDES = {
    "car_d1": {"dim_t": 512,  "batch_size": 256,  "steps": 2000, "weight_decay": 1e-4, "check_val_every": 200,  "reduce_lr_patience": 20},
    "car_d2": {"dim_t": 512,  "batch_size": 512,  "steps": 3000, "weight_decay": 1e-5, "check_val_every": 300,  "reduce_lr_patience": 30},
    "car_d3": {"dim_t": 1024, "batch_size": 1024, "steps": 4000, "weight_decay": 1e-5, "check_val_every": 400,  "reduce_lr_patience": 40},
    "car_d4": {"dim_t": 1024, "batch_size": 2048, "steps": 6000, "weight_decay": 0,    "check_val_every": 2000, "reduce_lr_patience": 50},
}


def build_config(name):
    import copy
    cfg = copy.deepcopy(_BASE)
    ov  = MODEL_OVERRIDES[name]
    cfg["unimodmlp_params"]["dim_t"]           = ov["dim_t"]
    cfg["train"]["main"]["batch_size"]         = ov["batch_size"]
    cfg["train"]["main"]["steps"]              = ov["steps"]
    cfg["train"]["main"]["weight_decay"]       = ov["weight_decay"]
    cfg["train"]["main"]["check_val_every"]    = ov["check_val_every"]
    cfg["train"]["main"]["reduce_lr_patience"] = ov["reduce_lr_patience"]
    return cfg


def clean(df):
    df = df.drop(columns=DROP_COLS)
    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    df["incident_date"] = (df["incident_date"] - df["incident_date"].min()).dt.days
    # NB: avoid "None"/"NA"/etc — pandas read_csv parses those back to NaN
    df["authorities_contacted"] = df["authorities_contacted"].fillna("NotContacted")
    df[TARGET] = (df[TARGET].astype(str).str.upper() == "Y").astype(int)
    return df.reset_index(drop=True)


def register(name, csv_path):
    data_dir = f"data/{name}"
    os.makedirs(data_dir, exist_ok=True)
    dst = f"{data_dir}/{name}.csv"
    shutil.copyfile(csv_path, dst)
    info = {
        "name": name, "task_type": "binclass", "header": "infer",
        "column_names": None,
        "num_col_idx": NUM_COL_IDX, "cat_col_idx": CAT_COL_IDX, "target_col_idx": TARGET_COL_IDX,
        "file_type": "csv", "data_path": dst, "val_path": None, "test_path": None,
    }
    with open(f"{INFO_DIR}/{name}.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"\n=== Registering {name} ===", flush=True)
    subprocess.run(["python", "process_dataset.py", "--dataname", name], check=True)


def main():
    df = clean(pd.read_csv(SRC))
    assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET, df.columns.tolist()
    assert len(df.columns) == N_COLS, len(df.columns)

    fraud    = df[df[TARGET] == 1].reset_index(drop=True)
    nonfraud = df[df[TARGET] == 0].reset_index(drop=True)
    n_nf     = len(nonfraud)
    print(f"Source: {len(df)} rows | fraud={len(fraud)} non-fraud={n_nf} | "
          f"fraud rate={len(fraud)/len(df):.1%}")

    os.makedirs(INFO_DIR, exist_ok=True)
    tmp_dir = "data/car_app2_partitions"
    os.makedirs(tmp_dir, exist_ok=True)

    doses = [
        ("car_d1", fraud),
        ("car_d2", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.25), random_state=2)], ignore_index=True)),
        ("car_d3", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.50), random_state=3)], ignore_index=True)),
        ("car_d4", df),
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
        fraud_n = int((partition[TARGET] == 1).sum())
        print(f"[{name}] {fraud_n} fraud + {len(partition)-fraud_n} non-fraud = {len(partition)} rows "
              f"| fraud share={fraud_n/len(partition):.1%}")
        register(name, csv_path)

    print("\nPreparation complete. Run run_car_app2.sh to train and generate.")


if __name__ == "__main__":
    main()
