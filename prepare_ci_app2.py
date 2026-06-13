"""Approach 2 (majority-dosage sweep) preparation for the car-insurance splits.

For EACH split (0..4) under datasets/car-insurance/splits/<s>/train.csv, build the
four majority-dosage partitions and register them as TabDiff datasets:

  ci_s<S>_d1 — fraud only             (~2750 rows,  ~100% fraud yield)
  ci_s<S>_d2 — fraud + 25% majority   (~8062 rows,  ~34%  fraud yield)
  ci_s<S>_d3 — fraud + 50% majority   (~13375 rows, ~21%  fraud yield)
  ci_s<S>_d4 — fraud + 100% majority  (~24000 rows, ~11.5% fraud yield)

(All fraud rows are in every partition; majority subsets are drawn independently.)

Also writes one TOML config per model, scaled to its partition size.

Schema notes (this dataset differs from exp-v01):
  - target column is `fraud_reported` (values Y/N)
  - dropped columns: policy_id (row identifier), incident_city (~15k unique,
    identifier-like free text), incident_date (731 unique raw date strings).
    These are near-unique / high-cardinality and would blow up TabDiff's
    categorical handling without adding learnable signal.

Run once before run_ci_app2.py.
"""
import os
import json
import copy
import shutil
import subprocess

import pandas as pd
import src

SPLITS_DIR = "/home/amad/projects/datasets/car-insurance/splits"
N_SPLITS   = 5
DROP_COLS  = ["policy_id", "incident_city", "incident_date"]
TARGET     = "fraud_reported"
INFO_DIR   = "data/Info"

# Column schema AFTER dropping DROP_COLS (21 columns, indices 0..20).
#  0 policy_state               cat
#  1 policy_deductible          num
#  2 policy_annual_premium      num
#  3 insured_age                num
#  4 insured_sex                cat
#  5 insured_education_level    cat
#  6 insured_occupation         cat
#  7 insured_hobbies            cat
#  8 incident_type              cat
#  9 collision_type             cat
# 10 incident_severity          cat
# 11 authorities_contacted      cat
# 12 incident_state             cat
# 13 incident_hour_of_the_day   num
# 14 number_of_vehicles_involved num
# 15 bodily_injuries            num
# 16 witnesses                  num
# 17 police_report_available    cat
# 18 claim_amount               num
# 19 total_claim_amount         num
# 20 fraud_reported             target
N_COLS         = 21
NUM_COL_IDX    = [1, 2, 3, 13, 14, 15, 16, 18, 19]
TARGET_COL_IDX = [20]
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

# Per-dose overrides, scaled to the larger partition sizes of this dataset.
DOSE_OVERRIDES = {
    "d1": {"dim_t": 256,  "batch_size": 256,  "steps": 3000, "weight_decay": 1e-4, "check_val_every": 300,  "reduce_lr_patience": 20},
    "d2": {"dim_t": 512,  "batch_size": 512,  "steps": 4000, "weight_decay": 1e-4, "check_val_every": 400,  "reduce_lr_patience": 25},
    "d3": {"dim_t": 512,  "batch_size": 1024, "steps": 5000, "weight_decay": 1e-5, "check_val_every": 500,  "reduce_lr_patience": 30},
    "d4": {"dim_t": 1024, "batch_size": 2048, "steps": 6000, "weight_decay": 0,    "check_val_every": 1000, "reduce_lr_patience": 40},
}


def build_config(dose):
    cfg = copy.deepcopy(_BASE)
    ov  = DOSE_OVERRIDES[dose]
    cfg["unimodmlp_params"]["dim_t"]           = ov["dim_t"]
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
    os.makedirs(INFO_DIR, exist_ok=True)
    tmp_dir = "data/ci_app2_partitions"
    os.makedirs(tmp_dir, exist_ok=True)

    for s in range(N_SPLITS):
        src_csv = f"{SPLITS_DIR}/{s}/train.csv"
        df = pd.read_csv(src_csv).drop(columns=DROP_COLS)
        assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET, \
            f"split {s}: col {TARGET_COL_IDX[0]} is '{df.columns[TARGET_COL_IDX[0]]}', expected '{TARGET}'"
        assert len(df.columns) == N_COLS, f"split {s}: expected {N_COLS} cols, got {len(df.columns)}"

        # Categorical missing values -> the 'nan' sentinel the TabDiff pipeline expects
        # (e.g. authorities_contacted has genuine missings). Numerical cols have no NaNs.
        # Use a literal sentinel (not "nan", which round-trips to NaN through CSV).
        cat_cols = [df.columns[i] for i in CAT_COL_IDX]
        df[cat_cols] = df[cat_cols].fillna("Missing")

        fraud    = df[df[TARGET] == "Y"].reset_index(drop=True)
        nonfraud = df[df[TARGET] == "N"].reset_index(drop=True)
        n_nf     = len(nonfraud)
        print(f"\n##### Split {s}: {len(df)} rows | fraud={len(fraud)} non-fraud={n_nf} "
              f"(fraud share {len(fraud)/len(df):.1%})")

        doses = [
            ("d1", fraud),
            ("d2", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.25), random_state=100 + s)], ignore_index=True)),
            ("d3", pd.concat([fraud, nonfraud.sample(n=int(n_nf * 0.50), random_state=200 + s)], ignore_index=True)),
            ("d4", df),
        ]

        for dose, partition in doses:
            name = f"ci_s{s}_{dose}"
            cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
            src.dump_config(build_config(dose), cfg_path)
            print(f"[{name}] config -> {cfg_path}")

            if os.path.exists(f"data/{name}/info.json"):
                print(f"[{name}] already registered — skipping data prep")
                continue

            partition = partition.sample(frac=1, random_state=42).reset_index(drop=True)
            csv_path  = f"{tmp_dir}/{name}.csv"
            partition.to_csv(csv_path, index=False)
            fraud_n   = (partition[TARGET] == "Y").sum()
            print(f"[{name}] {fraud_n} fraud + {len(partition)-fraud_n} non-fraud "
                  f"= {len(partition)} rows | fraud share={fraud_n/len(partition):.1%}")
            register(name, csv_path)

    print("\nPreparation complete. Run run_ci_app2.py to train across all GPUs.")


if __name__ == "__main__":
    main()
