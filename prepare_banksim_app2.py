"""Approach 2b preparation for banksim temporal splits — d1 (fraud-only) ONLY.

For each split S under datasets/banksim/splits_temporal/<S>/train.csv we build the
fraud-only partition and register it as a TabDiff dataset `bs<S>_d1`.

The split's test.csv is NEVER read by any part of this pipeline. To get a true
held-out overfitting / early-stop signal we carve 20% of the TRAIN fraud rows into
a held-out set and register it as the dataset's `test_path` (TabDiff's trainer uses
the processed test split as its held-out EMA-loss signal). The real test.csv stays
untouched.

Schema notes (banksim raw columns):
    step, customer, age, gender, zipcodeOri, merchant, zipMerchant, category, amount, fraud
  dropped:
    customer    (~4k unique account id — no learnable signal, blows up embedding)
    zipcodeOri  (constant, 1 unique)
    zipMerchant (constant, 1 unique)
  target: `fraud` (0/1, cast to str categorical)

Remaining 7 columns (indices after drop):
    0 step     num
    1 age      cat
    2 gender   cat
    3 merchant cat
    4 category cat
    5 amount   num
    6 fraud    target

Run once before run_banksim_app2.py.
"""
import os
import json
import copy
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import src

SPLITS_DIR = "/home/amad/projects/datasets/banksim/splits_temporal"
DROP_COLS  = ["customer", "zipcodeOri", "zipMerchant"]
TARGET     = "fraud"
INFO_DIR   = "data/Info"
HELDOUT_FRAC = 0.20
SEED         = 42

# Column schema AFTER dropping DROP_COLS (7 cols, indices 0..6):
N_COLS         = 7
NUM_COL_IDX    = [0, 5]
CAT_COL_IDX    = [1, 2, 3, 4]
TARGET_COL_IDX = [6]

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

# d1 (fraud-only) hyperparameters. banksim fraud counts per split (1.5k-6.2k) bracket
# paysim's d1 (6.6k), so the same memorization-aware recipe applies: small batch
# (SGD-noise regularization) + strong weight decay + EMA + a generous epoch cap that
# early stopping on the held-out 20% terminates well before. Now that we have a real
# held-out signal we ENABLE early stopping (patience in epochs).
D1_OVERRIDE = {
    "dim_t": 256, "batch_size": 256, "steps": 3500, "weight_decay": 1e-4,
    "check_val_every": 500, "reduce_lr_patience": 25,
    "best_ckpt_start_epoch": 200, "early_stop_patience": 300, "val_sample_size": 5000,
}


def build_config():
    cfg = copy.deepcopy(_BASE)
    ov  = D1_OVERRIDE
    cfg["unimodmlp_params"]["dim_t"]              = ov["dim_t"]
    m = cfg["train"]["main"]
    m["batch_size"]            = ov["batch_size"]
    m["steps"]                 = ov["steps"]
    m["weight_decay"]          = ov["weight_decay"]
    m["check_val_every"]       = ov["check_val_every"]
    m["reduce_lr_patience"]    = ov["reduce_lr_patience"]
    m["best_ckpt_start_epoch"] = ov["best_ckpt_start_epoch"]
    m["early_stop_patience"]   = ov["early_stop_patience"]
    m["val_sample_size"]       = ov["val_sample_size"]
    return cfg


def heldout_split(fraud, cat_cols):
    """Carve HELDOUT_FRAC of fraud rows as held-out, GUARANTEEING every categorical
    value present in held-out also appears in train (else encoders fit on train fail
    on the test split). Move offending rows back to train."""
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(fraud))
    n_held = int(round(len(fraud) * HELDOUT_FRAC))
    held_idx = set(idx[:n_held].tolist())
    train_mask = np.array([i not in held_idx for i in range(len(fraud))])

    # Ensure category coverage: any held-out row with a cat value not in train moves back.
    for _ in range(50):
        train_df = fraud[train_mask]
        moved = False
        for c in cat_cols:
            train_vals = set(train_df[c].unique())
            held_rows = fraud[~train_mask]
            bad = held_rows[~held_rows[c].isin(train_vals)]
            if len(bad):
                train_mask[bad.index.to_numpy()] = True
                moved = True
        if not moved:
            break
    train_df = fraud[train_mask].reset_index(drop=True)
    held_df  = fraud[~train_mask].reset_index(drop=True)
    return train_df, held_df


def register(name, train_csv, test_csv):
    data_dir = f"data/{name}"
    os.makedirs(data_dir, exist_ok=True)
    train_dst = f"{data_dir}/{name}.csv"
    test_dst  = f"{data_dir}/{name}_heldout.csv"
    shutil.copyfile(train_csv, train_dst)
    shutil.copyfile(test_csv, test_dst)
    info = {
        "name": name,
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": NUM_COL_IDX,
        "cat_col_idx": CAT_COL_IDX,
        "target_col_idx": TARGET_COL_IDX,
        "file_type": "csv",
        "data_path": train_dst,
        "val_path": None,
        "test_path": test_dst,
    }
    with open(f"{INFO_DIR}/{name}.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"\n=== Registering {name} ===", flush=True)
    subprocess.run(["python", "process_dataset.py", "--dataname", name], check=True)


def main():
    os.makedirs(INFO_DIR, exist_ok=True)
    tmp_dir = "data/banksim_app2_partitions"
    os.makedirs(tmp_dir, exist_ok=True)

    with open(f"{SPLITS_DIR}/manifest.json") as f:
        n_splits = json.load(f)["n_splits"]

    splits = [int(sys.argv[1])] if len(sys.argv) > 1 else list(range(n_splits))

    for s in splits:
        name = f"bs{s}_d1"
        cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
        src.dump_config(build_config(), cfg_path)
        print(f"[{name}] config -> {cfg_path}")

        if os.path.exists(f"data/{name}/info.json"):
            print(f"[{name}] already registered — skipping data prep")
            continue

        df = pd.read_csv(f"{SPLITS_DIR}/{s}/train.csv").drop(columns=DROP_COLS)
        assert len(df.columns) == N_COLS, f"expected {N_COLS} cols, got {len(df.columns)}: {df.columns.tolist()}"
        assert df.columns[TARGET_COL_IDX[0]] == TARGET, \
            f"col {TARGET_COL_IDX[0]} is '{df.columns[TARGET_COL_IDX[0]]}', expected '{TARGET}'"

        # Cast all categorical + target columns to str for clean handling. `age` holds
        # 'U' (unknown) alongside ints so it cannot go through int; a direct str cast
        # works for every cat col and for the 0/1 target.
        for i in CAT_COL_IDX + TARGET_COL_IDX:
            df[df.columns[i]] = df[df.columns[i]].astype(str)

        # Optional log1p transform of the heavy-tailed `amount` column for splits whose
        # numerical head diverges under the raw-scale quantile transform (set via env
        # BANKSIM_LOG_AMOUNT="2,3"). Generation output must be expm1'd back — see
        # postprocess_log_amount in run/validate.
        log_splits = {int(x) for x in os.environ.get("BANKSIM_LOG_AMOUNT", "").split(",") if x.strip()}
        if s in log_splits:
            df["amount"] = np.log1p(df["amount"])
            print(f"[{name}] applied log1p(amount): range "
                  f"[{df['amount'].min():.3f}, {df['amount'].max():.3f}]")

        fraud = df[df[TARGET] == "1"].reset_index(drop=True)
        print(f"\n##### Split {s}: train.csv {len(df)} rows | fraud={len(fraud)} "
              f"(share {len(fraud)/len(df):.3%})  [test.csv NOT touched]")

        cat_cols = [df.columns[i] for i in CAT_COL_IDX]
        train_df, held_df = heldout_split(fraud, cat_cols)
        print(f"[{name}] fraud-only: {len(train_df)} train + {len(held_df)} held-out "
              f"({len(held_df)/len(fraud):.1%})")

        train_csv = f"{tmp_dir}/{name}_train.csv"
        held_csv  = f"{tmp_dir}/{name}_heldout.csv"
        train_df.to_csv(train_csv, index=False)
        held_df.to_csv(held_csv, index=False)
        register(name, train_csv, held_csv)

    print("\nPreparation complete. Run run_banksim_app2.py to train+generate.")


if __name__ == "__main__":
    main()
