"""Prepare a positive-only (SeriousDlqin2yrs==1) GMSC dataset + config for one split.

Mirrors prepare_d1.py / prepare_paysim_app2.py (fraud-only pattern), adapted to the
GMSC ("Give Me Some Credit") schema:

  - target SeriousDlqin2yrs at column 0 (kept as a categorical, constant "1")
  - the remaining 10 columns are all numerical (cat_col_idx is empty)

Unlike paysim, GMSC has missing values in MonthlyIncome / NumberOfDependents, so we
median-impute the numerical columns BEFORE registering (otherwise process_dataset.py
hits a pdb breakpoint on residual NaNs).

  python prepare_gmsc.py <SPLIT_ID>      # SPLIT_ID in 0..4  ->  dataset gmsc_l<SPLIT_ID>

The held-out test.csv is never read.
"""
import os
import sys
import json
import shutil
import subprocess

import pandas as pd
import src
from prepare_paysim_app2 import build_config  # reuses _BASE + DOSE_OVERRIDES["d1"]

SPLIT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT_CSV = f"/srv/datasets/GMSC/splits/level_{SPLIT_ID}/train.csv"
name      = f"gmsc_l{SPLIT_ID}"
INFO_DIR  = "data/Info"
tmp_dir   = "data/gmsc_partitions"

# GMSC schema (11 columns, target at index 0; columns 1..10 all numerical).
TARGET         = "SeriousDlqin2yrs"
N_COLS         = 11
TARGET_COL_IDX = [0]
NUM_COL_IDX    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CAT_COL_IDX    = []


def register(name, csv_path):
    """Self-contained copy of prepare_paysim_app2.register using the GMSC schema."""
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
    src.dump_config(build_config("d1"), cfg_path)
    print(f"[{name}] config -> {cfg_path}", flush=True)

    if os.path.exists(f"data/{name}/info.json"):
        print(f"[{name}] already registered — skipping data prep", flush=True)
        return

    df = pd.read_csv(SPLIT_CSV)
    assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET, \
        f"col 0 is '{df.columns[0]}', expected '{TARGET}'"
    assert len(df.columns) == N_COLS, f"expected {N_COLS} cols, got {len(df.columns)}"

    # Positive-only subset.
    pos = df[df[TARGET] == 1].reset_index(drop=True)

    # Median-impute numerical columns (GMSC has NaNs in MonthlyIncome / NumberOfDependents);
    # process_dataset.py refuses residual NaNs in numerical columns.
    num_cols = [df.columns[i] for i in NUM_COL_IDX]
    medians = pos[num_cols].median()
    n_na = int(pos[num_cols].isna().sum().sum())
    pos[num_cols] = pos[num_cols].fillna(medians)
    print(f"[{name}] imputed {n_na} missing numerical values with split-positive medians", flush=True)

    # Target -> str for clean single-category handling.
    pos[TARGET] = pos[TARGET].astype(int).astype(str)

    pos = pos.sample(frac=1, random_state=42).reset_index(drop=True)
    csv_path = f"{tmp_dir}/{name}.csv"
    pos.to_csv(csv_path, index=False)
    print(f"[{name}] {len(pos)} positive rows -> registering (5x target = {5*len(pos)})", flush=True)
    register(name, csv_path)
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    main()
