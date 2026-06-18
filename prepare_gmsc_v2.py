"""Prepare v2 positive-only GMSC dataset for one split — improved fidelity.

Changes vs prepare_gmsc.py (v1), addressing the downstream quality report:

  1. MISSINGNESS PRESERVED. v1 median-imputed NaNs before training, so the model
     never saw missingness (biggest discriminator giveaway). v2 adds binary
     `MonthlyIncome_isna` / `NumberOfDependents_isna` CATEGORICAL columns and
     median-imputes the value, so the generator learns the JOINT distribution of
     (missing-flag, value, other features). gen_gmsc_v2.py restores NaN on output.
  2. INTEGER DEQUANT. dequant_dist="uniform", int_dequant_factor=1.0 so the model
     treats counts/age as smooth (uniform dequantization) and rounds on inverse.
  3. CAPACITY BUMP. dim_t 256 -> 512 to tighten the joint correlation structure.

Also writes a sidecar data/<name>/gmsc_clip.json with the real per-column [min,max]
(from the un-imputed positives) and the integer column list, used by gen to clip
tails and round integers.

  python prepare_gmsc_v2.py <SPLIT_ID>      # -> dataset gmsc_l<SPLIT_ID>_v2
"""
import os
import sys
import json
import shutil
import subprocess

import pandas as pd
import src
from prepare_paysim_app2 import build_config

SPLIT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT_CSV = f"/srv/datasets/GMSC/splits/level_{SPLIT_ID}/train.csv"
name      = f"gmsc_l{SPLIT_ID}_v2"
INFO_DIR  = "data/Info"
tmp_dir   = "data/gmsc_partitions"

TARGET   = "SeriousDlqin2yrs"
NA_COLS  = ["MonthlyIncome", "NumberOfDependents"]   # the only columns with missing values
# Base GMSC numerical feature names (cols 1..10), in original order.
NUM_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def build_gmsc_config():
    cfg = build_config("d1")                               # small-set tuning (batch 256, 3500 ep, wd 1e-4)
    cfg["unimodmlp_params"]["dim_t"] = 512                 # capacity bump for joint structure
    cfg["data"]["dequant_dist"] = "uniform"                # integer handling
    cfg["data"]["int_dequant_factor"] = 1.0
    return cfg


def register(name, csv_path, num_idx, cat_idx):
    data_dir = f"data/{name}"
    os.makedirs(data_dir, exist_ok=True)
    dst = f"{data_dir}/{name}.csv"
    shutil.copyfile(csv_path, dst)
    info = {
        "name": name, "task_type": "binclass", "header": "infer", "column_names": None,
        "num_col_idx": num_idx, "cat_col_idx": cat_idx, "target_col_idx": [0],
        "file_type": "csv", "data_path": dst, "val_path": None, "test_path": None,
    }
    with open(f"{INFO_DIR}/{name}.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"\n=== Registering {name} ===", flush=True)
    subprocess.run(["python", "process_dataset.py", "--dataname", name], check=True)


def main():
    os.makedirs(INFO_DIR, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
    src.dump_config(build_gmsc_config(), cfg_path)
    print(f"[{name}] config -> {cfg_path}", flush=True)

    df = pd.read_csv(SPLIT_CSV)
    assert df.columns.tolist()[0] == TARGET and len(df.columns) == 11
    pos = df[df[TARGET] == 1].reset_index(drop=True)

    # Real per-column [min,max] (un-imputed) for clipping; record integer columns.
    clip = {c: [float(pos[c].min()), float(pos[c].max())] for c in NUM_NAMES}
    int_cols = [c for c in NUM_NAMES if pos[c].dropna().mod(1).eq(0).all()]

    # Missingness indicators (categorical 0/1), then median-impute the value.
    for c in NA_COLS:
        pos[f"{c}_isna"] = pos[c].isna().astype(int).astype(str)
    pos[NUM_NAMES] = pos[NUM_NAMES].fillna(pos[NUM_NAMES].median())
    pos[TARGET] = pos[TARGET].astype(int).astype(str)

    # Column order: target(0), 10 numerical(1..10), 2 isna flags(11,12).
    isna_names = [f"{c}_isna" for c in NA_COLS]
    pos = pos[[TARGET] + NUM_NAMES + isna_names]
    num_idx = list(range(1, 11))
    cat_idx = [11, 12]

    pos = pos.sample(frac=1, random_state=42).reset_index(drop=True)
    csv_path = f"{tmp_dir}/{name}.csv"
    pos.to_csv(csv_path, index=False)

    # Sidecar for generation (clip bounds + integer columns + missing rates).
    side = {"clip": clip, "int_cols": int_cols, "na_cols": NA_COLS,
            "isna_names": isna_names,
            "na_rates": {c: float((df[df[TARGET] == 1][c].isna()).mean()) for c in NA_COLS}}
    register(name, csv_path, num_idx, cat_idx)
    with open(f"data/{name}/gmsc_clip.json", "w") as f:
        json.dump(side, f, indent=2)
    print(f"[{name}] {len(pos)} positive rows | int_cols={int_cols} | "
          f"na_rates={side['na_rates']} -> done", flush=True)


if __name__ == "__main__":
    main()
