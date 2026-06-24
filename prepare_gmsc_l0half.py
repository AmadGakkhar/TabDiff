"""Prepare positive-only GMSC dataset from splits_l0half/level_0/train.csv (ultra-tiny).

Same v2/v3 fidelity methodology (missingness indicators + integer dequant + clip sidecar),
but this split has only ~80 positives (even more extreme imbalance than level_0's 160), so
the config is dialed down further to avoid memorising ~72 training rows:
  - dim_t 128 (vs v3's 256), batch 32, weight_decay 5e-4, steps 2500.

  python prepare_gmsc_l0half.py      # -> dataset gmsc_l0half (source: splits_l0half/level_0/train.csv)
"""
import os
import json

import pandas as pd
import src
from prepare_paysim_app2 import build_config
from prepare_gmsc_v2 import register, TARGET, NA_COLS, NUM_NAMES

SPLIT_CSV = "/srv/datasets/GMSC/splits_l0half/level_0/train.csv"
name      = "gmsc_l0half"
tmp_dir   = "data/gmsc_partitions"


def build_config_ultratiny():
    cfg = build_config("d1")
    cfg["unimodmlp_params"]["dim_t"]   = 128        # minimal capacity for ~72 train rows
    cfg["data"]["dequant_dist"]        = "uniform"
    cfg["data"]["int_dequant_factor"]  = 1.0
    m = cfg["train"]["main"]
    m["batch_size"]            = 32                 # SGD noise on a tiny set
    m["steps"]                 = 2500
    m["weight_decay"]          = 5e-4               # stronger regularisation
    m["check_val_every"]       = 150
    m["reduce_lr_patience"]    = 20
    m["best_ckpt_start_epoch"] = 150
    m["early_stop_patience"]   = 0
    m["val_sample_size"]       = 2000
    return cfg


def main():
    os.makedirs(tmp_dir, exist_ok=True)

    df = pd.read_csv(SPLIT_CSV)
    assert df.columns.tolist()[0] == TARGET and len(df.columns) == 11
    pos = df[df[TARGET] == 1].reset_index(drop=True)
    n_pos = len(pos)

    cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
    src.dump_config(build_config_ultratiny(), cfg_path)
    print(f"[{name}] n_pos={n_pos} config -> {cfg_path}", flush=True)

    clip = {c: [float(pos[c].min()), float(pos[c].max())] for c in NUM_NAMES}
    int_cols = [c for c in NUM_NAMES if pos[c].dropna().mod(1).eq(0).all()]

    for c in NA_COLS:
        pos[f"{c}_isna"] = pos[c].isna().astype(int).astype(str)
    pos[NUM_NAMES] = pos[NUM_NAMES].fillna(pos[NUM_NAMES].median())
    pos[TARGET] = pos[TARGET].astype(int).astype(str)

    isna_names = [f"{c}_isna" for c in NA_COLS]
    pos = pos[[TARGET] + NUM_NAMES + isna_names]

    pos = pos.sample(frac=1, random_state=42).reset_index(drop=True)
    csv_path = f"{tmp_dir}/{name}.csv"
    pos.to_csv(csv_path, index=False)

    side = {"clip": clip, "int_cols": int_cols, "na_cols": NA_COLS,
            "isna_names": isna_names,
            "na_rates": {c: float((df[df[TARGET] == 1][c].isna()).mean()) for c in NA_COLS}}
    register(name, csv_path, list(range(1, 11)), [11, 12])
    with open(f"data/{name}/gmsc_clip.json", "w") as f:
        json.dump(side, f, indent=2)
    print(f"[{name}] {len(pos)} positive rows | int_cols={int_cols} | "
          f"na_rates={side['na_rates']} -> done", flush=True)


if __name__ == "__main__":
    main()
