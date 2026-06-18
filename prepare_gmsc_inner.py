"""Prepare positive-only GMSC dataset from train_inner.csv for one split (v2 methodology).

Identical methodology to prepare_gmsc_v2.py (missingness indicators + integer dequant
+ dim_t=512 + clip sidecar), but the source is level_<N>/train_inner.csv (the inner
training fold) instead of train.csv. val.csv and the held-out test.csv are NEVER read.

  python prepare_gmsc_inner.py <SPLIT_ID>      # -> dataset gmsc_l<SPLIT_ID>_inner
"""
import os
import sys
import json

import pandas as pd
import src
from prepare_gmsc_v2 import build_gmsc_config, register, TARGET, NA_COLS, NUM_NAMES

SPLIT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT_CSV = f"/srv/datasets/GMSC/splits/level_{SPLIT_ID}/train_inner.csv"
name      = f"gmsc_l{SPLIT_ID}_inner"
tmp_dir   = "data/gmsc_partitions"


def main():
    os.makedirs(tmp_dir, exist_ok=True)

    cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
    src.dump_config(build_gmsc_config(), cfg_path)
    print(f"[{name}] config -> {cfg_path}", flush=True)

    df = pd.read_csv(SPLIT_CSV)
    assert df.columns.tolist()[0] == TARGET and len(df.columns) == 11
    pos = df[df[TARGET] == 1].reset_index(drop=True)

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
