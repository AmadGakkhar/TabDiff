"""Prepare positive-only GMSC dataset from train.csv for one split (v3 — tiny-data tuning).

Same v2 fidelity methodology (missingness indicators + integer dequant + clip sidecar),
but the positive counts in the *new* splits are ~10x smaller (L0=160 ... L4=802), so the
model config is RE-TUNED to avoid memorising such tiny sets and to keep reasonable
variation in the samples:

  - dim_t  : 512 (v2) -> 128   (much less capacity to memorise ~150-800 rows)
  - batch  : 256 (v2) -> 32/64 (sub-full-batch => SGD noise = regularisation + variation)
  - steps  : 3500 (v2) -> 1500 (fewer passes; best-EMA-ckpt still selects the best)
  - w_decay: 1e-4 (v2) -> 1e-3 (stronger regularisation)
  - kept   : dequant_dist="uniform", int_dequant_factor=1.0, num_timesteps=100,
             stochastic sampler (variation), missingness indicators, tail clip.

  python prepare_gmsc_v3.py <SPLIT_ID>      # -> dataset gmsc_l<SPLIT_ID>_v3 (source: train.csv)
"""
import os
import sys
import json

import pandas as pd
import src
from prepare_paysim_app2 import build_config
from prepare_gmsc_v2 import register, TARGET, NA_COLS, NUM_NAMES

SPLIT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT_CSV = f"/srv/datasets/GMSC/splits/level_{SPLIT_ID}/train.csv"
name      = f"gmsc_l{SPLIT_ID}_v3"
tmp_dir   = "data/gmsc_partitions"


def build_config_v3(n_pos):
    """Size-adaptive, anti-memorisation config for tiny positive sets."""
    cfg = build_config("d1")                       # d1 base: lr 1e-3, ema, reduce_lr_on_plateau
    cfg["unimodmlp_params"]["dim_t"]   = 256        # modest capacity (128 under-fit the marginals)
    cfg["data"]["dequant_dist"]        = "uniform"  # integer handling (v2)
    cfg["data"]["int_dequant_factor"]  = 1.0
    m = cfg["train"]["main"]
    m["batch_size"]            = 64 if n_pos < 400 else 128  # SGD noise; >=2 batches/epoch
    m["steps"]                 = 2500
    m["weight_decay"]          = 3e-4               # regularise but allow the marginals to fit
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
    src.dump_config(build_config_v3(n_pos), cfg_path)
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
