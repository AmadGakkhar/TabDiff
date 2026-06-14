"""Prepare ONLY the d1 (fraud-only) dataset + config for a given paysim split.

Reuses build_config/register and the schema constants from prepare_paysim_app2,
but skips the expensive d2/d3/d4 majority partitions entirely.

  python prepare_d1.py <SPLIT_ID>
"""
import os
import sys
import pandas as pd
import src
from prepare_paysim_app2 import (
    build_config, register, DROP_COLS, TARGET, TARGET_COL_IDX, N_COLS,
)

SPLIT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT_CSV = f"/home/amad/projects/datasets/paysim/splits/{SPLIT_ID}/train.csv"
name      = f"ps{SPLIT_ID}_d1"
tmp_dir   = "data/paysim_app2_partitions"
os.makedirs(tmp_dir, exist_ok=True)

cfg_path = f"tabdiff/configs/tabdiff_configs_{name}.toml"
src.dump_config(build_config("d1"), cfg_path)
print(f"[{name}] config -> {cfg_path}", flush=True)

if os.path.exists(f"data/{name}/info.json"):
    print(f"[{name}] already registered — skipping data prep", flush=True)
    sys.exit(0)

df = pd.read_csv(SPLIT_CSV).drop(columns=DROP_COLS)
assert df.columns.tolist()[TARGET_COL_IDX[0]] == TARGET
assert len(df.columns) == N_COLS
for i in [8] + TARGET_COL_IDX:
    df[df.columns[i]] = df[df.columns[i]].astype(int).astype(str)

fraud = df[df[TARGET] == "1"].reset_index(drop=True)
fraud = fraud.sample(frac=1, random_state=42).reset_index(drop=True)
csv_path = f"{tmp_dir}/{name}.csv"
fraud.to_csv(csv_path, index=False)
print(f"[{name}] {len(fraud)} fraud rows -> registering", flush=True)
register(name, csv_path)
print(f"[{name}] done", flush=True)
