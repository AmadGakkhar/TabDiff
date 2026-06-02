"""Register exp_2 partitions as TabDiff datasets.

Each partition (full_train, fold0_train .. fold4_train) becomes its own dataset
so it can be modeled by an independent TabDiff model (no cross-fold leakage).

Column layout (from data/exp_2/schema.json): only `Age` (idx 10) is continuous,
`FraudFound_P` (idx 16) is the binary target, everything else is categorical.
"""
import os
import json
import shutil
import subprocess

PARTS = ["full_train", "fold0_train", "fold1_train", "fold2_train", "fold3_train", "fold4_train"]

NUM_COL_IDX = [10]          # Age
TARGET_COL_IDX = [16]       # FraudFound_P
N_COLS = 33
CAT_COL_IDX = [i for i in range(N_COLS) if i not in NUM_COL_IDX + TARGET_COL_IDX]

INFO_DIR = "data/Info"
SRC_DIR = "data/exp_2"


def register(part):
    data_dir = f"data/{part}"
    os.makedirs(data_dir, exist_ok=True)
    csv_dst = f"{data_dir}/{part}.csv"
    shutil.copyfile(f"{SRC_DIR}/{part}.csv", csv_dst)

    info = {
        "name": part,
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": NUM_COL_IDX,
        "cat_col_idx": CAT_COL_IDX,
        "target_col_idx": TARGET_COL_IDX,
        "file_type": "csv",
        "data_path": csv_dst,
        "val_path": None,
        "test_path": None,
    }
    with open(f"{INFO_DIR}/{part}.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n=== Registering {part} ===", flush=True)
    subprocess.run(["python", "process_dataset.py", "--dataname", part], check=True)


if __name__ == "__main__":
    for p in PARTS:
        register(p)
    print("\nAll partitions registered.")
