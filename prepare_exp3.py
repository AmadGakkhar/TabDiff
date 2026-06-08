"""Prepare data/exp_3/car_insurance_fraud.csv for TabDiff.

The file ships pre-encoded: 9 z-scored continuous columns, 58 one-hot binary
columns (11 mutually-exclusive groups) and a binary target `fraud_reported`.

TabDiff models mixed-type data natively, so we DECODE the one-hot groups back
to 11 categorical columns (9 continuous + 11 categorical + target = 21 cols),
register the dataset, and run process_dataset.py. We also persist
`data/car_fraud_exp3/encode_map.json` so the generated samples can be
re-encoded back to the exact original 68-column layout (drop-in replacement).
"""
import os
import json
import subprocess

import pandas as pd

SRC = "data/exp_3/car_insurance_fraud.csv"
NAME = "car_fraud_exp3"
INFO_DIR = "data/Info"
DATA_DIR = f"data/{NAME}"
TARGET = "fraud_reported"

CONT_COLS = [
    "policy_deductible", "policy_annual_premium", "insured_age",
    "incident_hour_of_the_day", "number_of_vehicles_involved",
    "bodily_injuries", "witnesses", "claim_amount", "total_claim_amount",
]

# One-hot group prefixes, in a fixed order. Each group's member columns are
# named "<prefix>_<value>"; decoding recovers <value> as the category.
GROUP_PREFIXES = [
    "policy_state", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "incident_type",
    "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "police_report_available",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_csv(SRC)
    orig_columns = df.columns.tolist()

    bin_cols = [c for c in df.columns if c not in CONT_COLS and c != TARGET]

    # Map each binary column to its group; recover category = suffix after prefix.
    groups = {p: [] for p in GROUP_PREFIXES}
    for c in bin_cols:
        prefix = next((p for p in GROUP_PREFIXES if c.startswith(p + "_")), None)
        assert prefix is not None, f"Column {c} matches no known group prefix"
        groups[prefix].append(c)

    decoded = pd.DataFrame(index=df.index)
    for col in CONT_COLS:
        decoded[col] = df[col]

    group_categories = {}
    for prefix in GROUP_PREFIXES:
        members = groups[prefix]
        sub = df[members]
        assert (sub.sum(axis=1) == 1).all(), f"{prefix} is not clean one-hot"
        # category value per row = member column name with prefix stripped
        cat = sub.idxmax(axis=1).str[len(prefix) + 1:]
        decoded[prefix] = cat
        group_categories[prefix] = [m[len(prefix) + 1:] for m in members]

    decoded[TARGET] = df[TARGET].astype(int)

    csv_dst = f"{DATA_DIR}/{NAME}.csv"
    decoded.to_csv(csv_dst, index=False)

    column_names = decoded.columns.tolist()
    n_cont = len(CONT_COLS)
    n_cat = len(GROUP_PREFIXES)
    num_col_idx = list(range(n_cont))
    cat_col_idx = list(range(n_cont, n_cont + n_cat))
    target_col_idx = [n_cont + n_cat]

    info = {
        "name": NAME,
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": csv_dst,
        "val_path": None,
        "test_path": None,
    }
    with open(f"{INFO_DIR}/{NAME}.json", "w") as f:
        json.dump(info, f, indent=4)

    # Everything needed to re-encode synthetic samples to the original layout.
    encode_map = {
        "orig_columns": orig_columns,
        "cont_cols": CONT_COLS,
        "group_prefixes": GROUP_PREFIXES,
        "group_categories": group_categories,
        "target": TARGET,
        "target_dist": df[TARGET].astype(int).value_counts().to_dict(),
        "n_rows": int(len(df)),
        "decoded_columns": column_names,
    }
    with open(f"{DATA_DIR}/encode_map.json", "w") as f:
        json.dump(encode_map, f, indent=4)

    print(f"Decoded -> {csv_dst}  shape={decoded.shape}")
    print(f"num_col_idx={num_col_idx}\ncat_col_idx={cat_col_idx}\ntarget_col_idx={target_col_idx}")
    print(f"Target distribution: {encode_map['target_dist']}")
    print(f"\n=== Registering {NAME} ===", flush=True)
    subprocess.run(["python", "process_dataset.py", "--dataname", NAME], check=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
