"""Generate a synthetic version of the exp_3 car-insurance-fraud dataset.

Samples from the trained TabDiff checkpoint until it has collected the target
number of fraud / non-fraud rows (default: the ORIGINAL class distribution and
size), then re-encodes the decoded mixed-type rows back to the exact original
68-column one-hot layout so the output is a drop-in replacement.

Usage:
  python gen_exp3.py --out synthetic_outputs/exp_3/car_insurance_fraud_synthetic.csv
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

from gen_balanced import build_trainer

import warnings
warnings.filterwarnings("ignore")

NAME = "car_fraud_exp3"


def reencode(df, emap):
    """Decoded mixed-type rows -> original 68-column one-hot layout."""
    out = pd.DataFrame(index=df.index)
    for col in emap["cont_cols"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for prefix in emap["group_prefixes"]:
        cats = emap["group_categories"][prefix]
        vals = df[prefix].astype(str)
        for cat in cats:
            out[f"{prefix}_{cat}"] = (vals == cat).astype(float)
    out[emap["target"]] = pd.to_numeric(df[emap["target"]], errors="coerce").astype(int)
    # exact original column order
    return out[emap["orig_columns"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", default=NAME)
    ap.add_argument("--exp_name", default=NAME)
    ap.add_argument("--out", default="synthetic_outputs/exp_3/car_insurance_fraud_synthetic.csv")
    ap.add_argument("--n_fraud", type=int, default=None, help="default: original count")
    ap.add_argument("--n_nonfraud", type=int, default=None, help="default: original count")
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--max_iters", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    emap = json.load(open(f"data/{args.dataname}/encode_map.json"))
    dist = {int(k): int(v) for k, v in emap["target_dist"].items()}
    n_fraud = args.n_fraud if args.n_fraud is not None else dist[1]
    n_nonfraud = args.n_nonfraud if args.n_nonfraud is not None else dist[0]

    device = f"cuda:{args.gpu}" if (args.gpu != -1 and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer, info = build_trainer(args.dataname, args.exp_name, args.ckpt_path, device)
    trainer.diffusion.eval()
    tcol = emap["target"]
    print(f"Targets: non-fraud={n_nonfraud}, fraud={n_fraud} (total={n_nonfraud + n_fraud})", flush=True)

    fraud_parts, non_parts = [], []
    n_f = n_n = total_gen = 0
    for it in range(1, args.max_iters + 1):
        df = trainer.sample_synthetic(args.chunk, keep_nan_samples=False)
        total_gen += len(df)
        tv = pd.to_numeric(df[tcol], errors="coerce")
        if n_f < n_fraud:
            f = df[tv == 1]
            if len(f):
                fraud_parts.append(f); n_f += len(f)
        if n_n < n_nonfraud:
            nf = df[tv == 0]
            if len(nf):
                non_parts.append(nf); n_n += len(nf)
        rate = (n_f / total_gen) if total_gen else 0
        print(f"[iter {it}] pool={total_gen} | fraud {n_f}/{n_fraud} | "
              f"non-fraud {n_n}/{n_nonfraud} | fraud_yield={rate:.3%}", flush=True)
        if n_f >= n_fraud and n_n >= n_nonfraud:
            break
    else:
        print(f"WARNING: hit max_iters={args.max_iters} before reaching targets "
              f"(fraud {n_f}/{n_fraud}, non-fraud {n_n}/{n_nonfraud})", flush=True)

    fraud_df = pd.concat(fraud_parts, ignore_index=True).iloc[:n_fraud]
    non_df = pd.concat(non_parts, ignore_index=True).iloc[:n_nonfraud]
    out_df = pd.concat([fraud_df, non_df], ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_df = reencode(out_df, emap)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    vc = out_df[tcol].value_counts().to_dict()
    print(f"\nWrote {len(out_df)} rows x {out_df.shape[1]} cols to {args.out}", flush=True)
    print(f"Class balance: {vc} | fraud share={vc.get(1, 0) / len(out_df):.3%}", flush=True)
    assert list(out_df.columns) == emap["orig_columns"], "Column layout mismatch!"
    print("Column layout matches original exactly.", flush=True)


if __name__ == "__main__":
    main()
