"""Generate synthetic fraud rows from one trained Approach-1 ensemble member.

Samples from the checkpoint in chunks and keeps only fraud rows (FraudFound_P==1).
Non-fraud rows produced by the model are discarded — Approach 1 only needs the
synthetic minority; the real non-fraud is kept for the downstream classifier.

Usage (called per member by run_expv01_app1.sh):
  python gen_expv01_app1.py --dataname expv01_m1 --exp_name expv01_m1 \
      --n_fraud 625 --out synthetic_outputs/expv01_app1/member_1_fraud.csv
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import torch

from gen_balanced import build_trainer


def _find_ckpt(dataname, exp_name):
    """Return the best available checkpoint compatible with Trainer.__init__.

    Trainer expects {'denoise_fn': ..., 'num_schedule': ..., 'cat_schedule': ...}.
    - best_ema_model_*.pt  uses this format (saved only when epoch > 4000)
    - model_*.pt           uses this format (saved every check_val_every steps)
    - ema_model_*.pt       is a raw state_dict — incompatible, never used here
    """
    parent = f"tabdiff/ckpt/{dataname}/{exp_name}"
    best = glob.glob(f"{parent}/best_ema_model*")
    if best:
        return best[0]
    # model_*.pt has the right dict format; pick the highest step number
    snapshots = glob.glob(f"{parent}/model_*.pt")
    assert snapshots, f"No checkpoint found under {parent}"
    return max(snapshots, key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))

import warnings
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname",  required=True)
    ap.add_argument("--exp_name",  required=True)
    ap.add_argument("--n_fraud",   type=int, required=True, help="Fraud rows to collect from this member")
    ap.add_argument("--out",       required=True)
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--chunk",     type=int, default=4096, help="Rows generated per iteration")
    ap.add_argument("--max_iters", type=int, default=30)
    ap.add_argument("--target_col", default=None, help="Target column name (overrides info/default)")
    ap.add_argument("--gpu",       type=int, default=0)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu != -1 and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = args.ckpt_path or _find_ckpt(args.dataname, args.exp_name)
    print(f"Using checkpoint: {ckpt_path}", flush=True)
    trainer, info = build_trainer(args.dataname, args.exp_name, ckpt_path, device)
    trainer.diffusion.eval()

    if args.target_col:
        tcol = args.target_col
    else:
        col_names = info.get("column_names")
        tcol = col_names[info["target_col_idx"][0]] if col_names else "FraudFound_P"
    print(f"Target column: {tcol} | collecting {args.n_fraud} fraud rows | device={device}", flush=True)

    fraud_parts = []
    n_f = total_gen = 0

    for it in range(1, args.max_iters + 1):
        df = trainer.sample_synthetic(args.chunk, keep_nan_samples=False)
        total_gen += len(df)
        tv = pd.to_numeric(df[tcol], errors="coerce")
        f = df[tv == 1]
        if len(f):
            fraud_parts.append(f)
            n_f += len(f)
        yield_rate = n_f / total_gen if total_gen else 0
        print(
            f"[iter {it}] generated={total_gen} | fraud_collected={n_f}/{args.n_fraud} | yield={yield_rate:.1%}",
            flush=True,
        )
        if n_f >= args.n_fraud:
            break
    else:
        print(
            f"WARNING: hit max_iters={args.max_iters} — only collected {n_f}/{args.n_fraud} fraud rows",
            flush=True,
        )

    if not fraud_parts:
        raise RuntimeError(
            "No fraud rows generated. Check that the model trained correctly "
            "and that the target column is being decoded."
        )

    out_df = pd.concat(fraud_parts, ignore_index=True).iloc[:args.n_fraud]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} fraud rows x {out_df.shape[1]} cols -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
