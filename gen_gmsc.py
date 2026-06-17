"""Generate synthetic positive (SeriousDlqin2yrs==1) GMSC rows from a trained
positive-only TabDiff model.

Reuses generate_paysim_app2.build/decode (checkpoint load + decoder), but filters
the target NUMERICALLY (pd.to_numeric == 1) instead of the brittle string compare
`astype(str) == "1"` used by the paysim script — the GMSC target decodes to 1.0,
so a string compare against "1" matches nothing. Since the model is trained on
positives only, yield is ~100%.

  python gen_gmsc.py --split <N> --n <COUNT> --gpu <G> --out <PATH>
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
import torch

import generate_paysim_app2 as G

TARGET = "SeriousDlqin2yrs"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--n", type=int, required=True, help="number of positive rows to generate")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu != -1 and torch.cuda.is_available()) else "cpu"
    dataname = f"gmsc_l{args.split}"
    diffusion, ds, info = G.build(dataname, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    kept = 0
    gen_total = 0
    wrote = False
    t0 = time.time()
    while kept < args.n:
        sample = diffusion.sample_all(args.batch, args.batch, keep_nan_samples=True)
        sample = sample[sample.sum(dim=1) != 0]   # drop all-zero (NaN) rows
        gen_total += args.batch
        if sample.shape[0] == 0:
            continue
        df = G.decode(sample, ds, info)
        df = df[pd.to_numeric(df[TARGET], errors="coerce") == 1]
        if len(df) == 0:
            continue
        df = df.iloc[: args.n - kept].copy()
        df[TARGET] = 1   # clean integer target, matching the original splits
        df.to_csv(args.out, mode="a", header=not wrote, index=False)
        wrote = True
        kept += len(df)
        print(f"[{dataname}] kept {kept}/{args.n}  gen {gen_total:,}  yield {kept/gen_total:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"[{dataname}] DONE: {kept} positive rows in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
