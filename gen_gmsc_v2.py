"""Generate v2 synthetic positive GMSC rows from a trained gmsc_l<N>_v2 model.

Post-processing (using data/<name>/gmsc_clip.json written by prepare_gmsc_v2.py):
  - filter target numerically (== 1), like gen_gmsc.py
  - restore NaN: where a *_isna flag rounds to 1, set the base column back to NaN
  - round integer columns to whole numbers (skip NaN)
  - clip every numerical column to the real [min, max] (skip NaN)
  - drop the isna helper columns, emit the original 11-column schema

  python gen_gmsc_v2.py --split <N> --n <COUNT> --gpu <G> --out <PATH>
"""
import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch

import generate_paysim_app2 as G

TARGET = "SeriousDlqin2yrs"
ORIG_COLS = [
    TARGET, "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def postprocess(df, side):
    # restore NaN from isna flags
    for c, flag in zip(side["na_cols"], side["isna_names"]):
        mask = pd.to_numeric(df[flag], errors="coerce").round() == 1
        df.loc[mask, c] = np.nan
    # round integers (non-NaN only)
    for c in side["int_cols"]:
        df[c] = df[c].round()
    # clip to real range (NaN passes through)
    for c, (lo, hi) in side["clip"].items():
        df[c] = df[c].clip(lo, hi)
    df[TARGET] = 1
    return df[ORIG_COLS]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu != -1 and torch.cuda.is_available()) else "cpu"
    dataname = f"gmsc_l{args.split}_v2"
    diffusion, ds, info = G.build(dataname, device)
    with open(f"data/{dataname}/gmsc_clip.json") as f:
        side = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    kept = gen_total = 0
    wrote = False
    t0 = time.time()
    while kept < args.n:
        sample = diffusion.sample_all(args.batch, args.batch, keep_nan_samples=True)
        sample = sample[sample.sum(dim=1) != 0]
        gen_total += args.batch
        if sample.shape[0] == 0:
            continue
        df = G.decode(sample, ds, info)
        df = df[pd.to_numeric(df[TARGET], errors="coerce") == 1]
        if len(df) == 0:
            continue
        df = postprocess(df.copy(), side)
        df = df.iloc[: args.n - kept]
        df.to_csv(args.out, mode="a", header=not wrote, index=False)
        wrote = True
        kept += len(df)
        print(f"[{dataname}] kept {kept}/{args.n}  gen {gen_total:,}  yield {kept/gen_total:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"[{dataname}] DONE: {kept} positive rows in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
