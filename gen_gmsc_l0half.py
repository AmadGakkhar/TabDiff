"""Generate synthetic positive GMSC rows from the trained gmsc_l0half model.

Same post-processing as gen_gmsc_v2.py (NaN restore / integer round / tail clip).

  python gen_gmsc_l0half.py --n <COUNT> --gpu <G> --out <PATH>
"""
import os
import json
import time
import argparse

import pandas as pd
import torch

import generate_paysim_app2 as G
from gen_gmsc_v2 import postprocess, TARGET


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu != -1 and torch.cuda.is_available()) else "cpu"
    dataname = "gmsc_l0half"
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
