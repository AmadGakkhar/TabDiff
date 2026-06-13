"""Generate synthetic FRAUD rows from a trained paysim Approach-2 TabDiff model.

Two modes:

  reject       (used for d1, the fraud-only model, ~100% fraud yield):
               unconditional sampling, keep rows whose isFraud == "1".

  conditional  (used for d2/d3/d4, the high-majority models, near-0% fraud yield):
               clamp the target column isFraud == "1" and generate (impute) all
               feature columns conditioned on it, via diffusion.sample_impute with
               the masks inverted relative to the paper's label-imputation setup
               (we OBSERVE the target and GENERATE the features). Yield is ~100%
               by construction, so collecting 1.6M fraud rows is feasible even
               for the full-data model. No y_only guidance model is used
               (guidance is built to steer the *masked* column with a target-only
               model; here the masked columns are the features, so plain
               conditioning is the correct mechanism).

Loads the best_ema checkpoint + cached config.pkl, reuses the dataset's encoders
for decoding (identical to Trainer.sample_synthetic).
"""
import os
import glob
import json
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

from utils_train import TabDiffDataset
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import split_num_cat_target, recover_data

TARGET_VALUE = "1"   # fraud


def build(dataname, device):
    info_path = f"data/{dataname}/info.json"
    with open(info_path) as f:
        info = json.load(f)

    ckpt_dir = f"tabdiff/ckpt/{dataname}/{dataname}"
    cands = glob.glob(f"{ckpt_dir}/best_ema_model*.pt")
    assert cands, f"No best_ema checkpoint under {ckpt_dir}"
    ckpt_path = cands[0]
    with open(os.path.join(ckpt_dir, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)   # complete: includes d_numerical, categories(+mask), per-column scheduler

    ds = TabDiffDataset(
        dataname, f"data/{dataname}", info, isTrain=True,
        dequant_dist=cfg["data"]["dequant_dist"],
        int_dequant_factor=cfg["data"]["int_dequant_factor"],
    )
    d_num, categories = ds.d_numerical, ds.categories

    backbone = UniModMLP(**cfg["unimodmlp_params"])
    model = Model(backbone, **cfg["diffusion_params"]["edm_params"]).to(device)
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_num,
        denoise_fn=model,
        y_only_model=None,
        **cfg["diffusion_params"],
        device=device,
    ).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    diffusion._denoise_fn.load_state_dict(sd["denoise_fn"])
    diffusion.num_schedule.load_state_dict(sd["num_schedule"])
    diffusion.cat_schedule.load_state_dict(sd["cat_schedule"])
    diffusion.eval()
    print(f"[{dataname}] loaded {os.path.basename(ckpt_path)} | d_num={d_num} categories={categories.tolist()}")
    return diffusion, ds, info


def decode(sample, ds, info):
    syn_num, syn_cat, syn_target = split_num_cat_target(
        sample, info, ds.num_inverse, ds.int_inverse, ds.cat_inverse
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    idx_name_mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    return syn_df


def fraud_target_code(ds):
    """Encoded integer code for isFraud == '1' (first categorical column)."""
    X = ds.X.cpu().numpy()
    xcat = X[:, ds.d_numerical:]
    decoded = ds.cat_inverse(xcat)
    # cat_inverse may return ints or strings depending on the encoder; compare as str.
    rows = np.where(decoded[:, 0].astype(str) == TARGET_VALUE)[0]
    assert len(rows), "no fraud rows found in training data"
    return int(round(float(xcat[rows[0], 0])))


def write_chunk(df, out_path, wrote_header):
    df.to_csv(out_path, mode="a", header=not wrote_header, index=False)
    return True


@torch.no_grad()
def gen_reject(diffusion, ds, info, n_fraud, batch, out_path, target_col):
    wrote = False
    kept = 0
    t0 = time.time()
    while kept < n_fraud:
        sample = diffusion.sample_all(batch, batch, keep_nan_samples=True)
        # drop all-zero (NaN) rows
        sample = sample[sample.sum(dim=1) != 0]
        if sample.shape[0] == 0:
            continue
        df = decode(sample, ds, info)
        df = df[df[target_col].astype(str) == TARGET_VALUE]
        if len(df) == 0:
            continue
        take = min(len(df), n_fraud - kept)
        df = df.iloc[:take]
        wrote = write_chunk(df, out_path, wrote)
        kept += len(df)
        print(f"[reject] kept {kept}/{n_fraud}  ({time.time()-t0:.0f}s)", flush=True)
    return kept


@torch.no_grad()
def gen_conditional(diffusion, ds, info, n_fraud, batch, out_path, target_col,
                    impute_condition="x_t", resample_rounds=1):
    d_num = ds.d_numerical
    d_cat = len(ds.categories)
    num_mask_idx = list(range(d_num))          # generate all numerical features
    cat_mask_idx = list(range(1, d_cat))       # generate all categorical features (target is col 0)
    code = fraud_target_code(ds)
    device = diffusion.device
    print(f"[conditional] fraud target code={code} | mask num={num_mask_idx} cat={cat_mask_idx}", flush=True)

    wrote = False
    kept = 0
    t0 = time.time()
    while kept < n_fraud:
        b = min(batch, n_fraud - kept)
        x_num = torch.zeros((b, d_num), dtype=torch.float32, device=device)
        x_cat = torch.zeros((b, d_cat), dtype=torch.long, device=device)
        x_cat[:, 0] = code                     # observe isFraud == 1
        sample = diffusion.sample_impute(
            x_num, x_cat, num_mask_idx, cat_mask_idx,
            resample_rounds, impute_condition, w_num=0.0, w_cat=0.0,
        )
        sample = sample[sample.sum(dim=1) != 0]
        if sample.shape[0] == 0:
            continue
        df = decode(sample, ds, info)
        # target is clamped, but keep the guard (decode/inverse edge cases)
        df = df[df[target_col].astype(str) == TARGET_VALUE]
        if len(df) == 0:
            continue
        wrote = write_chunk(df, out_path, wrote)
        kept += len(df)
        print(f"[conditional] kept {kept}/{n_fraud}  ({time.time()-t0:.0f}s)", flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_fraud", type=int, default=1_600_000)
    ap.add_argument("--mode", choices=["reject", "conditional"], required=True)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu != -1 and torch.cuda.is_available()) else "cpu"
    np.set_printoptions(suppress=True)

    diffusion, ds, info = build(args.dataname, device)
    target_col = info["column_names"][info["target_col_idx"][0]] \
        if info.get("column_names") else None
    # column_names is null in our info; resolve target name via idx_name_mapping
    if target_col is None:
        target_col = {int(k): v for k, v in info["idx_name_mapping"].items()}[info["target_col_idx"][0]]

    out_dir = f"tabdiff/synthetic_fraud/{args.dataname}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or f"{out_dir}/fraud_{args.n_fraud}.csv"
    if os.path.exists(out_path):
        os.remove(out_path)

    batch = args.batch or (4096 if args.mode == "reject" else 20000)
    print(f"[{args.dataname}] mode={args.mode} target_col={target_col} -> {out_path}", flush=True)

    t0 = time.time()
    if args.mode == "reject":
        kept = gen_reject(diffusion, ds, info, args.n_fraud, batch, out_path, target_col)
    else:
        kept = gen_conditional(diffusion, ds, info, args.n_fraud, batch, out_path, target_col)
    print(f"[{args.dataname}] DONE: {kept} fraud rows in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
