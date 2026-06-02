"""Generate a fully-synthetic, class-balanced version of an exp_2 partition.

Loads a trained TabDiff checkpoint and samples until it has collected the
requested number of synthetic fraud (FraudFound_P==1) and non-fraud rows,
then writes a shuffled balanced CSV. TabDiff has no native class-conditional
sampling, so we over-generate an unconditional pool and filter by the target
column (the same strategy used in the previous session).

Usage:
  python gen_balanced.py --dataname full_train --exp_name full_train \
      --n_fraud 11947 --n_nonfraud 11947 --out synthetic_outputs/exp_2/full_train_balanced.csv
"""
import os
import glob
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

import src
from tabdiff.metrics import TabMetrics
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer
from utils_train import TabDiffDataset

import warnings
warnings.filterwarnings("ignore")


def build_trainer(dataname, exp_name, ckpt_path, device):
    """Replicates the test-mode model/trainer construction from tabdiff/main.py."""
    info = json.load(open(f"data/{dataname}/info.json"))
    data_dir = f"data/{dataname}"
    curr_dir = os.path.dirname(os.path.abspath("tabdiff/main.py"))  # repo root + tabdiff

    if ckpt_path is None:
        parent = f"tabdiff/ckpt/{dataname}/{exp_name}"
        arr = glob.glob(f"{parent}/best_ema_model*")
        assert arr, f"No best_ema_model checkpoint under {parent}"
        ckpt_path = arr[0]
    print(f"Using checkpoint: {ckpt_path}", flush=True)

    raw_config = pickle.load(open(os.path.join(os.path.dirname(ckpt_path), "config.pkl"), "rb"))

    train_data = TabDiffDataset(dataname, data_dir, info, y_only=False, isTrain=True,
                                dequant_dist=raw_config["data"]["dequant_dist"],
                                int_dequant_factor=raw_config["data"]["int_dequant_factor"])
    val_data = TabDiffDataset(dataname, data_dir, info, y_only=False, isTrain=False,
                              dequant_dist=raw_config["data"]["dequant_dist"],
                              int_dequant_factor=raw_config["data"]["int_dequant_factor"])
    d_numerical, categories = train_data.d_numerical, train_data.categories

    real_data_path = f"synthetic/{dataname}/real.csv"
    test_data_path = f"synthetic/{dataname}/test.csv"
    metrics = TabMetrics(real_data_path, test_data_path, None, info, device, metric_list=["density"])

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    # match main.py: learnable per-column schedule for the main (non-y_only) model
    raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
    raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories, num_numerical_features=d_numerical,
        denoise_fn=model, y_only_model=None,
        **raw_config["diffusion_params"], device=device,
    )
    diffusion.to(device)

    class _NullLogger:
        def log(self, *a, **k): pass
        def define_metric(self, *a, **k): pass

    trainer = Trainer(
        diffusion, None, train_data, val_data, metrics, _NullLogger(),
        **raw_config["train"]["main"],
        sample_batch_size=raw_config["sample"]["batch_size"],
        num_samples_to_generate=None,
        model_save_path=None, result_save_path=None,
        device=device, ckpt_path=ckpt_path, y_only=False,
    )
    return trainer, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--n_fraud", type=int, required=True)
    ap.add_argument("--n_nonfraud", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--max_iters", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu != -1 and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer, info = build_trainer(args.dataname, args.exp_name, args.ckpt_path, device)
    trainer.diffusion.eval()
    tcol = info["column_names"][info["target_col_idx"][0]]
    print(f"Target column: {tcol} | targets: fraud={args.n_fraud}, non-fraud={args.n_nonfraud}", flush=True)

    fraud_parts, non_parts = [], []
    n_f = n_n = 0
    total_gen = 0
    for it in range(1, args.max_iters + 1):
        df = trainer.sample_synthetic(args.chunk, keep_nan_samples=False)
        total_gen += len(df)
        tv = pd.to_numeric(df[tcol], errors="coerce")
        if n_f < args.n_fraud:
            f = df[tv == 1]
            if len(f):
                fraud_parts.append(f); n_f += len(f)
        if n_n < args.n_nonfraud:
            nf = df[tv == 0]
            if len(nf):
                non_parts.append(nf); n_n += len(nf)
        rate = (n_f / total_gen) if total_gen else 0
        print(f"[iter {it}] pool={total_gen} | fraud {n_f}/{args.n_fraud} | "
              f"non-fraud {n_n}/{args.n_nonfraud} | fraud_yield={rate:.3%}", flush=True)
        if n_f >= args.n_fraud and n_n >= args.n_nonfraud:
            break
    else:
        print(f"WARNING: hit max_iters={args.max_iters} before reaching targets "
              f"(fraud {n_f}/{args.n_fraud}, non-fraud {n_n}/{args.n_nonfraud})", flush=True)

    fraud_df = pd.concat(fraud_parts, ignore_index=True).iloc[:args.n_fraud]
    non_df = pd.concat(non_parts, ignore_index=True).iloc[:args.n_nonfraud]
    out_df = pd.concat([fraud_df, non_df], ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    vc = pd.to_numeric(out_df[tcol], errors="coerce").value_counts().to_dict()
    print(f"\nWrote {len(out_df)} rows to {args.out}", flush=True)
    print(f"Class balance: {vc} | fraud share={vc.get(1,0)/len(out_df):.3%}", flush=True)


if __name__ == "__main__":
    main()
