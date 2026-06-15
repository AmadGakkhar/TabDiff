"""Plot per-epoch loss curves for banksim d1 from the loss_history.csv written by
the trainer (train loss + TRUE held-out EMA loss)."""
import os
import glob
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS = [0, 1, 2, 3, 4]
LOGDIR = "logs/banksim_app2_d1"


def csv_path(s):
    return f"tabdiff/ckpt/bs{s}_d1/bs{s}_d1/loss_history.csv"


def best_ema_epoch(s):
    c = glob.glob(f"tabdiff/ckpt/bs{s}_d1/bs{s}_d1/best_ema_model_*.pt")
    if not c:
        return None
    m = re.search(r"best_ema_model_([\d.]+)_(\d+)\.pt", os.path.basename(c[0]))
    return (float(m.group(1)), int(m.group(2))) if m else None


def main():
    data = {s: pd.read_csv(csv_path(s)) for s in SPLITS if os.path.exists(csv_path(s))}

    # ---- per-split: train vs held-out EMA (held-out on log scale; scales differ) ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    for i, s in enumerate(SPLITS):
        ax = axes[i]
        df = data.get(s)
        if df is None:
            ax.set_visible(False)
            continue
        ax.plot(df.epoch, df.train_total, lw=0.9, color="tab:blue", label="train total")
        ax.plot(df.epoch, df.heldout_ema_total, lw=0.9, color="tab:orange", label="held-out EMA total")
        ax.set_yscale("log")
        be = best_ema_epoch(s)
        if be:
            ax.axvline(be[1], color="tab:red", ls="--", lw=1, label=f"best_ema @ {be[1]} ({be[0]:g})")
        ax.set_title(f"bs{s}_d1  ({len(df)} epochs)")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    axes[-1].set_visible(False)
    fig.suptitle("banksim d1 — train vs held-out EMA loss (log y)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{LOGDIR}/loss_curves_per_split.png", dpi=120)
    print(f"wrote {LOGDIR}/loss_curves_per_split.png")

    # ---- overlay: held-out EMA total, log scale, all splits ----
    fig2, ax = plt.subplots(figsize=(11, 6))
    for s in SPLITS:
        df = data.get(s)
        if df is None:
            continue
        ax.plot(df.epoch, df.heldout_ema_total, lw=1.0, label=f"bs{s}_d1")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("held-out EMA total loss (log)")
    ax.set_title("banksim d1 — held-out EMA loss overlay (all splits)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(f"{LOGDIR}/loss_curves_heldout_overlay.png", dpi=120)
    print(f"wrote {LOGDIR}/loss_curves_heldout_overlay.png")

    # ---- summary ----
    print("\nsplit  epochs  final_train  best_heldout_ema")
    for s in SPLITS:
        df = data.get(s)
        if df is None:
            continue
        print(f"bs{s}    {len(df):5d}   {df.train_total.iloc[-1]:.3f}      "
              f"{df.heldout_ema_total.min():.3f}")


if __name__ == "__main__":
    main()
