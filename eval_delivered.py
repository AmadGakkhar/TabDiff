"""Standalone quality evaluation for an already-generated synthetic CSV.
Usage: python eval_delivered.py <synthetic_csv> [dataname]
Runs the repo's density + MLE + C2ST metrics against the real/test splits.
"""
import sys, json, torch, pandas as pd
from tabdiff.metrics import TabMetrics

syn_csv = sys.argv[1]
dataname = sys.argv[2] if len(sys.argv) > 2 else "fraud"

info = json.load(open(f"data/{dataname}/info.json"))
device = "cuda" if torch.cuda.is_available() else "cpu"

real_data_path = f"synthetic/{dataname}/real.csv"
test_data_path = f"synthetic/{dataname}/test.csv"

metrics = TabMetrics(real_data_path, test_data_path, None, info, device,
                     metric_list=["density", "mle", "c2st"])

syn = pd.read_csv(syn_csv)
print(f"Evaluating {syn_csv}  shape={syn.shape}\n")
out, _ = metrics.evaluate(syn)
print("\n===== QUALITY REPORT:", syn_csv, "=====")
for k, v in out.items():
    print(f"{k:24s} {v:.4f}")
