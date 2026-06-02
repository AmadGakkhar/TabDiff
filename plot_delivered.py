"""Save real-vs-synthetic density-comparison plots for a generated CSV.
Usage: python plot_delivered.py <synthetic_csv> <out_png> [dataname]
"""
import sys, json, torch, pandas as pd
from tabdiff.metrics import TabMetrics

syn_csv, out_png = sys.argv[1], sys.argv[2]
dataname = sys.argv[3] if len(sys.argv) > 3 else "fraud"

info = json.load(open(f"data/{dataname}/info.json"))
device = "cuda" if torch.cuda.is_available() else "cpu"
metrics = TabMetrics(f"synthetic/{dataname}/real.csv", f"synthetic/{dataname}/test.csv",
                     None, info, device, metric_list=["density"])

syn = pd.read_csv(syn_csv)
img = metrics.plot_density(syn)
img.save(out_png) if hasattr(img, "save") else open(out_png, "wb").write(img)
print(f"saved density plot -> {out_png}  (synthetic shape={syn.shape})")
