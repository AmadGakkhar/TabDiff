"""Throughput vs batch-size benchmark for the paysim TabDiff model.

Times forward+backward (one optimizer step) at several batch sizes on one GPU,
reporting samples/sec, time/epoch estimate, and peak GPU memory. Helps pick a
batch that is GPU-efficient WITHOUT slashing gradient-updates-per-epoch (which
matters now that early stopping is off and epochs are fixed).
"""
import sys, time, json, pickle
import numpy as np, torch
from torch.utils.data import DataLoader
from utils_train import TabDiffDataset
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion

dataname = sys.argv[1] if len(sys.argv) > 1 else "ps0_d4"
gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BATCHES = [int(b) for b in (sys.argv[3].split(",") if len(sys.argv) > 3 else
                            ["4096", "8192", "16384", "32768", "65536"])]
device = f"cuda:{gpu}"
STEPS = 25

info = json.load(open(f"data/{dataname}/info.json"))
cfg = pickle.load(open(f"tabdiff/ckpt/{dataname}/{dataname}/config.pkl", "rb")) \
    if False else None
import src
cfg = src.load_config(f"tabdiff/configs/tabdiff_configs_{dataname}.toml")
ds = TabDiffDataset(dataname, f"data/{dataname}", info, isTrain=True,
                    dequant_dist=cfg["data"]["dequant_dist"], int_dequant_factor=cfg["data"]["int_dequant_factor"])
n_rows = len(ds)
d_num, categories = ds.d_numerical, ds.categories
cfg["unimodmlp_params"]["d_numerical"] = d_num
cfg["unimodmlp_params"]["categories"] = (categories + 1).tolist()
cfg["diffusion_params"]["scheduler"] = "power_mean_per_column"
cfg["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"

print(f"{dataname}: {n_rows:,} train rows | dim_t={cfg['unimodmlp_params']['dim_t']}")
print(f"{'batch':>8} {'samp/s':>12} {'steps/epoch':>12} {'sec/epoch':>10} {'epochs->updates':>16} {'peakMEM_MB':>11}")

for bs in BATCHES:
    try:
        backbone = UniModMLP(**cfg["unimodmlp_params"])
        model = Model(backbone, **cfg["diffusion_params"]["edm_params"]).to(device)
        diff = UnifiedCtimeDiffusion(num_classes=categories, num_numerical_features=d_num,
                                     denoise_fn=model, y_only_model=None, **cfg["diffusion_params"], device=device).to(device)
        opt = torch.optim.AdamW(diff.parameters(), lr=1e-3)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
        it = iter(loader)
        torch.cuda.reset_peak_memory_stats(device)
        # warmup
        for _ in range(3):
            x = next(it).float().to(device)
            opt.zero_grad(); dl, cl = diff.mixed_loss(x); (dl + cl).backward(); opt.step()
        torch.cuda.synchronize(device)
        t0 = time.time(); seen = 0
        for _ in range(STEPS):
            x = next(it).float().to(device)
            opt.zero_grad(); dl, cl = diff.mixed_loss(x); (dl + cl).backward(); opt.step()
            seen += len(x)
        torch.cuda.synchronize(device)
        dt = time.time() - t0
        sps = seen / dt
        steps_ep = int(np.ceil(n_rows / bs))
        sec_ep = n_rows / sps
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"{bs:>8} {sps:>12,.0f} {steps_ep:>12,} {sec_ep:>10.1f} {steps_ep*500:>16,} {peak:>11.0f}")
        del diff, model, backbone, opt, loader, it; torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"{bs:>8}  OOM/err: {str(e)[:60]}")
        torch.cuda.empty_cache()
