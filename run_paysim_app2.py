"""Approach 2 train->generate orchestrator for paysim split 0.

One worker per GPU handles ONE dose end-to-end: train the main TabDiff model,
then immediately generate 1.6M synthetic fraud rows from it. This way each dose
produces usable data as soon as its own training finishes (no waiting for the
whole sweep).

  ps0_d1  GPU 0  reject sampling      (fraud-only, ~100% yield)
  ps0_d2  GPU 1  conditional impute   (+25% majority)
  ps0_d3  GPU 2  conditional impute   (+50% majority)
  ps0_d4  GPU 3  conditional impute   (full data; may be dropped later)

Run prepare_paysim_app2.py first.
"""
import os
import json
import time
import threading
import sys
import subprocess
from datetime import datetime

# Split index from argv (default 0); prefix ps<SPLIT_ID> keeps splits separate.
SPLIT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PFX = f"ps{SPLIT_ID}"

# Reject-only generation. Target 100k fraud rows from EVERY dose. Reject yield =
# training fraud share (~2,440 gen rows/s total), so realistic outcomes under the
# 4h cap: d1 (~100%) hits 100k in <1min; d2 (~0.51%) ~2.2h -> 100k; d3 (~0.26%)
# ~4h -> ~90k; d4 (~0.13%) cap-bound -> ~45k. Low-yield doses produce PARTIAL
# counts by design (training is uncapped; only generation is time-capped).
TARGETS = {"d1": 100_000, "d2": 100_000, "d3": 100_000, "d4": 100_000}
SAMPLE_BATCH = 32768
GEN_MAX_SECONDS = 14_400   # 4h wall-clock cap per generation (budget guard)

# (dataname, gpu, generation mode)
JOBS = [
    (f"{PFX}_d1", 0, "reject"),
    (f"{PFX}_d2", 1, "reject"),
    (f"{PFX}_d3", 2, "reject"),
    (f"{PFX}_d4", 3, "reject"),
]

LOGDIR = f"logs/paysim_app2_{PFX}"
os.makedirs(LOGDIR, exist_ok=True)
RUNLOG  = f"{LOGDIR}/run.log"
TIMINGS = f"{LOGDIR}/timings.json"

_log_lock = threading.Lock()
results = {}
results_lock = threading.Lock()


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{ts()}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(RUNLOG, "a") as f:
            f.write(line + "\n")


def fmt_dur(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def run(cmd, logfile):
    with open(logfile, "a") as lf:
        lf.write(f"\n===== {ts()} :: {' '.join(cmd)} =====\n")
        lf.flush()
        return subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)


def worker(dataname, gpu, mode):
    cfg = f"tabdiff/configs/tabdiff_configs_{dataname}.toml"
    train_log = f"{LOGDIR}/train_{dataname}.log"
    gen_log   = f"{LOGDIR}/gen_{dataname}.log"
    rec = {"gpu": gpu, "mode": mode}

    # ---- Train ----
    t0 = time.time()
    log(f">>> [{dataname}] TRAIN start on GPU {gpu}")
    rc = run(["python", "-u", "main.py",
              "--dataname", dataname, "--mode", "train", "--gpu", str(gpu),
              "--config_path", cfg, "--exp_name", dataname, "--no_wandb", "--resume"],
             train_log)
    rec["train_sec"] = time.time() - t0
    rec["train_rc"] = rc
    log(f"<<< [{dataname}] TRAIN {'done' if rc==0 else f'FAILED rc={rc}'} "
        f"on GPU {gpu} — {fmt_dur(rec['train_sec'])}")
    if rc != 0:
        with results_lock:
            results[dataname] = rec
        return

    # ---- Generate ----
    dose = dataname.split("_")[-1]            # d1..d4
    target = TARGETS[dose]
    t1 = time.time()
    log(f">>> [{dataname}] GENERATE ({mode}) {target} fraud on GPU {gpu}")
    grc = run(["python", "-u", "generate_paysim_app2.py",
               "--dataname", dataname, "--gpu", str(gpu),
               "--n_fraud", str(target), "--mode", mode,
               "--batch", str(SAMPLE_BATCH), "--max_seconds", str(GEN_MAX_SECONDS)],
              gen_log)
    rec["gen_sec"] = time.time() - t1
    rec["gen_rc"] = grc
    log(f"<<< [{dataname}] GENERATE {'done' if grc==0 else f'FAILED rc={grc}'} "
        f"on GPU {gpu} — {fmt_dur(rec['gen_sec'])} (see {gen_log})")
    with results_lock:
        results[dataname] = rec


def main():
    log(f"=== paysim Approach 2: train->generate {len(JOBS)} doses (reject), targets {TARGETS} ===")
    overall = time.time()
    threads = [threading.Thread(target=worker, args=j, daemon=True) for j in JOBS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = {
        "overall_start": datetime.fromtimestamp(overall).strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_clock": fmt_dur(time.time() - overall),
        "targets": TARGETS,
        "per_dose": results,
    }
    with open(TIMINGS, "w") as f:
        json.dump(summary, f, indent=2)
    log("=== ALL DONE ===")
    for name, r in results.items():
        log(f"    {name}: train {fmt_dur(r.get('train_sec',0))} rc={r.get('train_rc')} | "
            f"gen {fmt_dur(r.get('gen_sec',0))} rc={r.get('gen_rc')}")
    log(f"    total wall-clock: {summary['total_wall_clock']}")


if __name__ == "__main__":
    main()
