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

N_FRAUD = 1_600_000

# Split index from argv (default 0); prefix ps<SPLIT_ID> keeps splits separate.
SPLIT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PFX = f"ps{SPLIT_ID}"

# (dataname, gpu, generation mode)
JOBS = [
    (f"{PFX}_d1", 0, "reject"),
    (f"{PFX}_d2", 1, "conditional"),
    (f"{PFX}_d3", 2, "conditional"),
    (f"{PFX}_d4", 3, "conditional"),
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
    t1 = time.time()
    log(f">>> [{dataname}] GENERATE ({mode}) {N_FRAUD} fraud on GPU {gpu}")
    grc = run(["python", "-u", "generate_paysim_app2.py",
               "--dataname", dataname, "--gpu", str(gpu),
               "--n_fraud", str(N_FRAUD), "--mode", mode],
              gen_log)
    rec["gen_sec"] = time.time() - t1
    rec["gen_rc"] = grc
    log(f"<<< [{dataname}] GENERATE {'done' if grc==0 else f'FAILED rc={grc}'} "
        f"on GPU {gpu} — {fmt_dur(rec['gen_sec'])} (see {gen_log})")
    with results_lock:
        results[dataname] = rec


def main():
    log(f"=== paysim Approach 2: train->generate {len(JOBS)} doses, {N_FRAUD} fraud each ===")
    overall = time.time()
    threads = [threading.Thread(target=worker, args=j, daemon=True) for j in JOBS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = {
        "overall_start": datetime.fromtimestamp(overall).strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_clock": fmt_dur(time.time() - overall),
        "n_fraud_target": N_FRAUD,
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
