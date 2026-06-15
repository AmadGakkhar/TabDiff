"""Approach 2b train->generate orchestrator for banksim — d1 (fraud-only) ONLY.

One job per split: train the fraud-only TabDiff model, then immediately reject-sample
100k synthetic fraud rows from it (fraud-only model => ~100% yield, finishes in <1min).

Jobs are pulled from a queue by one worker per GPU, so with 5 splits on 4 T4s the
5th split starts as soon as any GPU frees up. The split's real test.csv is never
touched — the held-out signal is the 20% carved from train fraud in prepare.

Run prepare_banksim_app2.py first.
"""
import os
import json
import time
import queue
import threading
import sys
import subprocess
from datetime import datetime

N_GPUS       = 4
N_FRAUD      = 100_000
SAMPLE_BATCH = 4096
GEN_MAX_SECONDS = 3600   # 1h cap (fraud-only reject finishes in well under a minute)

with open("/home/amad/projects/datasets/banksim/splits_temporal/manifest.json") as f:
    N_SPLITS = json.load(f)["n_splits"]

SPLITS = [int(a) for a in sys.argv[1:]] or list(range(N_SPLITS))

LOGDIR = "logs/banksim_app2_d1"
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


def handle(split, gpu):
    dataname = f"bs{split}_d1"
    cfg = f"tabdiff/configs/tabdiff_configs_{dataname}.toml"
    train_log = f"{LOGDIR}/train_{dataname}.log"
    gen_log   = f"{LOGDIR}/gen_{dataname}.log"
    rec = {"gpu": gpu, "split": split}

    # ---- Train ----
    t0 = time.time()
    log(f">>> [{dataname}] TRAIN start on GPU {gpu}")
    rc = run(["python", "-u", "main.py",
              "--dataname", dataname, "--mode", "train", "--gpu", str(gpu),
              "--config_path", cfg, "--exp_name", dataname, "--no_wandb", "--resume"],
             train_log)
    rec["train_sec"] = time.time() - t0
    rec["train_rc"] = rc
    log(f"<<< [{dataname}] TRAIN {'done' if rc==0 else f'FAILED rc={rc}'} on GPU {gpu} — {fmt_dur(rec['train_sec'])}")
    if rc != 0:
        with results_lock:
            results[dataname] = rec
        return

    # ---- Generate (reject) ----
    t1 = time.time()
    log(f">>> [{dataname}] GENERATE (reject) {N_FRAUD} fraud on GPU {gpu}")
    grc = run(["python", "-u", "generate_paysim_app2.py",
               "--dataname", dataname, "--gpu", str(gpu),
               "--n_fraud", str(N_FRAUD), "--mode", "reject",
               "--batch", str(SAMPLE_BATCH), "--max_seconds", str(GEN_MAX_SECONDS)],
              gen_log)
    rec["gen_sec"] = time.time() - t1
    rec["gen_rc"] = grc
    log(f"<<< [{dataname}] GENERATE {'done' if grc==0 else f'FAILED rc={grc}'} on GPU {gpu} — {fmt_dur(rec['gen_sec'])} (see {gen_log})")
    with results_lock:
        results[dataname] = rec


def gpu_worker(gpu, q):
    while True:
        try:
            split = q.get_nowait()
        except queue.Empty:
            return
        try:
            handle(split, gpu)
        finally:
            q.task_done()


def main():
    log(f"=== banksim Approach 2b: d1-only train->generate for splits {SPLITS} "
        f"on {N_GPUS} GPUs, target {N_FRAUD} fraud/split ===")
    overall = time.time()
    q = queue.Queue()
    for s in SPLITS:
        q.put(s)
    workers = [threading.Thread(target=gpu_worker, args=(g, q), daemon=True)
               for g in range(N_GPUS)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    summary = {
        "overall_start": datetime.fromtimestamp(overall).strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_clock": fmt_dur(time.time() - overall),
        "n_fraud_target": N_FRAUD,
        "per_split": results,
    }
    with open(TIMINGS, "w") as f:
        json.dump(summary, f, indent=2)
    log("=== ALL DONE ===")
    for name, r in sorted(results.items()):
        log(f"    {name}: train {fmt_dur(r.get('train_sec',0))} rc={r.get('train_rc')} | "
            f"gen {fmt_dur(r.get('gen_sec',0))} rc={r.get('gen_rc')}")
    log(f"    total wall-clock: {summary['total_wall_clock']}")


if __name__ == "__main__":
    main()
