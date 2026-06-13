"""Approach 2 training orchestrator for the car-insurance splits.

Trains all 5 splits x 4 dosage models = 20 TabDiff models, keeping all GPUs busy:
one worker per GPU pulls jobs from a shared queue. Each job uses its own
per-model TOML config (via --config_path), so parallel runs don't collide on the
global config file.

Logs:
  - per-model start/finish timestamps + duration
  - per-split wall-clock (first model start -> last model finish for that split)
  - total wall-clock for all splits
  written to logs/ci_app2/timings.json and streamed to logs/ci_app2/run.log.

Run prepare_ci_app2.py first.
"""
import os
import json
import time
import queue
import threading
import subprocess
from datetime import datetime

N_SPLITS = 5
DOSES    = ["d1", "d2", "d3", "d4"]
GPUS     = [0, 1, 2, 3]

LOGDIR = "logs/ci_app2"
os.makedirs(LOGDIR, exist_ok=True)
RUNLOG   = f"{LOGDIR}/run.log"
TIMINGS  = f"{LOGDIR}/timings.json"

_log_lock = threading.Lock()


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


results = {}            # name -> dict(start, finish, duration, rc, gpu)
results_lock = threading.Lock()


def run_job(name, gpu):
    cfg = f"tabdiff/configs/tabdiff_configs_{name}.toml"
    train_log = f"{LOGDIR}/train_{name}.log"
    start = time.time()
    log(f">>> [{name}] TRAIN start on GPU {gpu}")
    with open(train_log, "a") as lf:
        lf.write(f"\n===== run start {ts()} (GPU {gpu}) =====\n")
        lf.flush()
        rc = subprocess.call(
            ["python", "-u", "main.py",
             "--dataname", name,
             "--mode", "train",
             "--gpu", str(gpu),
             "--config_path", cfg,
             "--exp_name", name,
             "--no_wandb",
             "--resume"],
            stdout=lf, stderr=subprocess.STDOUT,
        )
    finish = time.time()
    dur = finish - start
    with results_lock:
        results[name] = {
            "start": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            "finish": datetime.fromtimestamp(finish).strftime("%Y-%m-%d %H:%M:%S"),
            "start_epoch": start,
            "finish_epoch": finish,
            "duration_sec": dur,
            "rc": rc,
            "gpu": gpu,
        }
    status = "done" if rc == 0 else f"FAILED rc={rc}"
    log(f"<<< [{name}] TRAIN {status} on GPU {gpu} — {fmt_dur(dur)} (see {train_log})")
    return rc


def worker(gpu, jobq):
    while True:
        try:
            name = jobq.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(name, gpu)
        finally:
            jobq.task_done()


def main():
    jobs = [f"ci_s{s}_{d}" for s in range(N_SPLITS) for d in DOSES]
    log(f"=== Approach 2 training: {len(jobs)} models across GPUs {GPUS} ===")
    log(f"    jobs: {', '.join(jobs)}")

    jobq = queue.Queue()
    for j in jobs:
        jobq.put(j)

    overall_start = time.time()
    threads = [threading.Thread(target=worker, args=(g, jobq), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    overall_finish = time.time()

    # Per-split wall-clock = earliest model start -> latest model finish for that split.
    per_split = {}
    for s in range(N_SPLITS):
        names = [f"ci_s{s}_{d}" for d in DOSES if f"ci_s{s}_{d}" in results]
        if not names:
            continue
        st = min(results[n]["start_epoch"] for n in names)
        fi = max(results[n]["finish_epoch"] for n in names)
        sum_train = sum(results[n]["duration_sec"] for n in names)
        per_split[f"split_{s}"] = {
            "wall_start": datetime.fromtimestamp(st).strftime("%Y-%m-%d %H:%M:%S"),
            "wall_finish": datetime.fromtimestamp(fi).strftime("%Y-%m-%d %H:%M:%S"),
            "wall_clock_sec": fi - st,
            "wall_clock": fmt_dur(fi - st),
            "sum_model_train_sec": sum_train,
            "models": {n: {k: results[n][k] for k in ("gpu", "duration_sec", "rc")} for n in names},
        }

    summary = {
        "overall_start": datetime.fromtimestamp(overall_start).strftime("%Y-%m-%d %H:%M:%S"),
        "overall_finish": datetime.fromtimestamp(overall_finish).strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_clock_sec": overall_finish - overall_start,
        "total_wall_clock": fmt_dur(overall_finish - overall_start),
        "sum_all_model_train_sec": sum(r["duration_sec"] for r in results.values()),
        "n_models": len(results),
        "n_failed": sum(1 for r in results.values() if r["rc"] != 0),
        "per_split": per_split,
        "per_model": results,
    }
    with open(TIMINGS, "w") as f:
        json.dump(summary, f, indent=2)

    log("=== TRAINING COMPLETE ===")
    for s in range(N_SPLITS):
        k = f"split_{s}"
        if k in per_split:
            log(f"    split {s}: wall {per_split[k]['wall_clock']} "
                f"({per_split[k]['wall_start']} -> {per_split[k]['wall_finish']})")
    log(f"    TOTAL wall-clock (all splits): {summary['total_wall_clock']}")
    log(f"    sum of model train times:      {fmt_dur(summary['sum_all_model_train_sec'])}")
    log(f"    failed models: {summary['n_failed']}/{summary['n_models']}")
    log(f"    timings -> {TIMINGS}")


if __name__ == "__main__":
    main()
