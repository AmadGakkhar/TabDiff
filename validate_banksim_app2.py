"""Validate banksim d1 synthetic fraud: memorization + fidelity.

For each split S:
  real fraud  = fraud rows from datasets/banksim/splits_temporal/S/train.csv
                (same columns dropped as in prepare; this is what the model saw)
  synthetic   = tabdiff/synthetic_fraud/bsS_d1/fraud_100000.csv

Metrics
  - duplicate_rate     : 1 - unique(synth)/len(synth)            (self-repetition)
  - memorized_rate     : frac of synth rows that EXACTLY match a real train row
  - dcr_zero_rate      : frac of synth (sampled) with distance-to-closest-real == 0
  - dcr_median         : median normalized distance-to-closest-real (mixed metric)
  - cat TV distance    : per categorical column, total-variation dist real vs synth
  - num mean/std drift : per numerical column, |mean| & std ratio real vs synth
A healthy model: low duplicate/memorized/dcr_zero, small TV, num stats close.
"""
import os
import json
import numpy as np
import pandas as pd

SPLITS_DIR = "/home/amad/projects/datasets/banksim/splits_temporal"
DROP_COLS  = ["customer", "zipcodeOri", "zipMerchant"]
TARGET     = "fraud"
NUM_COLS   = ["step", "amount"]
CAT_COLS   = ["age", "gender", "merchant", "category"]
DCR_SAMPLE = 5000   # synth rows sampled for the (O(n*m)) DCR computation


def load_real_fraud(s):
    df = pd.read_csv(f"{SPLITS_DIR}/{s}/train.csv").drop(columns=DROP_COLS)
    df = df[df[TARGET] == 1].reset_index(drop=True)
    for c in CAT_COLS + [TARGET]:
        df[c] = df[c].astype(str)
    return df


def load_synth(s):
    df = pd.read_csv(f"tabdiff/synthetic_fraud/bs{s}_d1/fraud_100000.csv")
    for c in CAT_COLS + [TARGET]:
        df[c] = df[c].astype(str)
    return df


def encode_mixed(real, synth):
    """Normalised numerics (by real min-max) + one-hot-free integer-coded cats for DCR."""
    rn = real[NUM_COLS].to_numpy(float)
    sn = synth[NUM_COLS].to_numpy(float)
    lo, hi = rn.min(0), rn.max(0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    rn = (rn - lo) / rng
    sn = (sn - lo) / rng
    # categoricals -> codes from real categories; unseen -> -1 (always mismatches)
    rc, sc = [], []
    for c in CAT_COLS:
        cats = {v: i for i, v in enumerate(real[c].unique())}
        rc.append(real[c].map(cats).to_numpy())
        sc.append(synth[c].map(lambda v: cats.get(v, -1)).to_numpy())
    rc = np.stack(rc, 1).astype(float)
    sc = np.stack(sc, 1).astype(float)
    return rn, rc, sn, sc


def dcr(real, synth, rng):
    rn, rc, sn, sc = encode_mixed(real, synth)
    idx = rng.choice(len(synth), size=min(DCR_SAMPLE, len(synth)), replace=False)
    sn, sc = sn[idx], sc[idx]
    n_cat = rc.shape[1]
    mins = np.empty(len(sn))
    for i in range(len(sn)):
        num_d = np.abs(rn - sn[i]).sum(1)               # L1 on normalized numerics
        cat_d = (rc != sc[i]).sum(1)                    # Hamming on categoricals
        d = (num_d + cat_d) / (len(NUM_COLS) + n_cat)   # normalized to [0,~1]
        mins[i] = d.min()
    return float((mins == 0).mean()), float(np.median(mins))


def tv_distance(real, synth, col):
    rp = real[col].value_counts(normalize=True)
    sp = synth[col].value_counts(normalize=True)
    keys = set(rp.index) | set(sp.index)
    return 0.5 * sum(abs(rp.get(k, 0.0) - sp.get(k, 0.0)) for k in keys)


def main():
    rng = np.random.RandomState(0)
    report = {}
    for s in range(5):
        synth_path = f"tabdiff/synthetic_fraud/bs{s}_d1/fraud_100000.csv"
        if not os.path.exists(synth_path):
            print(f"bs{s}: no synthetic output, skipping")
            continue
        real = load_real_fraud(s)
        synth = load_synth(s)

        n = len(synth)
        n_unique = synth.drop_duplicates().shape[0]
        dup_rate = 1 - n_unique / n

        merged = synth.merge(real.drop_duplicates(), on=list(synth.columns), how="left", indicator=True)
        memo_rate = (merged["_merge"] == "both").mean()

        dcr_zero, dcr_med = dcr(real, synth, rng)

        cat_tv = {c: round(tv_distance(real, synth, c), 4) for c in CAT_COLS}
        num_stats = {}
        for c in NUM_COLS:
            r, sy = real[c].astype(float), synth[c].astype(float)
            num_stats[c] = {
                "real_mean": round(r.mean(), 3), "synth_mean": round(sy.mean(), 3),
                "real_std": round(r.std(), 3),   "synth_std": round(sy.std(), 3),
            }

        report[f"bs{s}_d1"] = {
            "real_fraud_rows": len(real),
            "real_unique_rows": int(real.drop_duplicates().shape[0]),
            "synth_rows": n,
            "synth_unique_rows": int(n_unique),
            "duplicate_rate": round(dup_rate, 4),
            "memorized_rate": round(float(memo_rate), 4),
            "dcr_zero_rate": round(dcr_zero, 4),
            "dcr_median": round(dcr_med, 4),
            "cat_TV_distance": cat_tv,
            "num_stats": num_stats,
        }
        r = report[f"bs{s}_d1"]
        print(f"\n=== bs{s}_d1 === real_fraud={len(real)} (uniq {r['real_unique_rows']})")
        print(f"  duplicate_rate={r['duplicate_rate']:.3f}  memorized_rate={r['memorized_rate']:.3f}  "
              f"dcr_zero={r['dcr_zero_rate']:.3f}  dcr_median={r['dcr_median']:.3f}")
        print(f"  cat_TV={cat_tv}")
        print(f"  num_stats={num_stats}")

    out = "logs/banksim_app2_d1/validation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
