"""Sanity / memorization check for paysim Approach-2 synthetic fraud sets.

For each synthetic fraud CSV, compare against the REAL fraud rows:
  - real TRAIN fraud (what the generator learned from) -> memorization reference
  - real TEST fraud (held out)                         -> generalization reference

Reports per set:
  - shape, NaN rows, isFraud purity
  - numerical column ranges (min/mean/max) vs real train fraud
  - categorical distributions (type, isFlaggedFraud) vs real train fraud
  - within-synthetic exact-duplicate rate (feature cols)
  - exact matches to real train fraud (hard memorization)
  - DCR: median nearest-neighbour distance (standardized numerics) of a synthetic
    sample to real train fraud, vs the test->train baseline. synthetic >= baseline
    is healthy; synthetic << baseline signals memorization.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SPLIT = "/home/amad/projects/datasets/paysim/splits/0"
DROP = ["nameOrig", "nameDest"]
NUM = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CAT = ["type", "isFlaggedFraud"]
FEATS = NUM + CAT
SETS = {
    "d1": "tabdiff/synthetic_fraud/ps0_d1/fraud_1600000.csv",
    "d2_v1": "tabdiff/synthetic_fraud/ps0_d2/fraud_1600000.v1_400ep.csv",
    "d3": "tabdiff/synthetic_fraud/ps0_d3/fraud_1600000.csv",
}


def real_fraud(path):
    df = pd.read_csv(path).drop(columns=DROP)
    return df[df["isFraud"] == 1].reset_index(drop=True)


def dcr(sample_num, ref_num, std):
    a = sample_num / std
    b = ref_num / std
    nn = NearestNeighbors(n_neighbors=1).fit(b)
    d, _ = nn.kneighbors(a)
    return np.median(d[:, 0])


def main():
    train_f = real_fraud(f"{SPLIT}/train.csv")
    test_f = real_fraud(f"{SPLIT}/test.csv")
    print(f"real fraud: train={len(train_f)} test={len(test_f)}")

    std = train_f[NUM].astype(float).std().replace(0, 1).values
    base = dcr(test_f[NUM].astype(float).values, train_f[NUM].astype(float).values, std)
    print(f"baseline DCR (real test fraud -> real train fraud): {base:.4f}\n")

    for name, path in SETS.items():
        print("=" * 70)
        print(f"### {name}  ({path})")
        try:
            syn = pd.read_csv(path)
        except FileNotFoundError:
            print("  MISSING — skipped")
            continue
        n = len(syn)
        nan_rows = syn.isna().any(axis=1).sum()
        purity = (syn["isFraud"].astype(str) == "1").mean()
        print(f"  rows={n}  nan_rows={nan_rows}  isFraud==1 purity={purity:.4f}")

        print("  numerical (syn min/mean/max  |  real-train min/mean/max):")
        for c in NUM:
            s = syn[c].astype(float); r = train_f[c].astype(float)
            print(f"    {c:16s} {s.min():14.1f}{s.mean():14.1f}{s.max():14.1f}  | "
                  f"{r.min():12.1f}{r.mean():12.1f}{r.max():12.1f}")

        print("  categorical (syn% vs real-train%):")
        for c in CAT:
            sv = syn[c].astype(str).value_counts(normalize=True)
            rv = train_f[c].astype(str).value_counts(normalize=True)
            keys = sorted(set(sv.index) | set(rv.index))
            parts = [f"{k}:{sv.get(k,0)*100:.1f}/{rv.get(k,0)*100:.1f}" for k in keys]
            print(f"    {c:16s} " + "  ".join(parts))

        dup_syn = syn.duplicated(subset=FEATS).mean()
        merged = syn.merge(train_f[FEATS].drop_duplicates(), on=FEATS, how="inner")
        mem = len(merged) / n
        print(f"  within-syn exact-dup rate: {dup_syn:.4f}")
        print(f"  exact matches to real train fraud: {mem:.4f}")

        k = min(5000, n)
        samp = syn.sample(k, random_state=0)
        d = dcr(samp[NUM].astype(float).values, train_f[NUM].astype(float).values, std)
        flag = "  <-- MEMORIZATION RISK" if d < 0.5 * base else ""
        print(f"  synthetic DCR (sample {k} -> real train fraud): {d:.4f}  "
              f"(baseline {base:.4f}){flag}")
    print("=" * 70)


if __name__ == "__main__":
    main()
