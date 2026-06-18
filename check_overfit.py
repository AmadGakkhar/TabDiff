"""Overfitting / memorization check for the GMSC synthetic positives.

For each split we compare the synthetic rows against the REAL training positives:

  1. exact-duplicate rate    — fraction of synthetic rows that exactly equal a real row
  2. DCR (Distance to Closest Record) — nearest-neighbour distance (z-scored, Euclidean)
     from each synthetic row to the real set, vs a real->real baseline (holdout half).
     A memorizing model gives DCR_syn ~ 0 and << DCR_real. Healthy: DCR_syn comparable
     to DCR_real (the model is no closer to the training data than real points are to
     each other).
  3. marginal drift          — max abs difference of per-column mean/std (z-scored).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

TARGET = "SeriousDlqin2yrs"


def dcr(query, ref):
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    d, _ = nn.kneighbors(query)
    return d[:, 0]


for s in range(5):
    real_path = f"data/gmsc_l{s}/gmsc_l{s}.csv"
    syn_path  = f"/srv/datasets/GMSC/splits/level_{s}/synthetic_positives_5x.csv"
    try:
        real = pd.read_csv(real_path)
        syn  = pd.read_csv(syn_path)
    except FileNotFoundError:
        print(f"level_{s}: (output not ready yet)")
        continue

    feat = [c for c in real.columns if c != TARGET]
    R = real[feat].astype(float).to_numpy()
    S = syn[feat].astype(float).to_numpy()

    # z-score using real stats
    mu, sd = R.mean(0), R.std(0) + 1e-9
    Rz, Sz = (R - mu) / sd, (S - mu) / sd

    # exact duplicates (round to 6 dp to avoid float noise)
    rset = set(map(tuple, np.round(R, 6)))
    dup = sum(t in rset for t in map(tuple, np.round(S, 6)))

    # DCR: split real in half -> baseline B->A, then S->A
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(Rz))
    A, B = Rz[idx[: len(Rz)//2]], Rz[idx[len(Rz)//2:]]
    dcr_real = np.median(dcr(B, A))
    dcr_syn  = np.median(dcr(Sz, A))

    drift_mean = np.abs(Sz.mean(0)).max()
    drift_std  = np.abs(Sz.std(0) - 1).max()

    print(f"level_{s}: real_pos={len(R):>5}  syn={len(S):>6}  "
          f"exact_dup={dup} ({100*dup/len(S):.2f}%)  "
          f"DCR_syn={dcr_syn:.3f} vs DCR_real={dcr_real:.3f} (ratio {dcr_syn/dcr_real:.2f})  "
          f"max|mean_z|={drift_mean:.3f} max|std_z-1|={drift_std:.3f}")
