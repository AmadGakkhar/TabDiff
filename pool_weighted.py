"""Quality-weighted pooling of per-model synthetic fraud (Approach 2).

Instead of taking an equal 25% from each dosage model, this scores every
model's synthetic fraud on intrinsic DATA QUALITY, converts the scores into
mixing weights, and samples `n_total` rows proportionally.

Quality is a geometric mean of four sub-scores, each in [0, 1] (higher = better):

  fidelity      how closely the synthetic fraud matches the REAL fraud
                distribution — per-column marginals (Wasserstein for numeric,
                total-variation for categorical) plus correlation structure.

  coverage      what fraction of real-fraud points have a synthetic neighbour
                within their local k-NN radius (manifold recall). Detects mode
                collapse: a model that only reproduces a few fraud "modes"
                covers little of the real fraud manifold.

  authenticity  1 - (share of synthetic rows that are near-duplicates of a
                TRAINING fraud row), via distance-to-closest-record. Penalises
                memorisation: a model that copies the 738 real fraud rows adds
                no new information even though its fidelity looks perfect.

  alignment     mean P(fraud) assigned to the synthetic rows by a classifier
                trained on REAL fraud-vs-non-fraud. Checks the rows actually
                land on the fraud side of the real decision boundary.

The geometric mean is deliberate: a model that fails ANY single axis (e.g.
perfect fidelity but memorised, so low authenticity) is driven down, which is
exactly the behaviour we want for augmentation data — we reward *useful
novelty*, not mimicry.

Usage:
  python pool_weighted.py \
      --reservoir_dir synthetic_outputs/expv01_app2/reservoirs \
      --real_csv      data/exp-v01/experiment2_real_train.csv \
      --out           synthetic_outputs/expv01_app2/pooled_fraud_weighted.csv \
      --n_total       2500
"""
import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── exp-v01 schema (32 cols after dropping PolicyNumber) ──────────────────────
DROP_COLS      = ["PolicyNumber"]
TARGET         = "FraudFound_P"
NUM_COL_IDX    = [1, 7, 10, 16, 17, 18, 30]
TARGET_COL_IDX = 15
EPS            = 1e-6


# ── distance helpers ──────────────────────────────────────────────────────────
def gower(A_num, A_cat, B_num, B_cat, ranges):
    """Gower distance matrix, rows=A, cols=B, values in [0, 1].

    Numeric features contribute |a-b| / range; categorical contribute 0/1;
    averaged over all features.
    """
    n_a, n_b = A_num.shape[0], B_num.shape[0]
    n_feat = A_num.shape[1] + A_cat.shape[1]
    D = np.zeros((n_a, n_b), dtype=np.float32)
    for j in range(A_num.shape[1]):
        rng = ranges[j] if ranges[j] > 0 else 1.0
        D += (np.abs(A_num[:, [j]] - B_num[:, j][None, :]) / rng).astype(np.float32)
    for j in range(A_cat.shape[1]):
        D += (A_cat[:, [j]] != B_cat[:, j][None, :]).astype(np.float32)
    D /= n_feat
    return D


def split_types(df, num_cols, cat_cols):
    num = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cat = df[cat_cols].astype(str).to_numpy()
    return num, cat


# ── sub-scores ────────────────────────────────────────────────────────────────
def fidelity_score(real_num, real_cat, syn_num, syn_cat, ranges):
    # marginals
    per_col = []
    for j in range(real_num.shape[1]):
        rng = ranges[j] if ranges[j] > 0 else 1.0
        w = wasserstein_distance(real_num[:, j], syn_num[:, j]) / rng
        per_col.append(1.0 - min(w, 1.0))
    for j in range(real_cat.shape[1]):
        cats = np.unique(np.concatenate([real_cat[:, j], syn_cat[:, j]]))
        pr = np.array([(real_cat[:, j] == c).mean() for c in cats])
        ps = np.array([(syn_cat[:, j] == c).mean() for c in cats])
        tv = 0.5 * np.abs(pr - ps).sum()
        per_col.append(1.0 - tv)
    f_marg = float(np.mean(per_col))

    # correlation structure (numeric only)
    if real_num.shape[1] >= 2:
        cr = np.corrcoef(real_num, rowvar=False)
        cs = np.corrcoef(syn_num, rowvar=False)
        cr = np.nan_to_num(cr); cs = np.nan_to_num(cs)
        denom = np.linalg.norm(cr) or 1.0
        f_corr = 1.0 - min(np.linalg.norm(cs - cr) / denom, 1.0)
    else:
        f_corr = f_marg
    return float(np.clip(0.5 * (f_marg + f_corr), EPS, 1.0)), f_marg, f_corr


def coverage_score(D_ff, D_sf, k):
    """Fraction of real-fraud points covered by >=1 synthetic neighbour."""
    n_f = D_ff.shape[0]
    dff = D_ff.copy()
    np.fill_diagonal(dff, np.inf)
    kth = np.sort(dff, axis=1)[:, min(k, n_f - 1) - 1]   # k-th NN radius per real point
    nearest_syn = D_sf.min(axis=0)                       # closest synthetic to each real point
    return float(np.clip((nearest_syn <= kth).mean(), EPS, 1.0))


def authenticity_score(D_ff, D_sf):
    """1 - share of synthetic rows that are near-duplicates of a training fraud row."""
    dff = D_ff.copy()
    np.fill_diagonal(dff, np.inf)
    real_nn = dff.min(axis=1)                # real fraud internal 1-NN distances
    tau = 0.5 * np.median(real_nn)           # memorisation threshold
    dcr = D_sf.min(axis=1)                   # distance to closest real fraud
    return float(np.clip((dcr >= tau).mean(), EPS, 1.0)), float(np.median(dcr)), float(tau)


def alignment_score(clf, syn_df, feat_cols):
    p = clf.predict_proba(syn_df[feat_cols])[:, 1]
    return float(np.clip(p.mean(), EPS, 1.0))


# ── capped proportional allocation (water-filling) ───────────────────────────
def allocate(weights, avail, n_total):
    w = np.array(weights, dtype=float)
    avail = np.array(avail, dtype=int)
    alloc = np.zeros(len(w), dtype=int)
    active = avail > 0
    remaining = n_total
    while remaining > 0 and active.any():
        wa = w * active
        if wa.sum() == 0:
            break
        give = np.floor(wa / wa.sum() * remaining).astype(int)
        give = np.minimum(give, avail - alloc)
        if give.sum() == 0:                       # rounding stalled — hand out singles
            order = np.argsort(-(wa))
            for idx in order:
                if active[idx] and alloc[idx] < avail[idx]:
                    give[idx] = 1
                    break
        alloc += give
        remaining = n_total - alloc.sum()
        active = (avail - alloc) > 0
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoir_dir", default="synthetic_outputs/expv01_app2/reservoirs")
    ap.add_argument("--real_csv",      default="data/exp-v01/experiment2_real_train.csv")
    ap.add_argument("--out",           default="synthetic_outputs/expv01_app2/pooled_fraud_weighted.csv")
    ap.add_argument("--report",        default="synthetic_outputs/expv01_app2/weight_report.json")
    ap.add_argument("--models",        nargs="+", default=["expv01_d1", "expv01_d2", "expv01_d3", "expv01_d4"])
    ap.add_argument("--suffix",        default="_fraud.csv")
    ap.add_argument("--n_total",       type=int, default=2500)
    ap.add_argument("--k",             type=int, default=5, help="k-NN radius for coverage")
    ap.add_argument("--exponents",     nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0],
                    help="fidelity coverage authenticity alignment exponents")
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ── real data ────────────────────────────────────────────────────────────
    real = pd.read_csv(args.real_csv).drop(columns=DROP_COLS)
    cols = real.columns.tolist()
    assert cols[TARGET_COL_IDX] == TARGET, f"target mismatch: {cols[TARGET_COL_IDX]}"
    num_cols = [cols[i] for i in NUM_COL_IDX]
    cat_cols = [c for i, c in enumerate(cols) if i not in NUM_COL_IDX + [TARGET_COL_IDX]]
    feat_cols = num_cols + cat_cols

    real_fraud = real[real[TARGET] == 1].reset_index(drop=True)
    rf_num, rf_cat = split_types(real_fraud, num_cols, cat_cols)
    ranges = rf_num.max(axis=0) - rf_num.min(axis=0)

    # reference distances among real fraud (reused by every model)
    D_ff = gower(rf_num, rf_cat, rf_num, rf_cat, ranges)

    # fraud-vs-non-fraud classifier for the alignment score
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    clf = Pipeline([("pre", pre),
                    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    clf.fit(real[feat_cols], real[TARGET].astype(int))

    print(f"Real: {len(real)} rows | fraud={len(real_fraud)} | "
          f"{len(num_cols)} num + {len(cat_cols)} cat features\n")

    # ── per-model scoring ──────────────────────────────────────────────────────
    rows, scores, avail = [], [], []
    for m in args.models:
        path = os.path.join(args.reservoir_dir, f"{m}{args.suffix}")
        if not os.path.exists(path):                       # fall back to the 625-row outputs
            path = os.path.join("synthetic_outputs/expv01_app2", f"{m}{args.suffix}")
        syn = pd.read_csv(path)
        syn = syn[pd.to_numeric(syn[TARGET], errors="coerce") == 1].reset_index(drop=True)
        s_num, s_cat = split_types(syn, num_cols, cat_cols)

        D_sf = gower(s_num, s_cat, rf_num, rf_cat, ranges)
        fid, f_marg, f_corr = fidelity_score(rf_num, rf_cat, s_num, s_cat, ranges)
        cov = coverage_score(D_ff, D_sf, args.k)
        auth, dcr_med, tau = authenticity_score(D_ff, D_sf)
        ali = alignment_score(clf, syn, feat_cols)

        e = args.exponents
        q = float(np.exp((e[0]*np.log(fid) + e[1]*np.log(cov) +
                          e[2]*np.log(auth) + e[3]*np.log(ali)) / sum(e)))

        avail.append(len(syn))
        rows.append(syn)
        scores.append(dict(model=m, n_available=len(syn),
                           fidelity=round(fid, 4), f_marginal=round(f_marg, 4),
                           f_correlation=round(f_corr, 4), coverage=round(cov, 4),
                           authenticity=round(auth, 4), dcr_median=round(dcr_med, 4),
                           alignment=round(ali, 4), quality=round(q, 4)))

    Q = np.array([s["quality"] for s in scores])
    weights = Q / Q.sum()
    alloc = allocate(weights, avail, args.n_total)

    # ── sample + pool ──────────────────────────────────────────────────────────
    parts = []
    for s, df_m, a in zip(scores, rows, alloc):
        s["weight"] = round(float(weights[scores.index(s)]), 4)
        s["allocated"] = int(a)
        if a > 0:
            idx = rng.choice(len(df_m), size=a, replace=(a > len(df_m)))
            parts.append(df_m.iloc[idx])
    pool = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pool.to_csv(args.out, index=False)

    # ── report ─────────────────────────────────────────────────────────────────
    report = dict(n_total=int(pool.shape[0]), seed=args.seed, k=args.k,
                  exponents=dict(zip(["fidelity", "coverage", "authenticity", "alignment"], args.exponents)),
                  models=scores)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    hdr = f"{'model':<11}{'avail':>7}{'fidel':>8}{'cover':>8}{'authn':>8}{'align':>8}{'QUAL':>8}{'weight':>8}{'rows':>7}"
    print(hdr); print("-" * len(hdr))
    for s in scores:
        print(f"{s['model']:<11}{s['n_available']:>7}{s['fidelity']:>8.3f}{s['coverage']:>8.3f}"
              f"{s['authenticity']:>8.3f}{s['alignment']:>8.3f}{s['quality']:>8.3f}"
              f"{s['weight']:>8.3f}{s['allocated']:>7}")
    print(f"\nPooled {len(pool)} rows -> {args.out}")
    print(f"Report -> {args.report}")


if __name__ == "__main__":
    main()
