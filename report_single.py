"""Detailed real-vs-synthetic fraud comparison for ONE split+dose.

Usage: python report_single.py <split_id> <dose>      e.g.  python report_single.py 3 d1

Produces a markdown report + figures covering:
  FEATURE-WISE (marginals): per-column stats + KS (numeric) / TVD (categorical),
                            validity checks, histograms & bar charts.
  JOINT:                    correlation matrices + diff, balance-identity check,
                            pairwise scatter/2D-hist, C2ST (joint distinguishability),
                            DCR (joint nearest-neighbour / memorization).
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SPLIT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DOSE     = sys.argv[2] if len(sys.argv) > 2 else "d1"
PFX      = f"ps{SPLIT_ID}"
NAME     = f"{PFX}_{DOSE}"
SPLIT    = f"/home/amad/projects/datasets/paysim/splits/{SPLIT_ID}"
DROP     = ["nameOrig", "nameDest"]
NUM      = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CAT      = ["type", "isFlaggedFraud"]
MONEY    = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FEATS    = NUM + CAT
OUT      = f"eval/paysim_app2_{PFX}/report_{NAME}"
os.makedirs(OUT, exist_ok=True)


def load_real(p):
    df = pd.read_csv(p).drop(columns=DROP)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(int).astype(str)
    df["type"] = df["type"].astype(str)
    return df


def main():
    syn_path = glob.glob(f"tabdiff/synthetic_fraud/{NAME}/fraud_*.csv")[0]
    syn = pd.read_csv(syn_path)
    syn["isFlaggedFraud"] = syn["isFlaggedFraud"].round().astype(int).astype(str)
    syn["type"] = syn["type"].astype(str)

    train = load_real(f"{SPLIT}/train.csv")
    test = load_real(f"{SPLIT}/test.csv")
    real = train[train["isFraud"] == 1].reset_index(drop=True)         # generator source
    real_test = test[test["isFraud"] == 1].reset_index(drop=True)      # held-out

    R = "\n".join
    rep = [f"# Real vs Synthetic Fraud — split {SPLIT_ID}, model {DOSE} ({NAME})\n",
           f"Source synthetic: `{syn_path}` ({len(syn):,} rows)  ",
           f"Real fraud (train): {len(real):,} rows | real fraud (test, held-out): {len(real_test):,}\n"]

    # ---------------- FEATURE-WISE ----------------
    rep.append("## 1. Feature-wise (marginal) comparison\n")
    rep.append("### 1a. Numerical columns — distribution stats + KS\n")
    rep.append("| column | source | mean | std | min | p25 | median | p75 | max | KS | KS p |")
    rep.append("|" + "---|"*11)
    ks_vals = {}
    for c in NUM:
        r = real[c].astype(float); s = syn[c].astype(float)
        ks = ks_2samp(r, s); ks_vals[c] = ks.statistic
        for lab, x in [("real", r), ("syn", s)]:
            q = x.quantile([.25, .5, .75])
            extra = f"{ks.statistic:.3f} | {ks.pvalue:.1e}" if lab == "real" else " | "
            rep.append(f"| {c if lab=='real' else ''} | {lab} | {x.mean():,.0f} | {x.std():,.0f} | "
                       f"{x.min():,.0f} | {q[.25]:,.0f} | {q[.5]:,.0f} | {q[.75]:,.0f} | {x.max():,.0f} | {extra} |")
    rep.append(f"\n*KS = Kolmogorov–Smirnov distance (0 = identical). max KS = **{max(ks_vals.values()):.3f}**.*\n")

    rep.append("### 1b. Categorical columns — proportions + TVD\n")
    tvd_vals = {}
    for c in CAT:
        rv = real[c].value_counts(normalize=True); sv = syn[c].value_counts(normalize=True)
        keys = sorted(set(rv.index) | set(sv.index))
        tvd = 0.5 * sum(abs(rv.get(k, 0) - sv.get(k, 0)) for k in keys); tvd_vals[c] = tvd
        rep.append(f"**{c}** (TVD={tvd:.3f}):  " +
                   ";  ".join(f"`{k}` real {rv.get(k,0)*100:.1f}% / syn {sv.get(k,0)*100:.1f}%" for k in keys))
        rep.append("")

    rep.append("### 1c. Validity\n")
    purity = (syn["isFraud"].astype(str) == "1").mean()
    type_ok = syn["type"].isin({"TRANSFER", "CASH_OUT"}).mean()
    neg = (syn["amount"].astype(float) < 0).mean()
    rep.append(f"- isFraud==1 purity: **{purity*100:.2f}%**")
    rep.append(f"- fraud-plausible type (TRANSFER/CASH_OUT): **{type_ok*100:.1f}%** (real fraud = 100%)")
    rep.append(f"- negative amounts: {neg*100:.3f}%\n")
    rep.append("**Numerical marginals (real=blue, synthetic=red):**\n")
    rep.append("![numerical marginals](marginals.png)\n")
    rep.append("**Categorical distributions:**\n")
    rep.append("![categorical distributions](categoricals.png)\n")

    # numeric marginal figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), NUM):
        r = real[c].astype(float).values; s = syn[c].astype(float).values
        if c in MONEY:
            r, s = np.log1p(r), np.log1p(s); ax.set_xlabel(f"log1p({c})")
        else:
            ax.set_xlabel(c)
        b = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 60)
        ax.hist(r, b, density=True, alpha=.55, label="real", color="#1f77b4")
        ax.hist(s, b, density=True, alpha=.55, label="syn", color="#d62728")
        ax.legend(fontsize=8)
    fig.suptitle(f"{NAME}: numerical marginals"); fig.tight_layout()
    fig.savefig(f"{OUT}/marginals.png", dpi=90); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, c in zip(axes, CAT):
        rv = real[c].value_counts(normalize=True); sv = syn[c].value_counts(normalize=True)
        keys = sorted(set(rv.index) | set(sv.index)); x = np.arange(len(keys))
        ax.bar(x-.2, [rv.get(k,0) for k in keys], .4, label="real", color="#1f77b4")
        ax.bar(x+.2, [sv.get(k,0) for k in keys], .4, label="syn", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(c); ax.legend(fontsize=8)
    fig.suptitle(f"{NAME}: categorical distributions"); fig.tight_layout()
    fig.savefig(f"{OUT}/categoricals.png", dpi=90); plt.close(fig)

    # ---------------- JOINT ----------------
    rep.append("## 2. Joint / multivariate comparison\n")
    rc = real[NUM].astype(float).corr(); sc = syn[NUM].astype(float).corr()
    corr_diff = (sc - rc).abs()
    rep.append(f"### 2a. Correlation structure\n")
    rep.append(f"Mean absolute correlation difference: **{corr_diff.values[np.triu_indices(len(NUM),1)].mean():.3f}** "
               f"(max pair diff {corr_diff.values[np.triu_indices(len(NUM),1)].max():.3f}). See `correlation.png`.\n")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m, t, cm, vmin in zip(axes, [rc, sc, corr_diff], ["real", "synthetic", "|diff|"],
                                  ["coolwarm","coolwarm","Reds"], [-1,-1,0]):
        im = ax.imshow(m, vmin=vmin, vmax=1, cmap=cm)
        ax.set_xticks(range(len(NUM))); ax.set_xticklabels(NUM, rotation=90, fontsize=7)
        ax.set_yticks(range(len(NUM))); ax.set_yticklabels(NUM, fontsize=7)
        ax.set_title(t); fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(f"{NAME}: correlation"); fig.tight_layout()
    fig.savefig(f"{OUT}/correlation.png", dpi=90); plt.close(fig)

    # balance identity (paysim): for TRANSFER/CASH_OUT, newbalanceOrig ≈ oldbalanceOrg - amount
    def bal_ok(df):
        d = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
        if not len(d): return float("nan")
        lhs = d["oldbalanceOrg"].astype(float) - d["amount"].astype(float)
        return (np.abs(lhs - d["newbalanceOrig"].astype(float)) <= 1.0).mean()
    rep.append("### 2b. Balance identity (origin side)\n")
    rep.append("Fraction of TRANSFER/CASH_OUT rows satisfying `newbalanceOrig ≈ oldbalanceOrg − amount` (±1):")
    rep.append(f"- real: **{bal_ok(real)*100:.1f}%**  |  synthetic: **{bal_ok(syn)*100:.1f}%**\n")

    # joint scatter for key pairs
    pairs = [("oldbalanceOrg", "amount"), ("oldbalanceOrg", "newbalanceOrig"), ("oldbalanceDest", "newbalanceDest")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    rs = real.sample(min(4000, len(real)), random_state=0); ss = syn.sample(4000, random_state=0)
    for j, (xa, ya) in enumerate(pairs):
        for i, (src, df, col) in enumerate([("real", rs, "#1f77b4"), ("syn", ss, "#d62728")]):
            ax = axes[i, j]
            ax.scatter(np.log1p(df[xa].astype(float)), np.log1p(df[ya].astype(float)), s=4, alpha=.3, color=col)
            ax.set_xlabel(f"log1p({xa})"); ax.set_ylabel(f"log1p({ya})"); ax.set_title(f"{src}: {xa} vs {ya}", fontsize=9)
    fig.suptitle(f"{NAME}: joint pairwise structure (top=real, bottom=synthetic)"); fig.tight_layout()
    fig.savefig(f"{OUT}/joint_scatter.png", dpi=90); plt.close(fig)

    # C2ST (joint distinguishability)
    n = min(len(real), len(syn), 6570)
    X = pd.get_dummies(pd.concat([real.sample(n, random_state=1)[FEATS],
                                  syn.sample(n, random_state=1)[FEATS]], ignore_index=True), columns=CAT)
    y = np.r_[np.zeros(n), np.ones(n)]
    idx = np.random.RandomState(0).permutation(len(X)); cut = int(.7*len(X))
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X.iloc[idx[:cut]], y[idx[:cut]])
    auc = roc_auc_score(y[idx[cut:]], clf.predict_proba(X.iloc[idx[cut:]])[:, 1])

    # DCR (joint nearest-neighbour, standardized numerics)
    std = real[NUM].astype(float).std().replace(0, 1).values
    nn = NearestNeighbors(n_neighbors=1).fit(real[NUM].astype(float).values / std)
    syn_d = nn.kneighbors(syn.sample(min(5000, len(syn)), random_state=0)[NUM].astype(float).values / std)[0][:, 0]
    test_d = nn.kneighbors(real_test[NUM].astype(float).values / std)[0][:, 0]
    base = np.median(test_d); med = np.median(syn_d)
    fig, ax = plt.subplots(figsize=(7, 4))
    hi = np.percentile(np.r_[syn_d, test_d], 99); b = np.linspace(0, hi, 50)
    ax.hist(test_d, b, density=True, alpha=.55, label="real test→train (baseline)", color="#2ca02c")
    ax.hist(syn_d, b, density=True, alpha=.55, label="synthetic→train", color="#d62728")
    ax.axvline(base, color="#2ca02c", ls="--"); ax.axvline(med, color="#d62728", ls="--")
    ax.set_xlabel("distance to closest real train fraud (standardized)"); ax.legend(fontsize=8)
    ax.set_title(f"{NAME}: DCR"); fig.tight_layout(); fig.savefig(f"{OUT}/dcr.png", dpi=90); plt.close(fig)

    rep.append("### 2c. Joint distinguishability & privacy\n")
    rep.append(f"- **C2ST AUC = {auc:.3f}** — a gradient-boosted classifier's ability to tell real fraud from "
               f"synthetic on all features jointly. 0.5 = indistinguishable (ideal); 1.0 = trivially separable.")
    rep.append(f"- **DCR**: median synthetic→train distance **{med:.4f}** vs real-test→train baseline **{base:.4f}** "
               f"(ratio {med/base:.2f}). ~1 = realistic & not memorized; ≪1 = memorization. See `dcr.png`.\n")

    # ---------------- verdict ----------------
    rep.append("## 3. Verdict\n")
    rep.append(f"- Marginals: max KS **{max(ks_vals.values()):.3f}**, max TVD **{max(tvd_vals.values()):.3f}**")
    rep.append(f"- Joint: mean |corr diff| **{corr_diff.values[np.triu_indices(len(NUM),1)].mean():.3f}**, "
               f"C2ST **{auc:.3f}**, DCR ratio **{med/base:.2f}**")
    rep.append(f"- Validity: type-valid **{type_ok*100:.1f}%**, purity **{purity*100:.1f}%**\n")
    rep.append("Figures: `marginals.png`, `categoricals.png`, `correlation.png`, `joint_scatter.png`, `dcr.png`")

    with open(f"{OUT}/REPORT.md", "w") as f:
        f.write(R(rep))
    print(R(rep))
    print(f"\n--> {OUT}/REPORT.md")


if __name__ == "__main__":
    main()
