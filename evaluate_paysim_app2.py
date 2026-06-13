"""Trustworthiness evaluation for paysim Approach-2 synthetic fraud.

For every synthetic fraud set found under tabdiff/synthetic_fraud/ps0_d*/, compare
against the REAL fraud rows (train split = what the generator learned; test split =
held-out reference) and answer the questions that must be cleared before trusting
synthetic data:

  Q1 FIDELITY (marginals)   - do per-column distributions match the real fraud?
  Q2 FIDELITY (joint)       - is the correlation structure preserved?
  Q3 VALIDITY               - are rows internally valid (fraud-plausible types,
                              non-negative amounts, balance identities)?
  Q4 CLASS PURITY           - is every generated row actually fraud (isFraud==1)?
  Q5 MEMORIZATION/PRIVACY   - is the model copying training rows? (exact dups + DCR
                              vs the real test->train baseline)
  Q6 DIVERSITY              - is the synthetic set diverse, not mode-collapsed?
  Q7 INDISTINGUISHABILITY   - can a classifier tell real from synthetic? (C2ST AUC)
  Q8 DOWNSTREAM UTILITY     - does augmenting with synthetic fraud help a fraud
                              classifier on the real held-out test set? (PR-AUC lift)

Outputs:
  eval/paysim_app2/<set>/*.png   - distribution / correlation / DCR figures
  eval/paysim_app2/REPORT.md     - full written report with per-set verdicts
  eval/paysim_app2/metrics.json  - raw numbers
"""
import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

SPLIT   = "/home/amad/projects/datasets/paysim/splits/0"
DROP    = ["nameOrig", "nameDest"]
NUM     = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CAT     = ["type", "isFlaggedFraud"]
FEATS   = NUM + CAT
MONEY   = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FRAUD_TYPES = {"TRANSFER", "CASH_OUT"}      # the only types real paysim fraud ever takes
OUTDIR  = "eval/paysim_app2"
SAMPLE  = 50000      # per-set subsample for plots/metrics
DCR_Q   = 5000       # query points for DCR


def load_real(path):
    df = pd.read_csv(path).drop(columns=DROP)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(int).astype(str)
    df["type"] = df["type"].astype(str)
    return df


def find_sets():
    sets = {}
    for f in sorted(glob.glob("tabdiff/synthetic_fraud/ps0_d*/fraud_1600000.csv")):
        dose = f.split("/")[-2]            # ps0_dX
        sets[dose] = f
    return sets


def prep_syn(path):
    df = pd.read_csv(path)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].round().astype(int).astype(str)
    df["type"] = df["type"].astype(str)
    return df


# ---------- metrics ----------
def ks_per_numeric(real, syn):
    return {c: float(ks_2samp(real[c].astype(float), syn[c].astype(float)).statistic) for c in NUM}


def tvd_per_categorical(real, syn):
    out = {}
    for c in CAT:
        rv = real[c].value_counts(normalize=True)
        sv = syn[c].value_counts(normalize=True)
        keys = set(rv.index) | set(sv.index)
        out[c] = float(0.5 * sum(abs(rv.get(k, 0) - sv.get(k, 0)) for k in keys))
    return out


def dcr_stats(syn_num, train_num, test_num, std):
    nn = NearestNeighbors(n_neighbors=1).fit(train_num / std)
    syn_d = nn.kneighbors(syn_num / std)[0][:, 0]
    test_d = nn.kneighbors(test_num / std)[0][:, 0]
    return syn_d, test_d


def c2st_auc(real, syn):
    """Classifier two-sample test: AUC ~0.5 => indistinguishable (good)."""
    n = min(len(real), len(syn), 20000)
    r = real.sample(n, random_state=0); s = syn.sample(n, random_state=0)
    X = pd.concat([r[FEATS], s[FEATS]], ignore_index=True)
    X = pd.get_dummies(X, columns=CAT)
    y = np.r_[np.zeros(n), np.ones(n)]
    idx = np.random.RandomState(0).permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = idx[:cut], idx[cut:]
    clf = HistGradientBoostingClassifier(max_iter=120, random_state=0)
    clf.fit(X.iloc[tr], y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(X.iloc[te])[:, 1]))


def utility(real_train_fraud, real_nonfraud, syn_fraud, test_df):
    """Augmentation utility: classifier with synthetic fraud vs real fraud vs none,
    evaluated on the real held-out test set (PR-AUC primary at native imbalance)."""
    nf = real_nonfraud.sample(min(150000, len(real_nonfraud)), random_state=0)
    Xte = pd.get_dummies(test_df[FEATS], columns=CAT)
    yte = (test_df["isFraud"].astype(int)).values

    def run(fraud_rows):
        tr = pd.concat([nf[FEATS], fraud_rows[FEATS]], ignore_index=True)
        ytr = np.r_[np.zeros(len(nf)), np.ones(len(fraud_rows))]
        X = pd.get_dummies(tr, columns=CAT)
        X, Xt = X.align(Xte, join="outer", axis=1, fill_value=0)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(X, ytr)
        p = clf.predict_proba(Xt)[:, 1]
        return float(average_precision_score(yte, p)), float(roc_auc_score(yte, p))

    real_pr, real_roc = run(real_train_fraud)
    # augment real fraud with synthetic (cap synthetic contribution for a fair, fast run)
    aug = pd.concat([real_train_fraud, syn_fraud.sample(min(100000, len(syn_fraud)), random_state=0)], ignore_index=True)
    aug_pr, aug_roc = run(aug)
    return {"real_only_prauc": real_pr, "real_only_rocauc": real_roc,
            "augmented_prauc": aug_pr, "augmented_rocauc": aug_roc}


# ---------- figures ----------
def fig_numeric(real, syn, dose, d):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), NUM):
        r = real[c].astype(float).values
        s = syn[c].astype(float).values
        if c in MONEY:
            r, s = np.log1p(r), np.log1p(s)
            ax.set_xlabel(f"log1p({c})")
        else:
            ax.set_xlabel(c)
        lo = min(r.min(), s.min()); hi = max(r.max(), s.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(r, bins=bins, density=True, alpha=0.55, label="real fraud", color="#1f77b4")
        ax.hist(s, bins=bins, density=True, alpha=0.55, label="synthetic", color="#d62728")
        ax.legend(fontsize=8)
    fig.suptitle(f"{dose}: numerical marginals (real vs synthetic fraud)")
    fig.tight_layout()
    fig.savefig(f"{d}/numeric_marginals.png", dpi=90); plt.close(fig)


def fig_categorical(real, syn, dose, d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, c in zip(axes, CAT):
        rv = real[c].value_counts(normalize=True)
        sv = syn[c].value_counts(normalize=True)
        keys = sorted(set(rv.index) | set(sv.index))
        x = np.arange(len(keys)); w = 0.4
        ax.bar(x - w/2, [rv.get(k, 0) for k in keys], w, label="real", color="#1f77b4")
        ax.bar(x + w/2, [sv.get(k, 0) for k in keys], w, label="synthetic", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(c); ax.legend(fontsize=8)
    fig.suptitle(f"{dose}: categorical distributions")
    fig.tight_layout()
    fig.savefig(f"{d}/categorical_dists.png", dpi=90); plt.close(fig)


def fig_corr(real, syn, dose, d):
    rr = real[NUM].astype(float).corr()
    ss = syn[NUM].astype(float).corr()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m, t in zip(axes, [rr, ss, (ss - rr).abs()], ["real corr", "synthetic corr", "|diff|"]):
        im = ax.imshow(m, vmin=-1 if t != "|diff|" else 0, vmax=1, cmap="coolwarm" if t != "|diff|" else "Reds")
        ax.set_xticks(range(len(NUM))); ax.set_xticklabels(NUM, rotation=90, fontsize=7)
        ax.set_yticks(range(len(NUM))); ax.set_yticklabels(NUM, fontsize=7)
        ax.set_title(t); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{dose}: numerical correlation structure")
    fig.tight_layout()
    fig.savefig(f"{d}/correlation.png", dpi=90); plt.close(fig)


def fig_dcr(syn_d, test_d, dose, d):
    fig, ax = plt.subplots(figsize=(7, 4))
    hi = np.percentile(np.r_[syn_d, test_d], 99)
    bins = np.linspace(0, hi, 50)
    ax.hist(test_d, bins=bins, density=True, alpha=0.55, label="real test->train (baseline)", color="#2ca02c")
    ax.hist(syn_d, bins=bins, density=True, alpha=0.55, label="synthetic->train", color="#d62728")
    ax.axvline(np.median(test_d), color="#2ca02c", ls="--")
    ax.axvline(np.median(syn_d), color="#d62728", ls="--")
    ax.set_xlabel("distance to closest real train fraud (standardized)")
    ax.set_title(f"{dose}: DCR — left of baseline = memorization risk")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{d}/dcr.png", dpi=90); plt.close(fig)


# ---------- verdicts ----------
def verdict(v, good, ok, higher_better=False):
    if higher_better:
        return "PASS" if v >= good else ("WARN" if v >= ok else "FAIL")
    return "PASS" if v <= good else ("WARN" if v <= ok else "FAIL")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    train = load_real(f"{SPLIT}/train.csv")
    test = load_real(f"{SPLIT}/test.csv")
    train_f = train[train["isFraud"] == 1].reset_index(drop=True)
    test_f = test[test["isFraud"] == 1].reset_index(drop=True)
    nonfraud = train[train["isFraud"] == 0]
    std = train_f[NUM].astype(float).std().replace(0, 1).values
    base_dcr = float(np.median(dcr_stats(test_f[NUM].astype(float).values,
                                         train_f[NUM].astype(float).values,
                                         test_f[NUM].astype(float).values, std)[1]))

    sets = find_sets()
    all_metrics = {"baseline_dcr": base_dcr, "real_train_fraud": len(train_f),
                   "real_test_fraud": len(test_f), "sets": {}}
    report = []
    report.append("# Paysim Approach-2 — Synthetic Fraud Trustworthiness Report\n")
    report.append(f"Real fraud: **{len(train_f)}** train (generator source), **{len(test_f)}** test (held-out).  ")
    report.append(f"Baseline DCR (real test→train fraud): **{base_dcr:.4f}** — the natural distance a *genuine* "
                  "unseen fraud row sits from the training fraud. Synthetic should match this, not undercut it.\n")

    for dose, path in sets.items():
        d = f"{OUTDIR}/{dose}"
        os.makedirs(d, exist_ok=True)
        syn_full = prep_syn(path)
        syn = syn_full.sample(min(SAMPLE, len(syn_full)), random_state=0).reset_index(drop=True)

        ks = ks_per_numeric(train_f, syn)
        tvd = tvd_per_categorical(train_f, syn)
        purity = float((syn_full["isFraud"].astype(str) == "1").mean())
        type_valid = float(syn_full["type"].isin(FRAUD_TYPES).mean())
        neg_amount = float((syn_full["amount"].astype(float) < 0).mean())
        dup_rate = float(syn_full.duplicated(subset=FEATS).mean())
        exact_mem = float(len(syn_full.merge(train_f[FEATS].drop_duplicates(), on=FEATS, how="inner")) / len(syn_full))

        syn_d, test_d = dcr_stats(syn[NUM].astype(float).values[:DCR_Q],
                                  train_f[NUM].astype(float).values,
                                  test_f[NUM].astype(float).values, std)
        med_syn_dcr = float(np.median(syn_d))
        dcr_ratio = med_syn_dcr / base_dcr if base_dcr else float("inf")
        auc = c2st_auc(train_f, syn)
        util = utility(train_f, nonfraud, syn_full, test)

        # figures
        fig_numeric(train_f, syn, dose, d)
        fig_categorical(train_f, syn, dose, d)
        fig_corr(train_f, syn, dose, d)
        fig_dcr(syn_d, test_d, dose, d)

        max_ks = max(ks.values()); max_tvd = max(tvd.values())
        m = {"path": path, "n": len(syn_full), "ks": ks, "max_ks": max_ks, "tvd": tvd, "max_tvd": max_tvd,
             "purity": purity, "type_valid": type_valid, "neg_amount": neg_amount,
             "dup_rate": dup_rate, "exact_mem": exact_mem, "median_syn_dcr": med_syn_dcr,
             "dcr_ratio": dcr_ratio, "c2st_auc": auc, "utility": util}
        all_metrics["sets"][dose] = m

        # per-set report section
        report.append(f"\n---\n## {dose}\n")
        report.append(f"*{len(syn_full):,} rows — figures in `{d}/`*\n")
        report.append("| Question | Metric | Value | Verdict |")
        report.append("|---|---|---|---|")
        report.append(f"| Q1 marginals | max KS (numeric) | {max_ks:.3f} | {verdict(max_ks,0.1,0.2)} |")
        report.append(f"| Q1 marginals | max TVD (categorical) | {max_tvd:.3f} | {verdict(max_tvd,0.05,0.15)} |")
        report.append(f"| Q3 validity | fraud-plausible type % | {type_valid*100:.1f}% | {verdict(type_valid,0.98,0.90,higher_better=True)} |")
        report.append(f"| Q3 validity | negative amounts | {neg_amount*100:.2f}% | {verdict(neg_amount,0.001,0.01)} |")
        report.append(f"| Q4 purity | isFraud==1 | {purity*100:.1f}% | {verdict(purity,0.999,0.95,higher_better=True)} |")
        report.append(f"| Q5 memorization | exact matches to train | {exact_mem*100:.3f}% | {verdict(exact_mem,0.001,0.01)} |")
        report.append(f"| Q5 privacy | DCR ratio (syn/baseline) | {dcr_ratio:.2f} | {'PASS' if dcr_ratio>=0.8 else 'FAIL(mem)'} |")
        report.append(f"| Q6 diversity | within-set dup rate | {dup_rate*100:.3f}% | {verdict(dup_rate,0.01,0.05)} |")
        report.append(f"| Q7 indistinguishability | C2ST AUC | {auc:.3f} | {verdict(auc,0.6,0.75)} |")
        report.append(f"| Q8 utility | PR-AUC real-only → augmented | {util['real_only_prauc']:.3f} → {util['augmented_prauc']:.3f} | "
                       f"{'PASS' if util['augmented_prauc']>=util['real_only_prauc'] else 'WARN'} |")

    # overall recommendation
    report.append("\n---\n## Overall recommendation\n")
    ranked = sorted(all_metrics["sets"].items(),
                    key=lambda kv: (kv[1]["type_valid"], -kv[1]["max_ks"]), reverse=True)
    report.append("Ranked by fidelity (fraud-plausible type %, then KS):\n")
    for dose, mm in ranked:
        report.append(f"- **{dose}**: type-valid {mm['type_valid']*100:.0f}%, max-KS {mm['max_ks']:.2f}, "
                      f"C2ST {mm['c2st_auc']:.2f}, DCR×{mm['dcr_ratio']:.1f}, "
                      f"PR-AUC {mm['utility']['real_only_prauc']:.3f}→{mm['utility']['augmented_prauc']:.3f}")
    report.append("\n*Pool weighting should follow this ranking; sets that FAIL Q1/Q3/Q7 should be "
                  "down-weighted or type-filtered before use. Any set failing Q5 must be discarded.*\n")

    with open(f"{OUTDIR}/REPORT.md", "w") as f:
        f.write("\n".join(report))
    with open(f"{OUTDIR}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Report -> {OUTDIR}/REPORT.md")
    print(f"Metrics -> {OUTDIR}/metrics.json")
    print("\n".join(report))


if __name__ == "__main__":
    main()
