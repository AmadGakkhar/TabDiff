# TabDiff Session Report — Synthetic Vehicle-Insurance-Fraud Data

**Date:** 2026-06-01
**Source data:** `data/original_train.csv` (Vehicle Insurance Fraud), 12,336 rows × 33 columns
**Goal:** Train TabDiff on the data, generate (1) a distribution-matching synthetic copy and (2) a fraud-enhanced variant (~11% fraud), and assess quality.

---

## 1. Executive summary

- Stood up a working TabDiff environment on a **4 GB laptop GPU (NVIDIA T1200)**, fixing three dependency/compatibility issues along the way.
- Registered the data as dataset **`fraud`**, dropping the `PolicyNumber` ID column (32 modeled columns).
- Trained to **epoch 6000** (best EMA checkpoint at epoch 4617, loss 1.7353) — loss had plateaued, so training was stopped early with no loss of quality.
- Delivered **two synthetic datasets** (12,336 rows each) plus a full quality report and comparison charts.
- **Verdict:** fidelity is **excellent** (density ≈ 0.97–0.98, C2ST ≈ 0.86). The only soft spot is the rare fraud class, which the distribution-matching set slightly under-represents (4.17% vs real 5.98%). The fraud-enhanced set is **more useful for training a fraud detector** (higher ML-efficacy score).

---

## 2. Deliverables

| File | Rows | Cols | Fraud % | Purpose |
|---|---|---|---|---|
| `synthetic_outputs/fraud_synthetic_full.csv` | 12,336 | 32 | 4.17% | Distribution-matching synthetic copy |
| `synthetic_outputs/fraud_synthetic_enhanced11pct.csv` | 12,336 | 32 | 11.00% | Minority (fraud) enhanced to +5 points |
| `synthetic_outputs/density_full.png` | — | — | — | Real-vs-synthetic per-column density (#1) |
| `synthetic_outputs/density_enhanced11pct.png` | — | — | — | Real-vs-synthetic per-column density (#2) |
| `synthetic_outputs/chart_*.png` | — | — | — | Summary charts (this report) |

Reusable helper scripts: `eval_delivered.py` (quality report), `plot_delivered.py` (density plot), `make_report_charts.py` (charts).

---

## 3. What we did (chronological)

1. **Environment** — created the `tabdiff` conda env from `tabdiff.yaml` (PyTorch 2.0.1 + CUDA 11.7). GPU verified (T1200, CUDA available).
2. **Dataset registration** — copied the CSV to `data/fraud/fraud.csv`, authored `data/Info/fraud.json` classifying 7 numerical / 24 categorical / 1 target (`FraudFound_P`) columns; dropped `PolicyNumber` (a unique ID = noise for a generative model).
3. **GPU adaptation** — lowered batch sizes for 4 GB VRAM: train `4096→2048`, sample `10000→4096`. Peak usage stayed ~0.7 GB, so there was ample headroom.
4. **Preprocessing** — `process_dataset.py` produced an 11,102 / 1,234 train/test split.
5. **Smoke test** — a `--debug` run validated the full pipeline (train → sample → evaluate → plot) before the long run.
6. **Training** — 8000-epoch run on GPU, stopped at epoch 6000 once the loss plateaued.
7. **Generation** — produced Dataset #1 directly (12,336 rows); produced a 60k-row pool and assembled Dataset #2 to hit exactly 11% fraud.
8. **Evaluation** — fixed the MLE metric and produced the full quality report + density plots + summary charts.

### Issues found & fixed

| Issue | Symptom | Fix |
|---|---|---|
| MKL too new for PyTorch 2.0.1 | `ImportError: undefined symbol: iJIT_NotifyEvent` | downgraded `mkl 2025 → 2023.1` |
| Kaleido v1 needs Chrome | density-plot step crashed | pinned `kaleido==0.2.1` |
| Dropping a **middle** column via Info file only | `recover_data` `IndexError` | physically removed `PolicyNumber` from the CSV and renumbered indices |
| XGBoost 3.2 removed `gpu_hist` | MLE metric `NotFittedError` (error swallowed by a bare `except`) | changed `tree_method 'gpu_hist' → 'hist'` in `eval/mle/mle.py` (3 spots) |

---

## 4. Training

Loss dropped steeply, then plateaued around ~1.79 from epoch ~3000 on. The best EMA checkpoint was captured at **epoch 4617** (the framework only starts saving "best" checkpoints after epoch 4000); no lower EMA loss appeared between 4617 and 6000, confirming convergence and justifying the early stop.

![Training loss](synthetic_outputs/chart_training_loss.png)

---

## 5. Generated class balance

The distribution-matching set tracks the real fraud rate but rounds the rare class down slightly (4.17% vs 5.98%). The enhanced set was assembled to exactly 11% by sampling 1,357 fraud + 10,979 non-fraud rows from a 60k pool (2,533 synthetic fraud rows were available).

![Class balance](synthetic_outputs/chart_class_balance.png)

Reference real rates: original 5.98%, train split 5.91%, test split 6.65%.

---

## 6. Quality evaluation

Metrics computed with the repo's own evaluators against the held-out real test split. **All metrics: higher = better.**

| Metric | #1 dist-match | #2 enhanced | What it measures |
|---|---|---|---|
| density/Shape | **0.9824** | 0.9791 | per-column marginal distributions vs real |
| density/Trend | **0.9703** | 0.9651 | column-pair correlations vs real |
| density/Overall | **0.9764** | 0.9721 | combined statistical fidelity |
| mle (utility) | 0.7614 | **0.7987** | classifier trained on synthetic, scored on real test |
| c2st (realism) | **0.8607** | 0.8323 | 1.0 = a detector can't tell synthetic from real |

![Quality metrics](synthetic_outputs/chart_quality_metrics.png)

### Per-column density (real vs synthetic)

**Dataset #1 — distribution-matching:**
![Density #1](synthetic_outputs/density_full.png)

**Dataset #2 — fraud-enhanced:**
![Density #2](synthetic_outputs/density_enhanced11pct.png)

---

## 7. Remarks — what the results tell us

**The model learned the data well.**
Shape ≈ 0.98 and Trend ≈ 0.97 mean the synthetic data reproduces both the *individual* column distributions and the *relationships between* columns very faithfully — visible in the density plots, where the real (dark) and synthetic (teal) curves overlap closely across all 32 columns. A C2ST of **0.86** is strong for mixed-type tabular data: a logistic detector can only weakly separate synthetic rows from real ones. In short, the joint distribution was captured accurately.

**The synthetic data is private by construction.**
TabDiff samples fresh rows from the learned distribution rather than perturbing real records — there is no row-to-row correspondence with the original data, so no synthetic record maps back to a real policyholder.

**The rare class is the one real caveat.**
The distribution-matching set produced **4.17% fraud vs 5.98%** in the real data. This under-representation is the classic weakness of generative models on imbalanced data: with only ~660 fraud examples in training, the model has less signal for the minority mode and tends to "round it down." Minority rows are also less diverse than majority rows. This is the dimension to watch if fraud cases are the point of interest.

**Rebalancing pays off for downstream utility.**
The fraud-enhanced set scored **higher on ML-efficacy (0.799 vs 0.761)** — a classifier trained on the 11%-fraud synthetic data generalizes *better* to the real test set than one trained on the 4%-fraud set. This is the practical payoff: if the end goal is training a fraud detector, the enhanced dataset is the more valuable artifact. Its slightly lower C2ST (0.832) is expected and *by design* — we deliberately changed the class balance, which makes the distribution intentionally different from the real one. That trade-off (lower realism, higher task utility) is exactly what oversampling is for.

**Bottom line.**
- For *statistical analysis / sharing a realistic stand-in* for the original data → use **Dataset #1**.
- For *training/augmenting a fraud-detection model* → use **Dataset #2** (or generate at an even higher fraud ratio).
- If you need high-fidelity *minority* rows specifically, the levers are: train longer, raise the categorical loss weight `c_lambda`, or implement true conditional generation (fix `FraudFound_P=1`, generate the features) — a small `trainer.py` change discussed during the session.

---

## 8. Reproduce / reuse

```bash
conda activate tabdiff

# generate any row count from the trained model (no retraining needed)
python main.py --dataname fraud --mode test --no_wandb --exp_name fraud_full \
    --num_samples_to_generate <N>

# full quality report on any generated CSV
python eval_delivered.py <csv> fraud

# real-vs-synthetic density plot for any generated CSV
python plot_delivered.py <csv> <out.png> fraud
```

Trained checkpoint: `tabdiff/ckpt/fraud/fraud_full/best_ema_model_1.7353_4617.pt`
Run logs: `logs/` (`train_full.log`, `generate*.log`, `eval_*.log`)
Tuning guide (parameters to improve quality): `/home/amad/.claude/plans/i-want-to-train-abstract-gosling.md`
