# Banksim Approach-2b — d1 (fraud-only) Run Log & Findings

*Author: automated run, 2026-06-15. Dataset: `/home/amad/projects/datasets/banksim/splits_temporal`
(5 temporal splits). Goal: train a **fraud-only (d1)** TabDiff model per split and
generate **100,000 synthetic fraud rows** per split, following
[approach_2b_extreme_imbalance_large_data.md](approach_2b_extreme_imbalance_large_data.md).*

---

## 1. Objective & constraints

- Run the Approach-2b pipeline on banksim's `splits_temporal`, **d1 dose only**
  (fraud-only model, reject-sampling generation).
- Train **only** on each split's `train.csv`. **`test.csv` must never be read** by any
  part of the pipeline.
- Hold out **20% of the training fraud** as a validation/early-stop signal (the real
  `test.csv` stays untouched).
- Produce **100k synthetic fraud rows per split**.

---

## 2. Dataset & schema decisions

Banksim raw columns:
`step, customer, age, gender, zipcodeOri, merchant, zipMerchant, category, amount, fraud`

**Dropped columns** (`DROP_COLS = ["customer", "zipcodeOri", "zipMerchant"]`):

| Column | Reason |
|---|---|
| `zipcodeOri` | constant (1 unique value) — zero signal |
| `zipMerchant` | constant (1 unique value) — zero signal |
| `customer` | ~4,024-unique account ID — no learnable structure, bloats embedding (same rationale as paysim's `nameOrig`/`nameDest`) |

**Kept 7 columns** (indices after drop):

| idx | column | role |
|---|---|---|
| 0 | step | numerical |
| 1 | age | categorical (`'U'` present → handled as string, not int) |
| 2 | gender | categorical |
| 3 | merchant | categorical |
| 4 | category | categorical |
| 5 | amount | numerical |
| 6 | fraud | target (binclass) |

→ `NUM_COL_IDX=[0,5]`, `CAT_COL_IDX=[1,2,3,4]`, `TARGET_COL_IDX=[6]`.

**Per-split fraud counts** (train.csv): bs0=1,480 / bs1=2,760 / bs2=3,920 / bs3=5,080 / bs4=6,160
(fraud rate ~1.2–1.5%). After the 80/20 held-out carve, train fraud =
1,184 / 2,208 / 3,136 / 4,064 / 4,928.

### Keeping `test.csv` untouched

TabDiff's trainer uses the dataset's **processed test split** as its held-out
EMA-loss / early-stop signal (`tabdiff/trainer.py`, `val_iter`). We never point at the
split's real `test.csv`. Instead, `prepare_banksim_app2.py` carves **20% of the train
fraud rows** and registers them as the dataset's `test_path`. The carve guarantees every
categorical value in the held-out set also appears in train (otherwise the encoders,
fit on train, fail on the held-out split). The real `test.csv` is referenced **nowhere**
in prepare / train / generate.

---

## 3. Scripts

| Script | Purpose |
|---|---|
| `prepare_banksim_app2.py` | Per split: drop cols, cast cats→str, carve 80/20 train/held-out fraud, register `bs<S>_d1` TabDiff dataset (with `test_path` = held-out), write per-split TOML config. Optional `BANKSIM_LOG_AMOUNT="2,3"` env applies `log1p(amount)`. |
| `run_banksim_app2.py` | GPU-pool orchestrator (4 T4s). One job per split: `train` → reject-`generate` 100k fraud. Splits passed as argv (default all 5). |
| `generate_paysim_app2.py` | Reused as-is (dataset-agnostic). `reject` mode: unconditional sample, keep `fraud==1`. |
| `validate_banksim_app2.py` | Memorization + fidelity metrics vs real train fraud (see §6). |
| `plot_banksim_app2_loss.py` | Plots per-epoch train + held-out EMA loss from `loss_history.csv`. |

### Trainer changes (`tabdiff/trainer.py`, config-driven, defaults preserve old behavior)

1. **Always-on per-epoch loss CSV** — writes `loss_history.csv` (epoch, lr, train d/c/total,
   held-out EMA d/c/total) to the checkpoint dir. wandb runs in `disabled` mode (`--no_wandb`),
   so without this no loss curve survives the run.
2. **`select_loss` option** (`"total"` default | `"cat"` | `"num"`) — chooses which held-out
   EMA-loss component drives `best_ema` checkpoint selection and early stopping. Needed because
   the held-out **numerical** loss is a broken signal on some splits (§5).

---

## 4. d1 hyperparameters

Baseline d1 recipe (all splits, memorization-aware for the tiny fraud sets):

```
dim_t=256, batch_size=256, num_timesteps=100,
lr=1e-3 (reduce_lr_on_plateau, factor 0.90), ema_decay=0.997,
weight_decay=1e-4, val_sample_size=5000, check_val_every=500.
```

Early stopping was **enabled** (we have a genuine held-out signal), but its interaction
with model selection caused the bs2 failure — see §5. Final per-split selection settings:

| Split | steps (max ep) | best_ckpt_start_epoch | early_stop_patience | select_loss | log1p(amount) |
|---|---|---|---|---|---|
| bs0 | 3500 | 200 | 300 | total | no |
| bs1 | 3500 | 200 | 300 | total | no |
| bs2 | **1500** | **1100** | **0 (disabled)** | **cat** | **yes** |
| bs3 | 3500 | 200 | 300 | total* | **yes** |
| bs4 | 3500 | 200 | 300 | total | no |

\* bs3's good checkpoint was selected on `total` at epoch 914, which *happened* to be
converged — see §7 "fragility".

Generation: **reject** mode, target 100k, `SAMPLE_BATCH=4096`. Fraud-only model →
~100% yield → 100k collected in <1 min.

---

## 5. Debugging journey (bs2 / bs3 failures)

Initial run: bs0/bs1/bs4 were clean; **bs2 and bs3 failed** with catastrophic held-out
EMA loss (~10⁶ vs ~1.8) and poor generation (categorical TV ~0.3–0.4, dcr ~0.17 vs
~0.01). The failure was **reproducible across re-runs** (not random-seed noise).

Interventions tried, in order:

1. **Lower LR (1e-3 → 3e-4).** No effect — ruled out optimization step size.
2. **`select_loss="cat"`.** Inspecting `loss_history.csv` showed the held-out **numerical**
   loss (`d_loss`) was ~10⁶ from **epoch 1** and wildly noisy, while the **categorical**
   loss (`c_loss`) was healthy (~1.0). The 10⁶ is an EDM-weighting artifact (loss
   ∝ 1/σ² at small sampled σ), not a real divergence — but it swamped the `total` loss used
   for `best_ema` selection and tripped early-stop prematurely (~250 epochs) → undertrained.
   Selecting on `cat` let bs2/bs3 train longer; fidelity improved but didn't reach parity.
3. **`log1p(amount)`** (heavy-tailed numerical column, raw range [0, 7665]). The held-out
   numerical loss **stayed at 10⁶** → confirmed it's *not* a data-scale problem. However
   log-amount **did** improve numerical generation: **bs3 snapped to full parity** (dcr
   0.17→0.006). bs2 remained bad.
4. **Distribution inspection.** bs2's generated categoricals were **over-flattened**
   (dominant categories under-weighted, mass leaked to minor ones) uniformly across all
   categorical columns — a "categorical sampling too hot" signature, not mode collapse.
5. **Learned-schedule inspection (root cause).** Dumping `cat_schedule.k_raw` from the
   checkpoints:
   - bs2 (bad): feature-cat `k ≈ [-7.27, -7.24, -7.07, -7.06]`
   - bs3/bs4 (good): `k ≈ [-9.1, -9.0, -8.8, -8.6]`

   `k` controls the categorical noise schedule's end-sharpness. bs2's `k` never reached the
   sharp ~−9 regime → noisy final denoising step → flattened distribution. **Cause:** bs2's
   `best_ema` was selected at **epoch 401**, before `k` converged (`k` needs ~900+ epochs).
   The flat/noisy held-out categorical loss couldn't distinguish converged from unconverged
   checkpoints, so selection latched onto a too-early one.

**Final fix for bs2:** `best_ckpt_start_epoch=1100` + `early_stop_patience=0` (disabled) +
`steps=1500` → forces `best_ema` selection among **converged** checkpoints. bs2's `k` then
reached −9.3 and fidelity snapped to parity (dcr 0.17→0.007, all cat TV ≤0.02).

---

## 6. Final validation results

Metrics vs real train fraud (`validate_banksim_app2.py`, full report in
`logs/banksim_app2_d1/validation_report.json`):

| Split | synth rows | duplicate_rate | memorized_rate | dcr_zero | dcr_median | age TV | gender TV | merchant TV | category TV |
|---|---|---|---|---|---|---|---|---|---|
| bs0 | 100,000 | 0.000 | 0.000 | 0.000 | 0.013 | 0.011 | 0.000 | 0.030 | 0.011 |
| bs1 | 100,000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.012 | 0.004 | 0.023 | 0.013 |
| bs2 | 100,000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.009 | 0.002 | 0.017 | 0.012 |
| bs3 | 100,000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.036 | 0.020 | 0.069 | 0.045 |
| bs4 | 100,000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.024 | 0.015 | 0.062 | 0.049 |

- **No memorization** anywhere (duplicate / exact-match / zero-distance all 0).
- All categorical Total-Variation distances ≤0.07; numerical means/stds within a few %.
- All 5 splits at full fidelity parity.

Outputs: `tabdiff/synthetic_fraud/bs<S>_d1/fraud_100000.csv` (×5).
Loss curves: `logs/banksim_app2_d1/loss_curves_per_split.png`, `loss_curves_heldout_overlay.png`.

> Note on the loss-curve "best held-out EMA" column: for bs2/bs3 it prints the **total**
> (numerical-dominated, ~10⁶) which is meaningless for them — their selection used the
> categorical component (~0.9–1.0).

---

## 7. Operational caveats & known fragilities

1. **`prepare_*` overwrites configs.** `prepare_banksim_app2.py` rewrites each split's TOML
   *before* the data-skip check. Re-preparing a split **wipes manual config edits**
   (`select_loss`, `best_ckpt_start_epoch`, etc.). bs2 carries hand-tuned settings; do not
   re-prepare it without re-applying them.

2. **bs2/bs3 were trained on `log1p(amount)`.** Their generated `amount` is inverse-
   transformed with `expm1` as a **manual post-step** after `run_banksim_app2.py`
   (`df.amount = np.expm1(df.amount).clip(lower=0)`). This is **not yet folded into the
   generate script** — regenerating bs2/bs3 requires re-running that step. Splits bs0/bs1/bs4
   use raw `amount` and need no post-step.

3. **bs3's config is "lucky", not robust.** bs3's successful checkpoint was selected on the
   broken `total` loss but happened to land at a converged epoch (914). Its config still has
   `select_loss=total`, `early_stop_patience=300`, `best_ckpt_start_epoch=200`. A re-run is
   **not guaranteed** to reproduce a good checkpoint. For robustness, bs3 should adopt bs2's
   settings (`select_loss=cat`, `best_ckpt_start_epoch=1100`, `early_stop_patience=0`).

4. **The held-out numerical EMA loss is unreliable** (~10⁶ on some splits due to EDM
   1/σ² weighting at small σ). Do not use `total` held-out loss for model selection on these
   tiny-data d1 runs; `select_loss="cat"` is the robust choice.

---

## 8. Reproduce from scratch

```bash
conda activate tabdiff
cd /home/amad/projects/TabDiff

# Prepare (bs0/bs1/bs4 raw; bs2/bs3 with log-amount)
python prepare_banksim_app2.py 0
python prepare_banksim_app2.py 1
python prepare_banksim_app2.py 4
BANKSIM_LOG_AMOUNT="2,3" python prepare_banksim_app2.py 2
BANKSIM_LOG_AMOUNT="2,3" python prepare_banksim_app2.py 3

# Harden bs2 (and recommended: bs3) config: select_loss=cat,
# best_ckpt_start_epoch=1100, early_stop_patience=0, steps=1500  (edit TOML)

# Train + generate (100k fraud each)
python run_banksim_app2.py            # all 5, pooled over 4 GPUs

# Inverse-transform amount for log-amount splits
python - <<'PY'
import pandas as pd, numpy as np
for s in [2,3]:
    p=f"tabdiff/synthetic_fraud/bs{s}_d1/fraud_100000.csv"; df=pd.read_csv(p)
    if df.amount.max()<20: df["amount"]=np.expm1(df["amount"]).clip(lower=0); df.to_csv(p,index=False)
PY

# Validate + plot
python validate_banksim_app2.py
python plot_banksim_app2_loss.py
```
