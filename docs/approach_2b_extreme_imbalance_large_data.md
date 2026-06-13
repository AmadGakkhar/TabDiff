# Approach 2b — Majority-Dosage Sweep for *Extremely Large, Extremely Imbalanced* Datasets

*An adaptation of [Approach 2](approach_2_majority_dosage_sweep.md) for datasets that
are (a) orders of magnitude larger (millions of rows) and (b) far more imbalanced
(minority share ≪ 1%) than the vehicle-insurance table Approach 2 was written for.
Worked against paysim: ~5.09M train rows, `isFraud` positive rate ≈ **0.129%**
(6,570 fraud vs 5,083,526 non-fraud) in split 0.*

---

## 1. Why Approach 2 breaks here

Approach 2's engine is **reject sampling**: generate whole rows unconditionally, keep
the ones that come out as fraud. Its yield equals the training set's fraud share. That
is fine at 6–11% fraud (worst case ~17× over-generation). It collapses completely at
paysim's ratios.

For a target of **1.6M fraud rows per model**, the over-generation required is:

| Model | Train rows | Fraud share ≈ yield | Rows to generate for 1.6M fraud |
|---|---|---|---|
| d1 — fraud only | 6,570 | ~100% | ~1.6M ✅ |
| d2 — +25% majority | ~1.28M | ~0.51% | **~311M** ❌ |
| d3 — +50% majority | ~2.55M | ~0.26% | **~620M** ❌ |
| d4 — full data | ~5.09M | ~0.129% | **~1.24B** ❌ |

Generating hundreds of millions to >1 billion rows through a 100-step diffusion
sampler on commodity GPUs (4× T4 here) is infeasible — days-to-weeks of compute and
terabytes of intermediate CSV per model. **Reject sampling is only viable for the
fraud-only model (d1).** Everything else needs a different generation mechanism.

---

## 2. The fix: conditional generation by clamping the target

TabDiff is an unconditional joint model `P(features, target)` with no class-conditional
sampler — which is exactly why Approach 2 resorted to reject sampling. But TabDiff
already ships a **diffusion in-painting / imputation** routine
(`UnifiedCtimeDiffusion.sample_impute`) that fixes some columns and generates the rest.
The paper uses it to **impute the label from observed features**. We run it in
**transpose**:

- **Observe** the target column `isFraud = 1`.
- **Mask (generate)** every feature column.
- At each of the 100 reverse-diffusion steps the target is re-clamped to fraud, so the
  generated features are drawn from `P(features | isFraud = 1)`.

Every emitted row is fraud by construction → **yield ≈ 100% for all four models**,
including the full-data model. 1.6M fraud per model becomes ~1.6M generated rows, not
a billion.

### Why no y_only guidance model is used

TabDiff's imputation supports classifier-free guidance via a small auxiliary **y_only**
model. Reading the code (`sample_impute` / `edm_update`), that guidance steers the
**masked** columns using a model of the **target**. In the paper's setup the masked
column *is* the target, so it fits. In our transpose the masked columns are the
**features**, and a target-only model cannot guide them — the guidance branch does not
apply. We therefore run **plain hard conditioning** (`y_only_model=None`, `w_num =
w_cat = 0`): the target is clamped directly at every step. This is the correct
mechanism for "generate features given the label," and it needs no extra model.

> Honest caveat: hard conditioning *forces* `isFraud=1` rather than letting fraud
> emerge naturally. For d4 (saw 0.129% fraud) we are leaning entirely on the joint
> model's learned fraud mode. The mechanism is sound and yields 100%, but **fraud
> fidelity from the high-majority models must be validated** (Section 5) before the
> 1.6M is trusted. If fidelity is weak, the remedy is to train a proper *feature*
> unconditional model for guidance (an "x_only" model) — deferred unless validation
> demands it.

### Per-model generation mode

| Model | Mode | Rationale |
|---|---|---|
| d1 | **reject** | fraud-only, ~100% yield already; unconditional sampling is the most faithful and needs no clamping |
| d2 / d3 / d4 | **conditional (clamped-target impute)** | only feasible way to reach 1.6M at <1% yield |

---

## 3. Pipeline (what actually runs here)

Implemented in three scripts:

- `prepare_paysim_app2.py` — from `datasets/paysim/splits/0/train.csv`, drop the
  near-unique identifier columns `nameOrig` (~5.08M unique) and `nameDest` (~2.27M
  unique), cast `isFraud`/`isFlaggedFraud` to string categoricals, build the four
  dosage partitions (fraud + 0/25/50/100% of the majority, intermediate draws
  independent), and register each as a TabDiff dataset with a per-dose config.
- `run_paysim_app2.py` — **one worker per GPU, one dose each**, running
  **train → generate end-to-end** so every dose yields usable data the moment its own
  training finishes (no barrier across the sweep).
- `generate_paysim_app2.py` — loads the `best_ema` checkpoint + cached `config.pkl`,
  reuses the dataset encoders for decoding, and runs the per-dose mode above in a loop
  until 1.6M fraud rows are collected, appending to
  `tabdiff/synthetic_fraud/<dose>/fraud_1600000.csv`.

The Approach-2 inviolable rule is unchanged: **synthetic fraud augments classifier
training only; validation and test stay real and at the native ~0.13% ratio.**

---

## 4. Per-model sizing at this scale

The dosage partitions span 6.6k → 5.09M rows. Three scale-specific traps dominate,
and **all three are non-obvious because TabDiff's `steps` config is actually a count
of EPOCHS, and each epoch is a full pass over every batch.**

1. **Epochs must shrink as data grows — hard.** Naively reusing small-data epoch
   counts (thousands of passes) is catastrophic at scale: 18,000 epochs over 4.58M
   train rows is ~20M gradient steps ≈ **8 days** on a T4. A diffusion model on
   millions of rows converges in *tens* of passes. So epochs go *down* with size,
   and the big batch is what keeps the T4 busy.

   | Dose | Train rows | dim_t | Batch | Max epochs | Weight decay | check_val_every |
   |---|---|---|---|---|---|---|
   | d1 | 6,570 | 256 | 256 | 3,500 | 1e-4 | 500 |
   | d2 | ~1.15M | 512 | 4,096 | 300 | 1e-4 | 50 |
   | d3 | ~2.29M | 512 | 4,096 | 200 | 1e-5 | 50 |
   | d4 | ~4.58M | 1,024 | 8,192 | 120 | 0 | 30 |

   Max epochs are upper bounds; early stopping (below) usually halts sooner. This
   brings every model under ~1.5h instead of hours-to-days.

2. **Validation cost is a hidden trap.** TabDiff's in-training validation generates
   `real_data_size` samples each time — ~4.6M rows *per validation* for d4, ×2
   (non-EMA + EMA). We added a `val_sample_size` cap (50k) so routine validation is
   bounded regardless of dataset size.

3. **"Early stopping" needs a held-out signal, which the stock code lacked.** The
   original `compute_loss` ran over the **training** set, so the "EMA loss" used for
   checkpoint selection could not see overfitting, and there was no stop-on-plateau —
   training always ran to `steps`, gated by a hard-coded `curr_epoch > 4000` that made
   best-checkpoint selection impossible for any short run. We changed three things in
   `tabdiff/trainer.py` (all config-driven, defaults preserve old behavior):
   - `compute_loss` now defaults to a **held-out** iterator (the test split) — a true
     generalization signal;
   - `best_ckpt_start_epoch` replaces the hard-coded 4000 gate (so best-EMA selection
     works for d1's 3,500-epoch run, starting at epoch 300);
   - `early_stop_patience` halts training after N epochs without held-out EMA-loss
     improvement, and `best_ema` is the checkpoint at that minimum.

   This is the real overfitting defense for **d1** (6,570 rows): EMA averaging + strict
   weight decay + early stop on held-out loss, with a post-hoc DCR / duplicate gate
   (Section 5) as the final backstop.

---

## 5. Validation — and the extra burden here

All Approach 2 / Approach 1 Section 7 checks apply (diversity two-sample test,
generative precision/recall vs real fraud, distance-to-closest-real for memorization,
native-ratio PR-AUC downstream, no synthetic in val/test). Two checks carry **extra
weight** under this regime:

- **Fidelity of conditionally-generated fraud (d2/d3/d4).** Because fraud is *forced*,
  verify the generated fraud actually lands on the real fraud manifold — per-column
  distributions and distance-to-real against the held-out real fraud. A high-majority
  model that under-resolved its 0.129% fraud mode will betray itself here.
- **Memorization of d1.** A fraud-only model trained on 6,570 rows and asked for 1.6M
  rows will repeat itself; duplicate-rate and distance-to-closest-real on d1's output
  are mandatory, and its pool contribution should be capped accordingly.

As in Approach 2, the validation report — not generation convenience — sets the
per-model pooling weights. d4 in particular is a candidate to **drop entirely** if its
conditional fraud fails fidelity, since it is both the most expensive to train and the
most capacity-starved on fraud.

---

## 6. Summary

At paysim's scale and imbalance, Approach 2's reject sampling survives only for the
fraud-only model. The other three doses switch to **conditional generation by clamping
`isFraud=1` and in-painting the features** via TabDiff's existing imputation routine
(no y_only guidance, since the masked columns are features), giving ~100% yield and
making 1.6M fraud per model feasible. Training is scaled per dose with validation
cadence widened to avoid million-row in-training sampling blowups, and each dose is
trained-then-generated independently per GPU so data lands as early as possible.
The forced-fraud mechanism shifts more of the burden onto validation: conditional
fraud fidelity (d2/d3/d4) and d1 memorization are the two gates that decide pooling
weights — and whether d4 is kept at all.
