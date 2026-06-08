# Approach 2 — Majority-Dosage Sweep Ensemble of TabDiff Generators

*Synthetic minority generation for a highly imbalanced fraud dataset, using an
ensemble of TabDiff models that differ in how much of the majority class they are
trained with. Companion to Approach 1 (balanced-bagging ensemble); the problem
framing, guardrails, validation, and model-sizing principles are shared.*

---

## 1. Problem and goal (shared with Approach 1)

The dataset is the vehicle-insurance-fraud table: ~12,300 rows, 33 mixed-type
columns, binary target `FraudFound_P`, with only ~6% fraud (~738 fraud vs ~11,600
non-fraud). The end goal is a **downstream binary classifier** that performs well on
an **unseen real test set at the native ~6% ratio**. TabDiff is used purely to
manufacture additional realistic **fraud (minority) rows** that improve that
classifier.

The same two firm decisions apply:

- **Augment, don't replace** — keep all real data and add synthetic fraud on top.
- **The evaluation distribution stays real and at 6%** — synthetic rows appear only
  inside classifier training, never in validation or test.

The same core constraint applies too: TabDiff has **no class-conditional sampling**,
so we sample whole rows and keep those that come out as fraud (reject sampling), and
the **yield** — the fraction of generated rows that are fraud — is governed by the
fraud share of the model's training set.

---

## 2. The idea: sweep the majority dose, then ensemble across the sweep

Where Approach 1 builds M models that are all the *same* balanced size (differing
only in which random majority slice they drew), Approach 2 builds M models that
differ in **how much of the majority class they see at all**. With M = 4:

| Model | Training set | Majority added | Fraud share (≈ yield) |
|---|---|---|---|
| 1 | fraud only | 0 | ~100% |
| 2 | fraud + 25% of majority | ~2,900 | 738 / 3,638 ≈ **20%** |
| 3 | fraud + 50% of majority | ~5,799 | 738 / 6,537 ≈ **11%** |
| 4 | fraud + 100% of majority | 11,598 | 738 / 12,336 ≈ **6%** |

(Here "X% of the majority" means X% of the non-fraud rows; *all* fraud rows are
included in every model. Model 4 is therefore the full dataset.)

These four models are deliberately placed at **different points on the
capacity↔regularization axis** discussed in Approach 1:

- **Model 1 (fraud only)** — maximum capacity on the fraud manifold, zero
  regularization from the majority, highest memorization risk, perfect yield.
- **Model 4 (full data)** — maximum regularization and the most "in-context
  realistic" fraud, but the fraud mode is capacity-starved and yield is poor.
- **Models 2 and 3** — interpolate between the two extremes.

Pooling samples across the four gives a **mixture that spans the whole spectrum**.
The appeal is twofold:

1. **Principled, interpretable diversity.** The models differ along a *meaningful*
   axis (majority dose), not just random seeds. After generation you can read off
   which dose produced the best fraud and learn the right operating point.
2. **Hedging.** You do not have to commit to the single "correct" majority ratio;
   you blend several and let validation weight them.

### How it differs from Approach 1

- **Source of diversity.** Approach 1: different random majority *slices* at a fixed
  balance. Approach 2: different *amounts* of majority context.
- **Member homogeneity.** Approach 1's members are interchangeable and equally good,
  so equal-weight pooling is natural. Approach 2's members are **qualitatively
  different and unequal in quality** — which changes both the sampling strategy and
  the configuration (below).
- **Yield.** Approach 1: every member ~50%. Approach 2: **heterogeneous**, from
  ~100% (model 1) down to ~6% (model 4).

### The honest ceiling (shared)

All four models still learn from the **same ~738 real fraud rows** — the ensemble
cannot create new fraud information. It improves capacity allocation, sampling
diversity, yield, and interpretability. Memorization risk is reduced relative to a
single fraud-only model but not eliminated, which is why validation is mandatory.

---

## 3. Two caveats that shape the design

1. **Heterogeneous and partly poor yield.** Each model's yield equals its training
   fraud share. Models 3 and 4 are *both* capacity-starved on fraud *and* expensive
   to sample from (model 4 needs ~17× over-generation). They are the costliest and
   weakest fraud generators.

2. **Unequal member quality → equal-weight pooling is not automatic.** Because the
   four models are qualitatively different (model 1 may memorize; model 4 may
   under-resolve fraud), giving each an equal share of the final pool gives equal
   voice to unequal quality. The per-model contribution should be **decided from the
   validation report**, not by default.

A smaller point: draw the 25% and 50% majority subsets **independently at random**,
not nested (25% ⊂ 50% ⊂ 100%), or the diversity between models 2, 3 and 4 shrinks.

---

## 4. End-to-end pipeline

**Stage 0 — Split once, up front (shared).** Stratified train / validation / test
preserving the ~6% ratio in every split. Generators use the **train split only**;
validation and test stay real and at 6%. Under cross-validation, this applies
per fold.

**Stage 1 — Build the four nested-dose partitions.** From the train split, form
four training sets: all fraud plus 0% / 25% / 50% / 100% of the majority, the
intermediate majority draws taken independently at random.

**Stage 2 — Train four TabDiff models**, one per partition, each with its own seed
and a configuration scaled to its size (Section 6).

**Stage 3 — Generate fraud from each model**, over-generating according to each
model's yield (Section 5), and reject-filtering for fraud. Discard the synthetic
non-fraud; keep only synthetic minority.

**Stage 4 — Pool** the fraud rows from the four models into one synthetic fraud set,
with per-model contributions set by the strategy in Section 5.

**Stage 5 — Validate** (shared with Approach 1): confirm the models are genuinely
different, that the difference is useful (pooled coverage of the real fraud manifold
exceeds any single model's), and that none memorized.

**Stage 6 — Hand off** the pooled, validated synthetic fraud set to the downstream
pipeline, which composes `real train + synthetic fraud`, sweeps the augmentation
ratio and classifier hyperparameters against the **real 6% validation set**, and
reports on the **real 6% test set**.

---

## 5. Sampling strategy (worked for a target of 2,500 fraud rows)

Because yields differ per model, the *effort* to collect a given number of fraud
rows differs sharply. For each model you over-generate by roughly
`target ÷ yield`, reject-filter on the target, and keep the target count of fraud.

**Default — equal contribution (625 fraud rows per model):**

| Model | Yield | Target fraud | Rows to generate (≈ target / yield) |
|---|---|---|---|
| 1 — fraud only | ~100% | 625 | ~650 (small margin for filter/NaN loss) |
| 2 — +25% | ~20% | 625 | ~3,100 |
| 3 — +50% | ~11% | 625 | ~5,600 |
| 4 — +100% | ~6% | 625 | ~10,500 |
| **Total** | — | **2,500** | **~19,900 generated, 2,500 kept** |

Then concatenate the four 625-row sets and shuffle → 2,500 pooled synthetic fraud.

**Practical sampling notes:**
- Size each model's reject-sampling loop (chunk size × iterations) to *its* yield.
  Model 4 at ~6% needs large over-generation, or it will exhaust its iteration budget
  before reaching 625.
- Discard the synthetic non-fraud — the real non-fraud is kept for the classifier;
  only synthetic *minority* is needed.
- Run the duplicate / distance-to-closest-real check **especially on model 1's
  share** — a fraud-only model on ~738 rows is the most likely to emit near-duplicate
  or repeated fraud.

**Recommended refinement — quality-weighted contribution.** Treat the equal 625-each
split as a starting point only. After running the validation report (Section 7),
**re-allocate the 2,500 toward the models that are both faithful and add coverage**:
cut a memorizing model's share and shift it to better-behaved doses; weight up a
model whose fraud is the most realistic even if it is the most expensive to sample.
The final split should follow validated quality, not generation convenience.

---

## 6. Model configuration — per-model sizing

The shared model-sizing principle (right-size capacity, batch, and training length
to subset size; regularize and early-stop the small ones; raise the number of
sampling steps for fidelity) is documented in full in Approach 1, Section 6. What is
specific to Approach 2 is that the four models span very different sizes
(738 → 12,336 rows), so the configuration is **scaled per model** rather than applied
uniformly:

| Model | Rows | Batch | Network width | Weight decay | Training length | Notes |
|---|---|---|---|---|---|---|
| 1 — fraud only | ~738 | smallest | reduced (small) | small (nonzero) | shortest + strict early stop | highest memorization risk — monitor distance-to-real hardest; target column is degenerate and can be dropped |
| 2 — +25% | ~3.6k | small–moderate | moderate | small | moderate | |
| 3 — +50% | ~6.5k | moderate | moderate–default | very small | mild reduction | |
| 4 — +100% | ~12.3k | default | default | none | default | this *is* the original full-data setting — keep stock config; emphasize the categorical/target head if its generated labels look noisy |

All four models also take the **shared** sampling-time tweaks: more denoising steps,
stochastic sampling and second-order correction left on, and a validation cadence
frequent enough for early stopping to be actionable.

The **stopping criterion** is the shared one: select the EMA checkpoint that
maximizes fidelity against a held-out real-fraud set *subject to* the
distance-to-closest-real staying above a memorization floor — most important for
model 1.

---

## 7. Validation, guardrails, and metrics (shared)

These are identical to Approach 1 and are not re-derived here:

- **Diversity** — a classifier two-sample test across model pairs (held-out AUC near
  0.5 ⇒ redundant; well above ⇒ genuinely different), plus per-column distance and a
  distribution-level distance.
- **Usefulness** — generative precision and recall against the real fraud rows, per
  model *and* pooled; the win condition is pooled recall above any single model's
  with precision held.
- **Memorization** — distance-to-closest-real per model; healthy models sit near but
  not on the reals.
- **Guardrails** — no synthetic in validation/test; no generator sees held-out data;
  evaluate at the native 6% ratio (precision-recall AUC primary, never raw accuracy);
  calibrate / tune the threshold on the real 6% validation set; judge synthetic
  quality on the fraud rows in isolation, not the pooled dataset.

Because Approach 2's members are unequal in quality, the validation report is not
just a safety check here — it is the **input to the pooling weights** (Section 5).

---

## 8. When to prefer Approach 2 over Approach 1

- **Prefer Approach 2** when you want an **interpretable diagnostic** of how much
  majority context produces the best fraud, or when you want to hedge across the
  capacity↔regularization spectrum rather than commit to one balance point.
- **Prefer Approach 1** when you want **homogeneous, comparable members** with simple
  equal-weight pooling and uniformly high yield (~50% everywhere), and you do not
  need the per-dose interpretability.

The two are not mutually exclusive: Approach 2 can be run first as a cheap diagnostic
to find the most productive majority dose, and Approach 1 then run at (or near) that
dose to mass-produce diverse fraud efficiently.

---

## 9. Summary

Approach 2 trains four TabDiff models that differ in **majority dose** — fraud-only,
+25%, +50%, +100% — and pools their synthetic fraud. This spans the
capacity↔regularization spectrum in one ensemble and yields an interpretable read on
which dose generates the best fraud. The cost is **heterogeneous yield** (perfect for
model 1, ~6% for model 4, so ~19,900 rows generated to keep 2,500) and **unequal
member quality**, which means the per-model pooling contribution should be set by the
validation report rather than split evenly. Configuration is scaled per model to its
subset size, smallest models most aggressively regularized. As in Approach 1, the
inviolable rule holds: **synthetic data augments classifier training only;
validation and test stay real and at the native 6% ratio.**
