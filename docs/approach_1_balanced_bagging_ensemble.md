# Approach 1 — Balanced-Bagging Ensemble of TabDiff Generators

*Synthetic minority generation for a highly imbalanced fraud dataset, in service of a downstream classifier that must perform well on an unseen, real test set at the native (~6%) fraud ratio.*

---

## 1. Problem and goal

We have a vehicle-insurance-fraud dataset: ~12,300 rows, 33 mixed-type columns
(numeric + categorical), with a binary target `FraudFound_P`. The data is highly
imbalanced — only ~6% of rows are fraud (~738 fraud vs ~11,600 non-fraud).

The end goal is **not** a beautiful synthetic dataset for its own sake. It is a
**downstream binary classifier** that ranks/flags fraud well on an **unseen real
test set whose fraud ratio is also ~6%**. TabDiff is used only as a tool to
manufacture additional, realistic **minority (fraud) examples** that make that
classifier better than it would be on the raw data alone.

Two firm decisions frame everything below:

- **Augment, don't replace.** We keep all the real data and *add* synthetic fraud
  on top. We do not train the classifier purely on synthetic data. Augmentation
  maximizes downstream accuracy; full replacement is for privacy, which is not our
  objective.
- **The evaluation distribution stays real and stays at 6%.** Validation and test
  are always real rows at the native ratio. Synthetic data appears **only** inside
  classifier training. This is non-negotiable: if synthetic rows leak into
  validation or test, every reported number is inflated and the operating point we
  pick will be wrong in deployment.

The classifier's hyperparameter tuning and augmentation-ratio sweep are owned by a
separate, existing pipeline. **The scope of this document is purely the generation
of synthetic fraud rows** — specifically, how to generate them well given the
imbalance.

---

## 2. The core constraint that shapes the design

TabDiff models the **joint** distribution of a row, including the target column.
It has **no native class-conditional sampling** — you cannot ask it for "a fraud
row." You sample whole rows unconditionally and **keep the ones that came out as
fraud** (reject sampling).

This makes the *fraction of generated rows that are fraud* — what we call the
**yield** — a central quantity:

> **yield = (number of fraud rows generated) / (total rows generated)**

A model trained on the raw 6% data reproduces ~6% fraud, so yield ≈ 6%: to collect
1,000 fraud rows you generate ~17,000 rows and discard ~16,000. Beyond being slow,
this exposes the deeper problem below.

### The capacity problem

A generative model spends its representational capacity roughly in proportion to
where the data mass is. With only ~6% of rows being fraud, the **fraud manifold is
under-resolved** — and the fraud rows are precisely the ones the downstream
classifier needs most. So the naive "train one model on everything, filter for
fraud" pipeline gives both poor yield *and* poor minority fidelity.

### The two single-model fixes, and why each is unsatisfying

- **Fraud-only model** — train TabDiff on the ~738 fraud rows alone. Generation is
  then 100% fraud (perfect yield) and the model dedicates all its capacity to the
  fraud manifold. But it never co-trains with the majority, so it loses the
  statistical regularization that shared structure provides, and with only ~738
  rows it is at serious risk of **memorizing** the reals (low diversity, no privacy,
  inflated downstream scores).

- **Joint model + reject-filter** — train on all the data, filter for fraud.
  Retains cross-class regularization and is the safest against memorization, but the
  fraud mode is capacity-starved (the 6% problem) and yield is poor.

Neither is strictly better. The tension is: *capacity on the minority mode* and
*low yield/diversity loss* pull in opposite directions from *regularization and
memorization safety*.

---

## 3. The idea: balanced bagging of generators

Borrow the **undersampling-ensemble** trick (EasyEnsemble / balanced bagging) — but
apply it to the **generator**, not to the downstream classifier.

Instead of training one TabDiff model, train **M** TabDiff models. Each model is
trained on a **balanced subset** of the data:

- **all** of the minority (every fraud row is included in every member), plus
- a **different random undersample of the majority** (non-fraud), chosen so the
  member's training set is roughly balanced.

Then **generate fraud from every member and pool the results** into one synthetic
fraud set.

This is the generative analog of why undersampling-bagging works for classifiers:
each member sees a balanced, manageable view; the ensemble as a whole still touches
all the data; and aggregating many members produces something better than any one.

### Why this resolves the tension

| Property | Joint+reject | Fraud-only | **Balanced-bagging ensemble** |
|---|---|---|---|
| Fraud-mode capacity | starved (~6%) | full | **full — each member is ~50% fraud** |
| Regularization from majority | yes | none | **yes — every member co-trains with majority** |
| Memorization of the ~738 reals | low | high | **reduced — mixture over M models smooths it** |
| Uses all majority data | yes | no | **yes — spread across members** |
| Yield | ~6% | 100% | **~50%** |

Two mechanisms do the work:

1. **Capacity.** Because each member's training set is balanced, the fraud manifold
   gets roughly half of that member's capacity instead of 6%. The minority mode is
   no longer starved.

2. **Mixture = diversity + anti-memorization.** Each member sees a *different*
   majority backdrop and a different random initialization, so each learns a
   slightly different fraud conditional. Pooling their samples yields a **mixture
   distribution** that is more diverse, and less prone to copying any individual
   real fraud row, than a single fraud-only model. Diversity here is a feature: it
   is exactly what makes additional minority data useful rather than redundant.

### The honest ceiling

Every member still learns from the **same ~738 real fraud rows** — the ensemble
cannot manufacture new fraud *information*. What it improves is **capacity
allocation, regularization, sampling diversity, and yield**. Memorization risk is
*reduced, not eliminated*, which is why validation (Section 8) is mandatory.

---

## 4. End-to-end pipeline

**Stage 0 — Split, once, up front.**
Partition the real data into train / validation / test by stratified sampling that
**preserves the ~6% fraud ratio in every split**. TabDiff and all generation use
the **train split only**. Validation and test remain real, at 6%, and are never
seen by any generator. If the downstream pipeline uses cross-validation, this
discipline applies per fold: a fold's generators are trained only on that fold's
training rows, never its validation rows.

**Stage 1 — Build M balanced partitions.**
From the train split, construct M training subsets. Each subset contains all the
fraud rows plus a distinct random undersample of the non-fraud rows, sized to make
the subset approximately balanced. The majority draws should differ across members
(rotating or disjoint) so that, collectively, the ensemble has been exposed to the
entire majority — even though we will ultimately discard the synthetic non-fraud,
the majority context is what makes each member's fraud conditional different.

**Stage 2 — Train M independent TabDiff models.**
One model per partition, each with its own random seed. Because each partition is
small and balanced, training is cheaper per model than training on the full data,
and the fraud mode is well-resourced.

**Stage 3 — Generate fraud from every member.**
Decide the total synthetic fraud budget T. Each model contributes ~T/M fraud rows.
Each model generates a pool of rows and we keep the ones that came out as fraud
(reject sampling); because the members are balanced, yield is ~50%, so we generate
only ~2× the rows we keep. The synthetic non-fraud that is produced is discarded —
we keep the real non-fraud for the downstream classifier and only need synthetic
*minority*.

**Stage 4 — Pool.**
Concatenate the fraud rows from all M members into a single synthetic fraud set.
This pooled set is the mixture distribution; it is the deliverable.

**Stage 5 — Validate the ensemble (Section 8).**
Before handing the pool downstream, confirm the members learned something
genuinely different and genuinely useful, and that they did not memorize.

**Stage 6 — Hand off.**
Deliver the pooled synthetic fraud set (optionally large and diverse enough that
the downstream pipeline can draw different subsets per classifier, e.g. for its own
ensembling or ratio sweep). The downstream pipeline composes
`real train + synthetic fraud`, tunes the augmentation ratio and classifier
hyperparameters against the **real 6% validation set**, and reports final metrics
on the **real 6% test set**.

---

## 5. Knobs worth exploring

These are the levers specific to generating well under imbalance.

**Ensemble structure**
- **M (number of members).** Enough to cover the majority across members. Smaller M
  with larger per-member majority subsets, or larger M with strictly balanced
  subsets — a coverage-vs-cost trade.
- **Majority subset size per member.** Strict balance maximizes fraud capacity per
  member; allowing somewhat more majority per member preserves more cross-class
  context but dilutes the fraud share. Worth sweeping.
- **Minority handling.** Keep *all* fraud rows in every member (the EasyEnsemble
  convention). Bootstrapping the fraud rows would shrink the effective unique
  minority per member and usually hurts.
- **Diversity sources.** Different majority draw per member, plus a different random
  seed per member.

**Per-model generation fidelity**
- **Number of denoising steps.** More steps → higher per-sample fidelity at the cost
  of sampling time. The minority samples are the hard ones, so this is a cheap win.
- **Stochastic sampling.** Adds diversity to samples; valuable when each model is
  trained on a small balanced set, to avoid collapsing onto a few modes.
- **Discrete-vs-continuous loss balance / target-column emphasis.** The target is a
  categorical column we reject-sample on; emphasizing the categorical heads keeps the
  generated labels crisp. (Less critical here, since balanced members already make
  fraud common.)

**Training duration / capacity per member**
- Each member trains on a small set, so training length and model size should guard
  against overfitting the tiny minority rather than chase the long schedule tuned for
  the full dataset.

---

## 6. Model configuration — sizing the model to the subset

*(This section applies to both approaches; it is duplicated by reference in the
Approach 2 document.)*

The stock TabDiff configuration was tuned for a **full-size dataset (~12k rows)**.
In both approaches most models train on **much smaller subsets** — down to ~738
rows for a fraud-only member. Running the default configuration on a few hundred
rows is a recipe for **memorization**: an over-wide network, a batch larger than
the dataset, and thousands of update steps will simply copy the real fraud rows
back out.

The governing principle is therefore simple: **scale capacity, batch size, and
training length to each model's subset size, and add regularization plus early
stopping for the small ones.** None of this is exotic — it is right-sizing the
model to the data it has.

### Tweaks that apply to every model (both approaches)

- **More denoising steps at sampling time.** Raise the number of sampling steps
  from the default toward 2–4× it. More steps means higher per-sample fidelity, and
  the minority rows are the hard ones. This costs sampling time only, not training.
- **Keep stochastic sampling and the second-order correction on.** The first adds
  sample diversity (important on small training sets, where collapse is a risk); the
  second improves per-sample accuracy.
- **Evaluate validation frequently.** The default validation cadence is too coarse
  to stop a short run in time. Check often enough that early stopping is actionable.

### Tweaks that scale with subset size (the real levers)

- **Batch size — the most urgent fix.** A batch larger than the training subset
  means every "batch" is essentially the whole dataset: almost no gradient
  stochasticity and very few genuine updates. Drop the batch well below the subset
  size (smallest subsets want the smallest batches).
- **Think in epochs, not steps.** The default step count, over a few hundred rows,
  is hundreds of passes over the data — guaranteed memorization. Hold the *epoch*
  budget modest and let early stopping end the run rather than training to a fixed
  step count.
- **Network width is the capacity dial.** The default hidden width is far
  over-parameterized for ~1k rows. Shrinking it is the main anti-overfit lever after
  batch size. Depth (number of layers) is already minimal and can stay as is.
- **Add weight decay for the small subsets.** The default is none; a small amount
  meaningfully regularizes models trained on little data, and can taper to zero for
  the full-size model.
- **EMA.** Keep the exponential-moving-average weights (they are what gets sampled);
  only shorten the EMA horizon slightly if a run is unusually short.

The size buckets:

| Bucket | Subset size | Batch | Network width | Weight decay | Training length |
|---|---|---|---|---|---|
| Small | ≤ ~2k rows | smallest (well below subset) | reduced | small (nonzero) | short + early stop |
| Mid | ~2–6k rows | moderate | moderate | very small | moderate |
| Full | ~12k rows | default | default | none | default |

### Applying this in Approach 1

Because every member is the **same** small, balanced size (all fraud plus an equal
majority undersample), apply **one uniform "small-subset" profile** to all M
members — reduced batch, reduced width, small weight decay, short training with
early stopping, and more sampling steps. Homogeneous members, homogeneous config.
(Approach 2, with members of very different sizes, instead scales the config
per-model; see that document.)

### Stopping criterion — do not simply minimize validation loss

For the small models, validation loss can keep falling while the model memorizes.
Select the EMA checkpoint that **maximizes fidelity** (a distributional quality
measure against a held-out real-fraud set) **subject to the distance-to-closest-real
staying above a memorization floor** — i.e. samples that sit *near but not on* the
reals. Checkpoint periodically and choose on this trade-off rather than training to
a fixed number of steps.

### Two specifics

- **A fraud-only member has a degenerate target column** (every row is fraud). That
  is harmless: no reject sampling is needed, every generated row is minority by
  construction, and the target column can even be dropped from its training.
- **The discrete-loss weight matters only where you reject-sample on the target** —
  i.e. members with a low fraud share. If the generated labels of such a member look
  noisy, emphasizing the categorical/target head sharpens them. Otherwise leave it at
  its default.

---

## 7. Methodological guardrails

- **No synthetic data in validation or test.** Ever. They stay real and at 6%.
- **No generator sees held-out data.** Train generators only on training rows
  (per fold under CV), or we leak the thing we are trying to measure.
- **Evaluate at the native ratio.** Accuracy is meaningless at 6% (predicting "never
  fraud" scores ~94%). Use ranking/threshold metrics — precision-recall AUC as the
  primary, ROC-AUC, recall at a fixed precision.
- **Calibration / threshold tuning.** Both balanced generation and any downstream
  balancing distort the prior away from 6%, so model scores are miscalibrated for
  deployment. Choose the operating threshold on the **real 6% validation set**, not
  on any balanced set.
- **Per-class evaluation.** Judge synthetic quality on the **fraud rows in
  isolation**, not on the pooled dataset. Pooled metrics are dominated by the 94%
  non-fraud and will look fine even when the fraud rows are poor.

---

## 8. Validating the ensemble

Diversity is necessary but not sufficient. Ask two questions, and always pair a
diversity check with a fidelity check (a model can be "different" simply by being
wrong — dropping a mode or drifting off the real manifold).

**Q1 — Are the M models actually different from each other?**
- **Classifier two-sample test (decisive).** Train a classifier to distinguish one
  model's samples from another's, for all member pairs. Held-out AUC near 0.5 means
  the two models produce the same distribution (redundant); AUC well above 0.5 means
  they are genuinely different. Averaging the pairwise AUCs gives one diversity
  score.
- **Per-column distribution distance** (divergence for categoricals, distance for
  numerics) between member pairs — shows *which features* differ.
- **Distribution-level distance** (e.g. maximum mean discrepancy) between member
  sample sets as a single summary.

**Q2 — Is the difference useful (complementary), not just random drift?**
- **Generative precision and recall against the real fraud rows**, computed per
  member *and* for the pooled set. Precision = synthetic fraud lands on the real
  fraud manifold (each member must stay high). Recall/coverage = how much of the
  real fraud manifold is reached. The win condition is **pooled recall higher than
  any single member's recall, with precision held** — that is the ensemble genuinely
  covering more of the true fraud distribution.
- **Nearest-real-neighbor coverage.** Count how many distinct real fraud rows are
  "covered" by each member vs. by the pool; the pool should cover more.

**Q3 — Did the models memorize?**
- **Distance-to-closest-record (DCR) / nearest-neighbor** per member. Two failure
  modes to rule out: all members copy the *same* reals (looks redundant), or each
  copies a *different* slice (looks diverse, but it is memorization, not
  generalization). Healthy models sit *near but not on* the real rows.

**Reading the result**

| Pairwise AUC | Pooled recall vs single member | Verdict |
|---|---|---|
| ≈ 0.5 | — | Models redundant — ensemble adds nothing; drop to one model |
| > 0.5 | higher (precision held) | **Useful diversity — ensemble working as intended** |
| > 0.5 | about the same | Diverse but not complementary — diminishing returns from larger M |
| > 0.5 | higher but precision drops | "Diversity" is off-manifold drift — bad |

---

## 9. Summary

Treat TabDiff as a **minority oversampler**, and handle the imbalance on the
generation side with a **balanced-bagging ensemble**: M models, each trained on all
the fraud plus a different majority undersample, sampled from collectively, and
pooled. This gives full capacity on the fraud mode, retains regularization from the
majority, raises yield from ~6% to ~50%, and produces a diverse mixture that resists
memorization — while respecting the one rule that protects the whole exercise:
**synthetic data augments training only; validation and test stay real and at the
native 6% ratio.** The pooled, validated synthetic fraud set is then handed to the
existing downstream pipeline for ratio sweeping and classifier tuning.
