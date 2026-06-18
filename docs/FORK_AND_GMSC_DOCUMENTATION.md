# TabDiff Fork — Evolution & GMSC Synthetic-Data Documentation

**Repo:** `/home/amad/projects/TabDiff` (fork of `MinkaiXu/TabDiff`, ICLR 2025)
**Author of fork work:** Amad Ud Din Gakkhar
**Scope:** what was added on top of upstream TabDiff, *why*, and what *effect* it had — with a deep, code-level focus on the **GMSC** (Give Me Some Credit) dataset work.

---

## 0. TL;DR

Upstream TabDiff is a research generator for mixed-type tabular data. This fork turns it into a **production tool for synthesising minority-class (fraud / default / positive) rows** that downstream imbalanced classifiers can train on. The fork **never touches the diffusion model itself** — all value is in three layers:

1. **Data-prep scripts** that filter a split to its minority class and register it as a TabDiff dataset.
2. **Generation scripts** that exploit reject-sampling (keep `target == minority`) — and an inverted-mask `sample_impute` fallback for extreme imbalance.
3. **Trainer/CLI plumbing** that makes training reliable on tiny, memorisation-prone minority sets (held-out loss, best-EMA-checkpoint selection, per-model configs, `select_loss`).

The **GMSC** work (commits `f775a87`, `fa955f9`, `42eee20`) is the most mature instance of this pattern. It went through three iterations: a v1 baseline, a critical generation-filter bug fix, and a **v2 fidelity overhaul** that cut the real-vs-synthetic discriminator AUC from **0.80–0.95 → 0.57–0.67** (closer to the 0.50 "indistinguishable" ideal), plus an **inner-train / 10×** variant built for leakage-free downstream model selection.

---

## 1. Upstream baseline (what we started from)

Boundary: commits by *Juntong Shi / Minkai Xu* (≤ `5ecdb33`, 2025) are upstream; everything from `06dc805` (2026) on is this fork.

- **Model** (`tabdiff/models/unified_ctime_diffusion.py`): one continuous-time diffusion over numerical **and** categorical columns jointly. Numerical columns use EDM Gaussian noising (`c_loss`); categorical columns use absorbing-state/masked discrete diffusion (`d_loss`). Per-column **learnable noise schedules** (`noise_schedule.py`: `power_mean_per_column`, `log_linear_per_column`). Backbone `UniModMLP` + EDM preconditioner.
- **Data contract** (`process_dataset.py`, `data/Info/<name>.json`): a dataset declares `num_col_idx` / `cat_col_idx` / `target_col_idx` (mutually exclusive) and `task_type`. Processing does a 90/10 stratified split, quantile-normalises numerics (`src/data.py: normalize`, `QuantileTransformer(output_distribution='normal')`), ordinal-encodes categoricals, and builds `idx_name_mapping` (position→column name). The fork registers with `column_names: null` and recovers names from `idx_name_mapping`.
- **Dequantizer** (`src/data.py: dequantizer`): `dequant_dist ∈ {none, uniform, beta, round}` controls integer handling. `none` (upstream default) is a **no-op on inverse** → integers come back as floats. `uniform` adds `U(0,1)*int_dequant_factor` in training and **floors on inverse**; `round` rounds on inverse. *(This setting becomes central to the GMSC v2 fix.)*
- **Training/sampling**: `python main.py --dataname X --mode train` and `--mode test [--num_samples_to_generate N]`. By default sampling produces as many rows as the real dataset; upstream `best_ema` checkpoints were only saved after epoch 4000 and EMA loss was computed on the **train** set (couldn't detect overfitting).

---

## 2. Fork architecture — the recurring pattern

Every dataset adaptation follows the same three-step pipeline:

```
prepare_<X>.py        # filter split → minority class, write CSV, register Info JSON, run process_dataset.py
        │
main.py --mode train  # train a TabDiff model on the minority-only (or dosed) data, per-model --config_path
        │
generate_<X>.py       # load best_ema ckpt, sample in chunks, KEEP rows where target == minority, write CSV
```

**Why reject-sampling?** TabDiff has no native class-conditional sampling. The fork over-generates an *unconditional* pool and filters by the target column. For a model trained on **minority-only** data, ~100 % of generated rows are minority, so reject-sampling is both faithful and fast. For models trained with majority context (very low minority yield), the fork instead **reuses `sample_impute` with inverted masks** — *observe* the target, *generate* the features — giving ~100 % yield by construction.

### 2.1 Trainer/CLI plumbing the fork added (depended on by GMSC)

These are fork additions to `tabdiff/trainer.py` / `tabdiff/main.py` (introduced in `8267dbc`/`8032634`), **not upstream**:

- `--config_path` — each model uses its own TOML (enables parallel per-GPU training of many splits/doses).
- `--resume` — resume from the latest checkpoint.
- **Held-out EMA loss**: a validation iterator over the *test/held-out split* drives `best_ema` selection and early-stopping, so checkpoint selection reflects **generalisation** (upstream used train loss).
- `best_ckpt_start_epoch` (overridable; upstream hard-coded 4000), `early_stop_patience`, `val_sample_size` (cap in-training sampling cost), `loss_history.csv`.
- `select_loss ∈ {total, cat, num}` — track only the categorical or numerical component of held-out loss (the quantile transformer's numerical-tail blow-ups can otherwise swamp model selection on minority-only data).

---

## 3. Evolution timeline of fork additions (pre-GMSC)

| Date | Commit | What | Why |
|------|--------|------|-----|
| 2026-06-02 | `06dc805` | `eval_delivered.py`, `plot_delivered.py`, density charts; shrink default batch sizes | Score/visualise delivered synthetic CSVs without retraining |
| 2026-06-02 | `992a92c` | `register_exp2.py`, **`gen_balanced.py`** (reject-sample a class-balanced pool), `run_exp2.sh`, `--resume` | First balanced fully-synthetic datasets; establishes reject-sampling |
| 2026-06-08 | `d618c8d` | `prepare_exp3.py` (decode one-hot → categorical, `encode_map.json`), `gen_fraud_synthetic.py` (distribution-preserving) | Drop-in synthetic replacements preserving original encoding & imbalance |
| 2026-06-08/09 | `6389ada`,`d8f627a` | **Approach 1 — balanced bagging ensemble**: `prepare_expv01_app1.py` (M=4 balanced bags), `gen_expv01_app1.py` (keep fraud only), pool 4×625 | Ensemble of generators on different balanced bags → diverse minority set; member-sized config to avoid memorising ~1.5k-row bags |
| 2026-06-13 | `8267dbc` | **Approach 2 — majority-dosage sweep**: `prepare_paysim_app2.py` (`_BASE`+`DOSE_OVERRIDES` d1–d4), `generate_paysim_app2.py` (**reject** + **conditional/impute**), `run_paysim_app2.py`; **trainer plumbing** (§2.1) | Test whether feeding the generator some majority context yields more realistic fraud; reject for cheap d1, impute for impossibly-low-yield high doses |
| 2026-06-14 | `59f83f3` | `prepare_d1.py` (d1-only per split, imports from `prepare_paysim_app2`), `run_d1_splits.sh`, **`--max_seconds`** time cap | Scale across splits; bound generation wall-clock (d1 fraud-only reject is the workhorse) |
| 2026-06-15 | `8032634` | **BankSim** `prepare_banksim_app2.py`; `heldout_split()` (carve 20 % of *train* fraud as held-out, never touch real test); `select_loss` | Second, more-imbalanced fraud dataset; true held-out signal + robust model selection |
| 2026-06-17 | `d98dc92` | **diabetes** `prepare_diab_d1.py` (non-fraud positive class); generalise `generate_paysim_app2.py` with **`--out_dir` / `--label`** | Prove the pipeline is dataset-agnostic; decouple output naming from "fraud" |

The `DOSE_OVERRIDES` from `prepare_paysim_app2.py` (reused by GMSC via `build_config("d1")`) are worth recording, since the GMSC config is derived from `d1`:

| dose | dim_t | batch | steps | weight_decay | best_ckpt_start_epoch | val_sample_size |
|------|-------|-------|-------|--------------|-----------------------|-----------------|
| **d1** (minority-only) | 256 | 256 | 3500 | 1e-4 | 300 | 5000 |
| d2 (+25% maj) | 512 | 4096 | 800 | 1e-4 | 40 | 50000 |
| d3 (+50% maj) | 512 | 4096 | 500 | 1e-5 | 40 | 50000 |
| d4 (+100% maj) | 1024 | 8192 | 500 | 0 | 20 | 50000 |

d1's small batch (256) + weight decay (1e-4) + best-EMA-checkpoint selection are deliberate **anti-memorisation** measures for small minority sets — the foundation GMSC builds on.

---

## 4. GMSC — the centerpiece

### 4.1 Context: the dataset and the downstream task

GMSC ("Give Me Some Credit") is a Kaggle credit-default dataset: 150 k borrowers, binary target `SeriousDlqin2yrs` (~6.68 % positive), 10 numeric features, **no categorical features**, with real-world missingness (`MonthlyIncome` ~16 %, `NumberOfDependents` ~2 %) and sentinel codes (96/98) in the past-due counters.

The downstream consumer (a separate XGBoost repo) runs a **controlled-imbalance study**: 5 training "levels" (`level_0…level_4`) holding the negative class fixed while scaling the positive class from 20 % → 100 % of available positives, all sharing one hidden test set. The job of *this* repo is to **generate synthetic positive rows** to augment each level's minority class. Splits live at `/srv/datasets/GMSC/splits/level_N/`.

GMSC differs from the prior fork datasets in two ways that drove new code:
- **The target is a single constant** in minority-only data (`SeriousDlqin2yrs == 1`), and there are **no other categorical columns** → the categorical diffusion has a degenerate single-category column.
- **Real missing values** are present in numeric columns → upstream `process_dataset.py` refuses NaNs in numeric columns (it hits a `pdb.set_trace()`), so missingness must be handled in prep.

### 4.2 v1 — `prepare_gmsc.py` + `run_gmsc_splits.sh` (commit `f775a87`)

**Design** (mirrors `prepare_d1.py`): for each `level_N`, read `train.csv`, keep `SeriousDlqin2yrs == 1`, register as `gmsc_l{N}`, train, then generate **5×** the original positive count.

Key GMSC-specific decisions in `prepare_gmsc.py`:
- **Schema**: `target_col_idx=[0]`, `num_col_idx=[1..10]`, **`cat_col_idx=[]`** (all features numeric; the constant target is the only categorical column after the loader concatenates it).
- **NaN handling** (new vs. PaySim, which had none): median-impute `MonthlyIncome` / `NumberOfDependents` on the positive subset *before* registering — otherwise `process_dataset.py` aborts on residual NaNs.
- **Config**: `build_config("d1")` (dim_t 256, batch 256, 3500 epochs, wd 1e-4) — the small-set anti-memorisation profile.
- `register()` is a self-contained copy (GMSC indices) so the PaySim module isn't mutated.

`run_gmsc_splits.sh` trains all 5 levels in parallel across the 4 T4 GPUs (l0–l3 concurrently, l4 after) and generates `5×` per split. **The held-out `test.csv` is never read** — only `level_N/train.csv` positives.

5× targets: 8 020 / 16 040 / 24 065 / 32 085 / 40 105.

### 4.3 The generation bug & fix — `gen_gmsc.py` (commit `fa955f9`)

**Problem discovered at run time:** the shared `generate_paysim_app2.py` filters kept rows with a *string* compare `df[target].astype(str) == "1"`. GMSC's target decodes to the float `1.0`, so `"1.0" != "1"` → **every row rejected**, and reject-sampling looped forever at 0 % yield (observed: 614 sampling rounds, 0 rows kept).

**Fix:** `gen_gmsc.py` filters **numerically** — `pd.to_numeric(df[TARGET], errors="coerce") == 1` (the same robust approach `gen_expv01_app1.py` uses), and writes a clean integer target. Effect: generation dropped from "infinite loop" to **4–18 s per split at ~100 % yield**. The diagnosis was confirmed empirically: a 256-row sample matched `to_numeric == 1` on 256/256 rows but `astype(str) == "1"` on 0/256.

`run_gmsc_splits.sh` was updated to call `gen_gmsc.py` instead of the paysim script.

### 4.4 Overfitting verification — `check_overfit.py` (commit `42eee20`)

Because minority sets are small (1.6 k–8 k rows), memorisation is a real risk. `check_overfit.py` measures, per split:
- **Exact-duplicate rate** of synthetic rows vs. real positives.
- **DCR** (Distance to Closest Record): nearest-neighbour distance (z-scored) from each synthetic row to the real set, vs. a real→real baseline. A memorising model gives ratio ≈ 0; healthy ≈ ≥ 1.
- Per-column mean/std drift.

**Result (v1 pools):** exact duplicates ≈ 0 %, DCR ratio **1.09–1.16** across all levels → no memorisation. The best-EMA checkpoints were also selected well before the final epoch (e.g. l0 at epoch 2267/3500), confirming the anti-memorisation guard worked.

### 4.5 v2 — fidelity overhaul (`prepare_gmsc_v2.py` + `gen_gmsc_v2.py`, commit `42eee20`)

A downstream quality audit found the v1 pools were **easily detectable as synthetic** (real-vs-synthetic discriminator ROC-AUC **0.80–0.95**, ideal 0.50) and gave **no downstream benefit**. Three root-cause defects were traced into the TabDiff data path and fixed:

| Defect (downstream report) | Root cause (in code) | v2 fix |
|---|---|---|
| **Missing-value pattern absent** (top discriminator giveaway: every synthetic borrower "income-complete") | v1 median-imputed NaNs *before* training, so the model never saw missingness | Add binary `MonthlyIncome_isna` / `NumberOfDependents_isna` **categorical** columns; the diffusion learns the *joint* missingness pattern; `gen_gmsc_v2.py` restores `NaN` where the flag fires |
| **Integers became continuous** (age = 49.9) | config `dequant_dist="none"` → `inverse_transform` is a no-op | Set `dequant_dist="uniform"`, `int_dequant_factor=1.0` (model treats counts/age as smooth, floors on inverse) **and** round integer columns on output |
| **Heavy tails over-dispersed** (RevolvingUtil mean 3× too high) | `QuantileTransformer` extrapolates in the tails | Clip each column to its real `[min, max]` (captured per split in a sidecar `gmsc_clip.json`) |
| **Weakened joint correlations** (delinquency counters 0.99 → 0.85) | tiny model underfits joint structure | bump `dim_t` 256 → **512** |

`prepare_gmsc_v2.py` therefore:
- writes 13 columns: target, 10 numerics, 2 `*_isna` flags (`cat_col_idx=[11,12]`);
- median-imputes the *value* (so no residual NaN) but records the missingness in the flag;
- writes a sidecar `gmsc_clip.json` with per-column real `[min,max]`, the integer-column list, and the empirical missing rates;
- config = `build_config("d1")` + `dim_t=512`, `dequant_dist="uniform"`, `int_dequant_factor=1.0`.

`gen_gmsc_v2.py` post-processes each decoded batch: restore `NaN` where a `*_isna` flag rounds to 1, round integer columns, clip to real `[min,max]`, drop the helper flags, emit the original 11-column schema.

**Effect (measured with the downstream `synthetic_quality.py`):**

| level | discriminator AUC v1 → v2 | integers whole v1→v2 | synthetic NaN v1→v2 | income KS v1→v2 |
|------:|:-------------------------:|:--------------------:|:-------------------:|:---------------:|
| 0 | **0.946 → 0.673** | 75 % → 100 % | none → ~16 % | 0.086 → 0.058 |
| 1 | **0.889 → 0.614** | 86 % → 100 % | none → ~17 % | 0.086 → 0.013 |
| 2 | **0.846 → 0.593** | 90 % → 100 % | none → ~17 % | 0.089 → 0.018 |
| 3 | **0.838 → 0.582** | 92 % → 100 % | none → ~18 % | 0.088 → 0.010 |
| 4 | **0.805 → 0.568** | 93 % → 100 % | none → ~16 % | 0.125 → 0.014 |

The data went from "easily detectable" to "hard to detect," best on the data-rich levels. The missingness fix also showed up in training: v2's `DLoss` became non-zero (0.13) whereas v1's was 0.0 (nothing categorical to learn).

### 4.6 Inner-train / 10× variant (`prepare_gmsc_inner.py`, `gen_gmsc_inner.py`, `run_gmsc_inner.sh`, commit `42eee20`)

The downstream pipeline introduced a `train_inner` / `val` split so it can select the augmentation multiplier on a **leakage-free** validation set. For that to be honest, the **synthetic generator must be built from `train_inner` only** — not the full `train.csv` that `val` was carved from.

`prepare_gmsc_inner.py` is the v2 methodology pointed at `level_N/train_inner.csv` (datasets named `gmsc_l{N}_inner`); `gen_gmsc_inner.py` reuses v2's post-processing. `run_gmsc_inner.sh` generates **10×** the inner-train positives:

| split | inner positives | 10× target |
|------:|----------------:|-----------:|
| 0 | 1 283 | 12 830 |
| 1 | 2 566 | 25 660 |
| 2 | 3 850 | 38 500 |
| 3 | 5 134 | 51 340 |
| 4 | 6 417 | 64 170 |

Output: `level_N/synthetic_positives_10x_inner.csv` (this is exactly what the downstream `xgb_gmsc_synthetic.yaml` consumes via `version: 10x_inner`). `val.csv` and the real `test.csv` are never read; the only validation TabDiff itself uses is its own internal 90/10 split of the inner-train positives (for best-EMA selection).

### 4.7 Net outcome

- **v1**: correct 5× pools, no memorisation (DCR ≥ 1.09), but mediocre fidelity (discriminator 0.80–0.95) → no downstream lift.
- **v2**: discriminator 0.57–0.67, integers/ missingness/ tails fixed — a faithful pool.
- **10× inner**: v2 fidelity + leakage-safe provenance + 10× volume for the downstream multiplier sweep.

---

## 5. File reference (fork additions)

**GMSC (this work):**
- `prepare_gmsc.py`, `run_gmsc_splits.sh` — v1 (5×) prep + orchestration *(f775a87)*
- `gen_gmsc.py` — numeric-filter generation fix *(fa955f9)*
- `check_overfit.py` — DCR / duplicate / drift audit *(42eee20)*
- `prepare_gmsc_v2.py`, `gen_gmsc_v2.py`, `run_gmsc_v2.sh` — fidelity overhaul (missingness, int dequant, capacity, clip) *(42eee20)*
- `prepare_gmsc_inner.py`, `gen_gmsc_inner.py`, `run_gmsc_inner.sh` — inner-train / 10× variant *(42eee20)*

**Reused upstream-of-GMSC fork code:**
- `prepare_paysim_app2.py` — `_BASE` config, `DOSE_OVERRIDES`, `build_config`, `register` (GMSC v2 imports `build_config`/`register`)
- `generate_paysim_app2.py` — `build()` (checkpoint load) and `decode()` (reused by all GMSC generators); reject/conditional modes
- `tabdiff/trainer.py`, `tabdiff/main.py` — `--config_path`, `--resume`, held-out loss, `best_ckpt_start_epoch`, `select_loss`

**Upstream (unchanged by the fork):** `tabdiff/models/unified_ctime_diffusion.py`, `tabdiff/models/noise_schedule.py`, `tabdiff/modules/main_modules.py`.

---

## 6. Key lessons encoded in the code

1. **Reject-sampling on a minority-only model is the simplest faithful generator** — but the target-equality filter must be numeric, not string (the GMSC bug).
2. **Don't impute away structure you want to reproduce.** v1 erased missingness; v2 models it explicitly with indicator columns — the single biggest fidelity gain.
3. **Honor column types end-to-end.** `dequant_dist="uniform"` + output rounding restored integers; tail-clipping tamed the quantile-transform extrapolation.
4. **Generator provenance must match the downstream split.** Building from `train_inner` only is what makes the downstream `val` a clean model-selection signal.
5. **Always verify against memorisation** (DCR) *and* detectability (a discriminator AUC) — marginal KS alone (which looked fine in v1) hides joint-distribution and missingness defects.
