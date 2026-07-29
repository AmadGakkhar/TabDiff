# TabDiff × AWS SageMaker — Integration Changes

This document details every change made to enable running TabDiff training and generation on
**AWS SageMaker** (both *local mode* and *managed cloud instances*), and exactly what changed
for **local running**. It is a companion to [`sagemaker.md`](sagemaker.md), which is the
end-user usage guide; this file is the design/changelog record of *what was modified and why*.

All changes landed in commit `37e2123` ("Add SageMaker integration with entry points for
training and generation").

---

## 1. Goal & design principle

Run TabDiff training and batch generation on a managed SageMaker GPU instance, submitted
**from the laptop**, billed **only while the job runs** (per-second, instance auto-torn-down),
with **no changes to the TabDiff model code itself**.

The guiding principle is **one code path for three execution modes**:

| Mode | What runs | Billing |
|---|---|---|
| Local (original) | `python main.py …` directly | your machine |
| SageMaker **local mode** | same code in the DLC container via Docker on your machine | your machine |
| SageMaker **cloud** | same code in the DLC container on a managed instance | per-second while running |

The toggle between them is a single `--backend {local,cloud}` flag (or a `BACKEND` env var in
the pipeline scripts). Jobs run as SageMaker **Training Jobs** — used for *both* train/test and
batch generate. There is **no persistent endpoint**, which is what keeps "charged only while
training" true.

The model runs inside AWS's **prebuilt PyTorch DLC** (`torch 2.0.1 / py310`) with
`cloud/requirements.txt` layered on top — **no custom Docker image is built**.

---

## 2. New files (all under `cloud/`)

### `cloud/launch.py` — the launcher / backend toggle
The single entry you run **instead of** `python main.py`. Responsibilities:

- **Backend resolution** (`resolve_instance_type`, `make_session`): `local` →
  `LocalSession` + `local`/`local_gpu` instance (auto-picks GPU if `nvidia-smi` is present,
  unless `force_local_cpu`); `cloud` → real `sagemaker.Session` + the configured
  `ml.*` instance type.
- **Code bundling** (`build_code_bundle`): copies the repo into a temp `source_dir`,
  **excluding** bulk/output dirs (`data`, `synthetic`, `synthetic_outputs`, `logs`, `debug`,
  `images`, `impute`, `eval/report_runs`, `tabdiff/ckpt`, `tabdiff/result`, `.git`,
  `__pycache__`, …). Keeps the uploaded tarball small; bulk data travels as **input channels**
  instead of source.
- **Input staging** (`prepare_inputs`): only the single `data/<dataname>` (+ `synthetic/…`,
  and the checkpoint for `test`/`generate`) a job needs is provided as channels —
  `file://` URIs for local, uploaded to S3 for cloud.
- **Estimator** (`build_estimator`): one `sagemaker.pytorch.PyTorch` estimator pointed at the
  container entry point; image URI auto-resolved from `FRAMEWORK_VERSION`/`PY_VERSION`.
- **Argument pass-through**: only `--backend`, `--config`, `--gen-script`, `--dry-run` belong
  to the launcher; **everything else is forwarded verbatim** to `main.py` / the gen script,
  encoded as the `argv_json` hyperparameter (see §5.1).
- **Artifact sync-back** (`fetch_model_artifacts`, `sync_back_train`, `sync_back_generate`):
  downloads the job's `model.tar.gz` and maps its subtrees back into the local repo layout —
  `ckpt → tabdiff/ckpt`, `result → tabdiff/result`, `report_runs → eval/report_runs`,
  `impute → impute` — so **downstream steps can't tell where training happened**. Generation's
  single CSV is copied to the user's real `--out` path.
- **`--dry-run`**: goes through the entire SDK path (session, estimator, image URI resolution,
  identity check) but stops before `fit()` — nothing is uploaded, no container launches.

### `cloud/entry_train.py` — container entry for train / test / impute
Runs *inside* the DLC. It:
1. `chdir`s to the code root and puts it on `sys.path` so TabDiff's **relative paths resolve**;
2. stages input channels into the expected relative layout
   (`data/<dataname>`, `synthetic/<dataname>`, `tabdiff/ckpt/<dataname>/<exp>`);
3. rebuilds the exact `args` namespace `tabdiff.main.main` expects from `--argv_json`,
   **forcing `--no_wandb`** and **auto-detecting the device** (GPU if visible, else CPU);
4. calls `tabdiff_main(args)`;
5. copies produced checkpoints / eval results into the model dir for upload.

It keeps its own copy of `main.py`'s argument parser, deliberately kept in sync.

### `cloud/entry_generate.py` — container entry for generation
Wraps any existing standalone `gen_*.py` script unchanged. Stages the trained checkpoint +
data channels, **forces `--out`** to a fixed path in the model dir and **`--gpu`** to whatever
the container actually has, then execs the requested script via `subprocess`. The launcher
retrieves the single `gen_output.csv` and writes it to the user's real `--out`.

### `cloud/sm_common.py` — shared container helpers
Bridges TabDiff's fixed-relative-path assumption to SageMaker's channel layout:
- `channel_dir` / `stage_channel` — copy `SM_CHANNEL_<NAME>` mounts into the relative dirs the
  code expects (copies, not symlinks, so read-only mounts are safe to write over).
- `copy_out` — copy produced artifacts into the model dir (`/opt/ml/model`) for upload.
- `autodetect_device` — `('cuda:0', 0)` if CUDA visible, else `('cpu', -1)`.
- `decode_argv` — decode the `--argv_json` hyperparameter back into a token list (see §5.1).

### `cloud/requirements.txt` — extra deps layered on the DLC
Mirrors the pip section of `tabdiff.yaml` **minus the torch stack** (already in the DLC).
Includes **`wandb`** — required because `tabdiff/main.py` imports it unconditionally at module
load (see §5.2).

### `cloud/sagemaker_config.toml` — per-user AWS settings
`region`, `s3_bucket`, `role_arn`, instance types, volume/runtime limits, and optional
overrides (`force_local_cpu`, pinned `image_uri`). Filled in for this account (see §5.3).

---

## 3. Changes to existing (tracked) files

### `run_car_app2.sh` — `BACKEND` toggle added
The pipeline gained a `BACKEND` env var, **defaulting to the original local behavior when
unset**. Both the train and generate stages branch:

- **`BACKEND` unset** → *identical to before*: config-swap-in-place, `python -u main.py …`,
  restore config; `python -u gen_expv01_app1.py …`.
- **`BACKEND=local` or `BACKEND=cloud`** → route through the launcher:
  - Train: `python -u cloud/launch.py train --backend "$BACKEND" … --config_path "$MODEL_CFG"`
    (no in-place config swap — the per-model config is passed explicitly).
  - Generate: `python -u cloud/launch.py generate --backend "$BACKEND"
    --gen-script gen_expv01_app1.py …` (**`--gpu` omitted** — the container manages it).

Usage: `BACKEND=cloud ./run_car_app2.sh`. The same one-line substitution pattern applies to the
other `run_*.sh` scripts if you want to route them through SageMaker.

### `tabdiff.yaml` — two deps added
Added `sagemaker` and `boto3` so the **launcher-side** (your machine) can build sessions and
estimators. (These are for submitting jobs; they are not needed inside the container.)

### `tabdiff/main.py` — **unchanged**
The model code was intentionally **not touched**. The `wandb` import problem was solved by
adding the dependency to the container, not by editing the model.

---

## 4. Impact on local running

**Default local behavior is unchanged.** Running `python main.py …` directly, or
`./run_car_app2.sh` with no `BACKEND` set, behaves exactly as before — same config-swap logic,
same commands, same outputs. The SageMaker path is strictly additive and opt-in.

New optional local capability: **SageMaker local mode** (`--backend local`) runs the *same
container code* under Docker on your machine — useful to validate the container path before
paying for a cloud run. It needs Docker (+ NVIDIA Container Toolkit for `local_gpu`), and still
pulls the DLC image from ECR once, so AWS credentials are required even locally.

Because the container `chdir`s to the code root and stages channels into TabDiff's expected
relative paths, and because the launcher syncs artifacts back into the exact local layout
(`tabdiff/ckpt/<dataname>/<exp_name>/`), **downstream steps are identical regardless of where a
job ran.** A cloud-trained checkpoint is indistinguishable from a locally trained one.

---

## 5. Bugs found & fixed during bring-up

These were the failures hit on the first real cloud runs, in order, and their fixes.

### 5.1 `argv_json` hyperparameter mangling  *(code fix)*
**Symptom:** `json.decoder.JSONDecodeError: Expecting value` — the container received
`[--mode,` instead of `["--mode", …]`.
**Cause:** SageMaker's training toolkit rebuilds the container command line by splitting on
whitespace and stripping quotes, corrupting any raw JSON string passed as a hyperparameter.
**Fix:** encode the argv as a single quote-free, space-free **base64** token.
- `cloud/launch.py`: `argv_json = base64.urlsafe_b64encode(json.dumps(passthrough).encode())`
- `cloud/sm_common.py` `decode_argv`: base64-decode first, with a plain-JSON fallback for
  backward compatibility / local runs.

### 5.2 Missing `wandb` in the container  *(dependency fix)*
**Symptom:** `ModuleNotFoundError: No module named 'wandb'` at
`tabdiff/main.py` import time.
**Cause:** `--no_wandb` only disables logging at *runtime*; `import wandb` at the **top of the
module** runs regardless. The container didn't have it.
**Fix:** added `wandb` to `cloud/requirements.txt` (model code left unchanged).

### 5.3 Config / identity fixes  *(configuration, not code)*
- **Role ARN:** `CreateTrainingJob` requires an **IAM role** ARN
  (`arn:aws:iam::…:role/…`), not the STS **assumed-role session** ARN of the SSO caller.
  Set `role_arn` to the domain's execution role
  `arn:aws:iam::652765021279:role/ascend-ml-sagemaker-exec`.
- **Region:** the SageMaker domain and role live in **us-east-1**; the config (and the `ats`
  SSO profile) had `us-west-2`. Set `region = "us-east-1"`.
- **Bucket:** set `s3_bucket = "sagemaker-us-east-1-652765021279"` (the default SageMaker
  bucket, which the execution role already has S3 access to by naming convention).
- **Auth:** submit under the SSO profile — `aws sso login --profile ats` then
  `export AWS_PROFILE=ats` — so the launcher targets account `652765021279`, not the laptop's
  stale `default` credentials.

---

## 6. How to run

Prereqs (one-time): `pip install sagemaker boto3`; AWS creds configured; for cloud, a bucket +
execution role in the config; for local mode, Docker running.

```bash
export AWS_PROFILE=ats            # target the right account

# Preview only — builds the estimator, submits nothing
python cloud/launch.py train --backend cloud --dataname car_d1 --exp_name car_d1 --dry-run

# Train on a managed GPU instance
python cloud/launch.py train --backend cloud --dataname car_d1 --exp_name car_d1

# Test / report
python cloud/launch.py test --backend cloud --dataname car_d1 --exp_name car_d1 --report

# Generate synthetic data from a trained checkpoint
python cloud/launch.py generate --backend cloud \
    --gen-script gen_expv01_app1.py \
    --dataname car_d1 --exp_name car_d1 --n_fraud 625 \
    --out synthetic_outputs/car_d1_fraud.csv

# Route a whole pipeline through SageMaker
BACKEND=cloud ./run_car_app2.sh
```

`--no_wandb` and `--gpu` are forced by the container and can be omitted.

---

## 7. Billing model

- Training Jobs bill **per-second, only for "Billable seconds"** (the "Training in progress"
  window, which includes the in-container `pip install`). Instance provisioning and image
  download are **not** billed. The instance is **auto-terminated** when the job ends.
- No persistent endpoint ⇒ nothing bills between jobs.
- Reference: the first `car_d1` run was 1892 billable seconds on `ml.g4dn.xlarge`
  (us-east-1, SageMaker training rate ~$0.7364/hr) ≈ **$0.39**.
- Only lingering cost is trivial **S3 storage** of inputs + `model.tar.gz` outputs.
- Cost levers for longer runs: **managed spot training** (~60–70% savings, interruptible), or
  a smaller instance — both are config/launcher changes, not model changes.

**Watch-out:** SageMaker **Studio** apps (the notebook IDE) bill continuously while running and
are *independent* of Training Jobs. Since jobs are submitted from the laptop, Studio never needs
to be opened; if it was, stop its running apps.

---

## 8. Known limits

- Multi-file generation scripts that use `--out_dir` instead of a single `--out` CSV are not
  wrapped yet; the generate path expects one `--out` file.
- No real-time inference endpoint — batch Training Jobs only.
- First local-mode run downloads the (large) DLC image once.
- `cloud/entry_train.py` keeps its own copy of `main.py`'s arg parser; if `main.py` gains new
  flags, that parser must be updated to match.
