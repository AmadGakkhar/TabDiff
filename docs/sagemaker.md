# Running TabDiff on AWS SageMaker (local mode ⇄ cloud)

TabDiff training and generation can run either **on this machine** (SageMaker *local mode*,
inside Docker) or **on a managed SageMaker instance** — through the *same* container code.
The only thing that changes is `--backend local` vs `--backend cloud`. This keeps one code
path: the model code, the entry points, and the input/output staging are identical either way.

Jobs run as SageMaker **Training Jobs** (used for both `train`/`test` and batch `generate`);
there is no persistent endpoint. The container is AWS's **prebuilt PyTorch DLC**
(torch 2.0.1 / py310) with `cloud/requirements.txt` installed on top — no custom image build.

## What lives where

| File | Role |
|---|---|
| `cloud/launch.py` | The launcher / toggle you run instead of `python main.py` |
| `cloud/entry_train.py` | Container entry for `main.py` train/test |
| `cloud/entry_generate.py` | Container entry that wraps a `gen_*.py` script |
| `cloud/sm_common.py` | Channel staging, device autodetect, artifact copy helpers |
| `cloud/requirements.txt` | pip deps installed on the DLC |
| `cloud/sagemaker_config.toml` | Your region / bucket / role / instance types |

## One-time setup

### 1. Install the SDK and Docker tooling
```
pip install sagemaker boto3
```
- **Local mode** needs **Docker** and **docker compose** installed and the daemon running.
- For GPU local mode (`local_gpu`), also install the **NVIDIA Container Toolkit**. If it's
  not present, the launcher automatically falls back to CPU (`local`).

### 2. Configure AWS credentials + region (required even for local mode)
Local mode still pulls the prebuilt DLC image from AWS ECR (cached after the first pull), so
valid credentials are needed:
```
aws configure          # or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
```

### 3. Cloud backend only: bucket + execution role
```
aws s3 mb s3://YOUR-TABDIFF-BUCKET
```
Create a SageMaker execution role that trusts `sagemaker.amazonaws.com` and has SageMaker +
S3 access (the AWS-managed `AmazonSageMakerFullAccess` policy plus read/write on your bucket
is sufficient for a first run). Copy its ARN.

### 4. Fill in `cloud/sagemaker_config.toml`
Set at least `region`; for cloud runs also `s3_bucket` and `role_arn`. Adjust
`train_instance_type` / `generate_instance_type` as needed.

## Usage

Preview the plan without submitting anything:
```
python cloud/launch.py train --backend local  --dataname car_d1 --exp_name car_d1 --debug --dry-run
python cloud/launch.py train --backend cloud  --dataname car_d1 --exp_name car_d1 --dry-run
```
A dry run goes through the **same SageMaker SDK path** as a real run: it creates the session
(`LocalSession` for local, `Session` for cloud), builds the `PyTorch` estimator, and resolves
the actual DLC image URI — it just stops before uploading inputs or calling `fit()`, so no
container launches and nothing is written to S3. It therefore needs `sagemaker`/`boto3`
installed (setup step 1) and `region` set. It also reports whether Docker is present (local)
or verifies your AWS identity (cloud). Example local output:
```
session type   : LocalSession
instance_type  : local_gpu
resolved image : 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310
```

### Train
```
# local mode (Docker on this machine)
python cloud/launch.py train --backend local --dataname car_d1 --exp_name car_d1

# managed cloud instance
python cloud/launch.py train --backend cloud --dataname car_d1 --exp_name car_d1
```
The trained checkpoint is synced back to `tabdiff/ckpt/<dataname>/<exp_name>/` exactly as a
local run would leave it, so downstream steps don't care where training happened.

### Test / report
```
python cloud/launch.py test --backend cloud --dataname car_d1 --exp_name car_d1 --report
```
Report outputs are synced back under `eval/report_runs/`.

### Generate
```
python cloud/launch.py generate --backend local \
    --gen-script gen_expv01_app1.py \
    --dataname car_d1 --exp_name car_d1 --n_fraud 625 \
    --out synthetic_outputs/car_d1_fraud.csv
```
Any `gen_*.py` that takes `--dataname/--exp_name/--out` works via `--gen-script`. The launcher
sends the trained checkpoint as an input channel, runs the script in the container, and copies
the produced CSV back to your `--out` path. `--out` and `--gpu` are managed automatically inside
the container, so you don't set `--gpu`.

## Everything after the launcher flags is passed straight through

`--backend`, `--config`, `--gen-script`, and `--dry-run` belong to the launcher; **all other
arguments are forwarded verbatim** to `main.py` (train/test) or the generation script. So
`--debug`, `--resume`, `--deterministic`, `--non_learnable_schedule`, `--config_path`, `--report`,
`--num_runs`, imputation flags, etc. all behave as they do locally.

`--no_wandb` and the device (`--gpu`) are forced by the container (jobs have no wandb creds and
pick GPU/CPU based on the instance), so you can omit them.

## Using it from the run_*.sh pipelines

The pipelines read a `BACKEND` env var (default `local` = original behavior when unset routes
through the launcher's local mode). To send a whole pipeline through SageMaker:
```
BACKEND=cloud ./run_car_app2.sh
```
The same one-line substitution pattern (`python main.py ...` → `python cloud/launch.py train
--backend "$BACKEND" ...`, and `python gen_*.py ...` → `python cloud/launch.py generate
--backend "$BACKEND" --gen-script gen_*.py ...`) applies to the other `run_*.sh` scripts.

## Notes & limits

- The uploaded code bundle excludes `data/`, `synthetic/`, `tabdiff/ckpt/`, `tabdiff/result/`,
  `logs/`, etc. — bulk data travels as input channels, not source. Only the single
  `data/<dataname>` + `synthetic/<dataname>` a job needs is uploaded (cloud backend).
- First local-mode run downloads the DLC image (large, one-off).
- Multi-file generation scripts that use `--out_dir` instead of a single `--out` are not wrapped
  yet; the generation path expects one `--out` CSV.
- No real-time inference endpoint — batch Training Jobs only.
