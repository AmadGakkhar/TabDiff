"""Local ⇄ cloud launcher for TabDiff SageMaker jobs.

One code path for both backends: a single ``sagemaker.pytorch.PyTorch`` estimator runs the
container entry point (``cloud/entry_train.py`` or ``cloud/entry_generate.py``). The only
things the ``--backend`` toggle changes are the ``instance_type`` and whether inputs/outputs
live on the local filesystem (``file://``, SageMaker *local mode*) or in S3 (managed cloud).

Usage:
  python cloud/launch.py train    --backend local --dataname car_d1 --exp_name car_d1 --debug
  python cloud/launch.py test     --backend cloud --dataname car_d1 --exp_name car_d1 --report
  python cloud/launch.py generate --backend local --gen-script gen_expv01_app1.py \
      --dataname car_d1 --exp_name car_d1 --n_fraud 5 --out synthetic_outputs/car_d1.csv

Everything after the launcher's own flags is passed through verbatim to main.py / the
generation script. See docs/sagemaker.md for AWS setup.
"""
import argparse
import base64
import json
import os
import shutil
import sys
import tarfile
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sm_common  # noqa: E402  (OUT_BASENAME; no import-time side effects)

try:
    import tomllib as _toml  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 (the tabdiff env)
    import tomli as _toml


def load_toml(path):
    with open(path, "rb") as f:
        return _toml.load(f)

DEFAULT_CONFIG = os.path.join(_HERE, "sagemaker_config.toml")
FRAMEWORK_VERSION = "2.0.1"
PY_VERSION = "py310"

# Paths (relative to repo root) never uploaded in the code bundle -- data/checkpoints
# travel as input channels instead, keeping the source tarball small.
_EXCLUDE_REL = {
    "data", "synthetic", "synthetic_outputs", "logs", "debug", "images", "impute",
    os.path.join("eval", "report_runs"),
    os.path.join("tabdiff", "ckpt"), os.path.join("tabdiff", "result"),
    ".git",
}
_EXCLUDE_NAMES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


# --------------------------------------------------------------------------- bundle
def build_code_bundle():
    """Copy code-only files into a temp dir used as the estimator's ``source_dir``.

    Also places requirements.txt at the bundle root so the DLC auto-installs it.
    Returns the bundle path (caller cleans it up).
    """
    bundle = tempfile.mkdtemp(prefix="tabdiff_src_")
    shutil.rmtree(bundle)  # copytree needs a non-existent dest

    def ignore(dirpath, names):
        rel = os.path.relpath(dirpath, _ROOT)
        skip = set()
        for n in names:
            child = os.path.normpath(os.path.join(rel, n)) if rel != "." else n
            if n in _EXCLUDE_NAMES or child in _EXCLUDE_REL:
                skip.add(n)
        return skip

    shutil.copytree(_ROOT, bundle, ignore=ignore)
    shutil.copy2(os.path.join(_HERE, "requirements.txt"),
                 os.path.join(bundle, "requirements.txt"))
    return bundle


# --------------------------------------------------------------------------- backend
def resolve_instance_type(backend, kind, cfg):
    """Map (backend, kind) -> SageMaker instance_type."""
    if backend == "cloud":
        key = "generate_instance_type" if kind == "generate" else "train_instance_type"
        return cfg.get(key, "ml.g4dn.xlarge")
    # local: prefer GPU if the host clearly supports nvidia-docker, unless forced off.
    if cfg.get("force_local_cpu", False):
        return "local"
    if shutil.which("nvidia-smi"):
        return "local_gpu"
    return "local"


def make_session(backend, region):
    import boto3
    boto_session = boto3.Session(region_name=region) if region else boto3.Session()
    if backend == "local":
        from sagemaker.local import LocalSession
        sess = LocalSession(boto_session=boto_session)
        sess.config = {"local": {"local_code": True}}
        return sess
    import sagemaker
    return sagemaker.Session(boto_session=boto_session)


# ------------------------------------------------------------------- inputs & outputs
def prepare_inputs(backend, kind, dataname, exp_name, cfg, session):
    """Return a dict of channel-name -> URI (file:// for local, s3:// for cloud)."""
    channels = {}
    local_dirs = {
        "data": f"data/{dataname}",
        "synthetic": f"synthetic/{dataname}",
    }
    if kind in ("generate", "test"):
        # generation and test need a trained checkpoint mounted.
        ckpt = f"tabdiff/ckpt/{dataname}/{exp_name}"
        if os.path.isdir(ckpt):
            local_dirs["model"] = ckpt

    if backend == "local":
        for name, d in local_dirs.items():
            if os.path.isdir(d):
                channels[name] = "file://" + os.path.abspath(d)
    else:
        bucket = cfg["s3_bucket"]
        base = cfg.get("base_job_name", "tabdiff")
        for name, d in local_dirs.items():
            if os.path.isdir(d):
                channels[name] = session.upload_data(
                    path=d, bucket=bucket, key_prefix=f"{base}/inputs/{dataname}/{name}")
    return channels


def resolve_output_path(backend, cfg, session, dry_run=False):
    if backend == "local":
        if dry_run:
            # Don't litter a real tempdir when we're not going to fit.
            return "file:///tmp/tabdiff_dryrun_output"
        out = tempfile.mkdtemp(prefix="tabdiff_out_")
        return "file://" + out
    base = cfg.get("base_job_name", "tabdiff")
    return f"s3://{cfg.get('s3_bucket', '<bucket>')}/{base}/output"


def fetch_model_artifacts(estimator, backend):
    """Download + extract the job's model.tar.gz into a temp dir; return that dir."""
    model_data = estimator.model_data  # s3:// or file:// or local path
    workdir = tempfile.mkdtemp(prefix="tabdiff_artifacts_")
    if model_data.startswith("s3://"):
        from sagemaker.s3 import S3Downloader
        S3Downloader.download(model_data, workdir)
        tar_path = os.path.join(workdir, os.path.basename(model_data))
    else:
        tar_path = model_data.replace("file://", "")
    extract_dir = os.path.join(workdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)
    return extract_dir


def sync_back_train(extract_dir):
    """Map the container's model-dir subtrees back to the local repo layout."""
    mapping = {
        "ckpt": os.path.join(_ROOT, "tabdiff", "ckpt"),
        "result": os.path.join(_ROOT, "tabdiff", "result"),
        "report_runs": os.path.join(_ROOT, "eval", "report_runs"),
        "impute": os.path.join(_ROOT, "impute"),
    }
    for sub, dest in mapping.items():
        src = os.path.join(extract_dir, sub)
        if os.path.isdir(src):
            os.makedirs(dest, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"[launch] synced {sub} -> {os.path.relpath(dest, _ROOT)}")


def sync_back_generate(extract_dir, out_path):
    """Copy the single generated CSV back to the user's --out path."""
    csv_src = os.path.join(extract_dir, sm_common.OUT_BASENAME)
    if not os.path.exists(csv_src):
        raise SystemExit(f"[launch] generation output not found in artifacts: {csv_src}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    shutil.copy2(csv_src, out_path)
    print(f"[launch] generated CSV -> {out_path}")


# --------------------------------------------------------------------------- estimator
def build_estimator(kind, backend, cfg, session, bundle, instance_type,
                    output_path, hyperparameters):
    from sagemaker.pytorch import PyTorch
    entry = "cloud/entry_generate.py" if kind == "generate" else "cloud/entry_train.py"
    # The SDK rejects an empty role even in local mode (where it is never used). Fall back
    # to a placeholder so the estimator builds; a real cloud fit() with this would fail
    # server-side, and do_dry_run() already warns when role_arn is unset for cloud.
    role = cfg.get("role_arn") or "arn:aws:iam::000000000000:role/placeholder"
    kwargs = dict(
        entry_point=entry,
        source_dir=bundle,
        role=role,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        instance_count=int(cfg.get("instance_count", 1)),
        instance_type=instance_type,
        output_path=output_path,
        hyperparameters=hyperparameters,
        sagemaker_session=session,
        base_job_name=cfg.get("base_job_name", "tabdiff"),
    )
    if cfg.get("image_uri"):
        kwargs["image_uri"] = cfg["image_uri"]  # escape hatch to pin a specific DLC
    if backend == "cloud":
        kwargs["volume_size"] = int(cfg.get("volume_size_gb", 50))
        kwargs["max_run"] = int(cfg.get("max_run_seconds", 86400))
    return PyTorch(**kwargs)


# --------------------------------------------------------------------------- dry run
def do_dry_run(backend, kind, estimator, channels, output_path, hyperparameters, cfg):
    """Report the plan resolved through the real SageMaker SDK, then stop before fit().

    The estimator was built with the same code path a real submission uses, so this
    exercises session creation, estimator args, and DLC image resolution -- it just does
    not upload inputs or launch any container.
    """
    print("=== DRY RUN (SageMaker estimator built; no job submitted) ===")
    print(f"backend        : {backend}")
    print(f"job kind       : {kind}")
    print(f"session type   : {type(estimator.sagemaker_session).__name__}")
    print(f"instance_type  : {estimator.instance_type}")
    try:
        image = estimator.training_image_uri()
    except Exception as e:  # noqa
        image = f"(could not resolve: {e})"
    print(f"resolved image : {image}")
    print(f"entry_point    : {estimator.entry_point}")
    print(f"role_arn       : {estimator.role}")
    print(f"output_path    : {output_path}")
    print(f"input channels : {json.dumps(channels, indent=2)}")
    print(f"hyperparameters: {json.dumps(hyperparameters, indent=2)}")
    if backend == "local":
        ok = shutil.which("docker")
        print(f"docker present : {'yes' if ok else 'NO -- a real local run needs Docker'}")
    else:
        for req in ("s3_bucket", "role_arn"):
            if not cfg.get(req):
                print(f"WARNING: cloud backend requires '{req}' in the config")
        try:
            import boto3
            ident = boto3.client("sts").get_caller_identity()
            print(f"aws identity   : {ident.get('Arn')}")
        except Exception as e:  # noqa
            print(f"aws identity   : could not verify ({e})")


# --------------------------------------------------------------------------- main
def main():
    top = argparse.ArgumentParser(description="TabDiff local/cloud SageMaker launcher")
    top.add_argument("kind", choices=["train", "test", "generate"])
    top.add_argument("--backend", choices=["local", "cloud"], default="local")
    top.add_argument("--config", default=DEFAULT_CONFIG,
                     help="Path to sagemaker_config.toml")
    top.add_argument("--gen-script", dest="gen_script", default=None,
                     help="generate only: the gen_*.py script to run")
    top.add_argument("--dry-run", action="store_true")
    launcher_args, passthrough = top.parse_known_args()

    kind, backend = launcher_args.kind, launcher_args.backend
    if kind == "generate" and not launcher_args.gen_script:
        top.error("generate requires --gen-script")

    cfg = load_toml(launcher_args.config) if os.path.exists(launcher_args.config) else {}
    region = cfg.get("region")

    # main.py wants an explicit --mode; the launcher subcommand supplies it.
    if kind in ("train", "test") and "--mode" not in passthrough:
        passthrough = ["--mode", kind] + passthrough

    dataname = _peek(passthrough, "--dataname")
    exp_name = _peek(passthrough, "--exp_name")
    out_path = _peek(passthrough, "--out")
    if not dataname:
        top.error("--dataname is required")
    if kind == "generate" and not out_path:
        top.error("generate requires --out")

    # base64 so the value is a single token with no quotes/spaces -- SageMaker's training
    # toolkit rebuilds the container command line by splitting on whitespace and dropping
    # quotes, which corrupts a raw JSON string. See sm_common.decode_argv.
    hyperparameters = {
        "argv_json": base64.urlsafe_b64encode(json.dumps(passthrough).encode()).decode()
    }
    if kind == "generate":
        hyperparameters["gen_script"] = launcher_args.gen_script

    instance_type = resolve_instance_type(backend, kind, cfg)
    dry_run = launcher_args.dry_run

    # Same SDK path for dry-run and real run: build the session, code bundle, and estimator.
    # A dry run stops before uploading inputs / calling fit(), so no container launches.
    session = make_session(backend, region)
    bundle = build_code_bundle()
    try:
        output_path = resolve_output_path(backend, cfg, session, dry_run=dry_run)
        est = build_estimator(kind, backend, cfg, session, bundle, instance_type,
                              output_path, hyperparameters)

        if dry_run:
            channels = {k: "file://" + os.path.abspath(v) if backend == "local"
                        else f"s3://{cfg.get('s3_bucket', '<bucket>')}/.../{k}"
                        for k, v in _local_input_dirs(kind, dataname, exp_name).items()}
            do_dry_run(backend, kind, est, channels, output_path, hyperparameters, cfg)
            return

        channels = prepare_inputs(backend, kind, dataname, exp_name, cfg, session)
        print(f"[launch] submitting {kind} job (backend={backend}, instance={instance_type})")
        est.fit(channels)
        extract_dir = fetch_model_artifacts(est, backend)
        if kind == "generate":
            sync_back_generate(extract_dir, out_path)
        else:
            sync_back_train(extract_dir)
        print("[launch] done.")
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def _peek(argv, name):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _local_input_dirs(kind, dataname, exp_name):
    dirs = {"data": f"data/{dataname}", "synthetic": f"synthetic/{dataname}"}
    if kind in ("generate", "test") and exp_name:
        ckpt = f"tabdiff/ckpt/{dataname}/{exp_name}"
        if os.path.isdir(ckpt):
            dirs["model"] = ckpt
    return {k: v for k, v in dirs.items() if os.path.isdir(v)}


if __name__ == "__main__":
    main()
