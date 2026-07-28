"""Shared helpers for the SageMaker entry points and launcher.

The TabDiff code assumes a fixed working directory with relative paths
(``data/<dataname>/``, ``synthetic/<dataname>/``, ``tabdiff/ckpt/<dataname>/<exp_name>/``).
SageMaker instead mounts input channels at ``/opt/ml/input/data/<channel>/`` and uploads
``/opt/ml/model/`` out to ``output_path``. These helpers bridge the two so the model code
can run unchanged inside the container, whether the job runs in SageMaker *local mode*
(Docker on this machine) or on a managed cloud instance -- the container never knows which.
"""
import base64
import binascii
import json
import os
import shutil

# Channel names used consistently by the launcher (inputs) and the entry points (staging).
CHANNEL_DATA = "data"
CHANNEL_SYNTHETIC = "synthetic"
CHANNEL_MODEL = "model"

# Fixed basename the generation entry point writes and the launcher fetches back.
OUT_BASENAME = "gen_output.csv"


def channel_dir(name):
    """Return the local path of an input channel, or None if it was not provided.

    SageMaker exposes each input channel as ``SM_CHANNEL_<NAME>`` (uppercased).
    """
    return os.environ.get(f"SM_CHANNEL_{name.upper()}")


def model_dir():
    """Directory whose contents SageMaker uploads to ``output_path`` (S3 or file://)."""
    return os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


def stage_channel(name, dest_rel):
    """Copy an input channel into the relative layout the TabDiff code expects.

    Copies (not symlinks) so the code may freely read/write under ``dest_rel`` even when
    the channel mount is read-only. No-op if the channel was not provided. Existing files
    at the destination are left in place (``dirs_exist_ok``).
    """
    src = channel_dir(name)
    if not src or not os.path.isdir(src):
        return None
    os.makedirs(os.path.dirname(os.path.abspath(dest_rel)), exist_ok=True)
    shutil.copytree(src, dest_rel, dirs_exist_ok=True)
    print(f"[sm] staged channel '{name}': {src} -> {dest_rel}", flush=True)
    return dest_rel


def copy_out(src_rel, subpath=""):
    """Copy a produced artifact (file or dir) into the model dir for upload.

    ``subpath`` names the location under the model dir; defaults to the basename of
    ``src_rel``. Silently skips if the source does not exist.
    """
    if not os.path.exists(src_rel):
        print(f"[sm] copy_out: source missing, skipped: {src_rel}", flush=True)
        return None
    dest = os.path.join(model_dir(), subpath or os.path.basename(os.path.normpath(src_rel)))
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.isdir(src_rel):
        shutil.copytree(src_rel, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src_rel, dest)
    print(f"[sm] copied artifact: {src_rel} -> {dest}", flush=True)
    return dest


def autodetect_device():
    """Return (device_str, gpu_index) based on what the container actually has.

    The launcher does not know the instance's GPU state, so the entry points decide at
    runtime: ``('cuda:0', 0)`` when CUDA is visible, else ``('cpu', -1)``.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda:0", 0
    return "cpu", -1


def decode_argv(raw):
    """Decode the ``--argv_json`` hyperparameter back into a list of CLI tokens.

    The launcher base64-encodes ``json.dumps(argv)`` so the value survives SageMaker's
    toolkit, which rebuilds the container command line by splitting on whitespace and
    dropping quotes -- a raw JSON string arrives corrupted (``[--mode,`` instead of
    ``["--mode",``). We base64-decode first, then fall back to treating ``raw`` as plain
    JSON for backward compatibility / local runs.
    """
    raw = raw.strip()
    val = None
    try:
        decoded = base64.urlsafe_b64decode(raw.encode()).decode()
        val = json.loads(decoded)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        val = json.loads(raw)
    if isinstance(val, str):
        val = json.loads(val)
    if not isinstance(val, list):
        raise ValueError(f"argv_json did not decode to a list: {val!r}")
    return [str(x) for x in val]
