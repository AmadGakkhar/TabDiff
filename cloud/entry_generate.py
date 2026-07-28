"""SageMaker container entry point for TabDiff synthetic-data generation.

Wraps the existing standalone ``gen_*.py`` scripts (e.g. gen_expv01_app1.py, gen_balanced.py)
so they run inside the DLC unchanged. It stages the trained checkpoint + data channels into
TabDiff's relative layout, forces ``--out`` to a fixed path under the model dir and ``--gpu``
to whatever the container actually has, then execs the requested script.

Invoked by the launcher with hyperparameters ``--gen_script <path>`` and
``--argv_json '<json list>'`` (the generation script's own CLI args). The launcher retrieves
the single output file (``gen_output.csv``) from the job output and writes it to the user's
real ``--out`` path.
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

import sm_common  # noqa: E402


def _get_opt(argv, name, default=None):
    """Read the value of ``--name`` from a token list, or return default."""
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def _set_opt(argv, name, value):
    """Set ``--name value`` in a token list, replacing any existing occurrence."""
    value = str(value)
    if name in argv:
        argv[argv.index(name) + 1] = value
    else:
        argv += [name, value]
    return argv


def main():
    outer = argparse.ArgumentParser()
    outer.add_argument('--gen_script', required=True)
    outer.add_argument('--argv_json', required=True)
    outer_args, _ = outer.parse_known_args()

    gen_argv = sm_common.decode_argv(outer_args.argv_json)
    dataname = _get_opt(gen_argv, '--dataname')
    exp_name = _get_opt(gen_argv, '--exp_name')

    device, gpu_index = sm_common.autodetect_device()
    print(f"[sm] entry_generate: script={outer_args.gen_script} dataname={dataname} "
          f"exp_name={exp_name} device={device}", flush=True)

    # Stage inputs: checkpoint + the data/synthetic the gen script reads.
    sm_common.stage_channel(sm_common.CHANNEL_DATA, f"data/{dataname}")
    sm_common.stage_channel(sm_common.CHANNEL_SYNTHETIC, f"synthetic/{dataname}")
    sm_common.stage_channel(sm_common.CHANNEL_MODEL,
                            f"tabdiff/ckpt/{dataname}/{exp_name}")

    # Redirect output into the model dir and pin the device.
    out_path = os.path.join(sm_common.model_dir(), sm_common.OUT_BASENAME)
    _set_opt(gen_argv, '--out', out_path)
    _set_opt(gen_argv, '--gpu', gpu_index)

    cmd = [sys.executable, outer_args.gen_script] + gen_argv
    print(f"[sm] running: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=_ROOT).returncode
    if rc != 0:
        raise SystemExit(f"generation script exited with code {rc}")
    if not os.path.exists(out_path):
        raise SystemExit(f"expected output not produced: {out_path}")
    print(f"[sm] generation output ready: {out_path}", flush=True)


if __name__ == "__main__":
    main()
