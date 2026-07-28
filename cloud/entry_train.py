"""SageMaker container entry point for TabDiff training / testing.

Runs *inside* the prebuilt PyTorch DLC (local mode or cloud -- identical code). It:
  1. chdir's to the code root and puts it on sys.path (so TabDiff's relative paths resolve),
  2. stages the input channels into the layout the code expects,
  3. reconstructs the exact ``args`` namespace ``tabdiff.main.main`` wants from ``--argv_json``,
     forcing ``--no_wandb`` and auto-detecting the device,
  4. calls ``main()``,
  5. copies the produced checkpoints / eval results into the model dir for upload.

Invoked by the launcher with hyperparameters ``--argv_json '<json list>'``. Any extra
SageMaker-injected args (e.g. ``--model_dir``) are ignored via ``parse_known_args``.
"""
import argparse
import os
import sys

# Make imports and relative paths work no matter how the toolkit launches this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo / code root, e.g. /opt/ml/code
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

import sm_common  # noqa: E402


def build_inner_parser():
    """The same argument surface as top-level main.py -- kept in sync deliberately."""
    p = argparse.ArgumentParser(description="TabDiff (SageMaker container)")
    p.add_argument('--dataname', type=str, default='adult')
    p.add_argument('--mode', type=str, default='train')
    p.add_argument('--method', type=str, default='tabdiff')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--config_path', type=str, default=None)
    p.add_argument('--y_only', action='store_true')
    p.add_argument('--non_learnable_schedule', action='store_true')
    p.add_argument('--num_samples_to_generate', type=int, default=None)
    p.add_argument('--ckpt_path', type=str, default=None)
    p.add_argument('--report', action='store_true')
    p.add_argument('--num_runs', type=int, default=20)
    p.add_argument('--impute', action='store_true')
    p.add_argument('--trial_start', type=int, default=0)
    p.add_argument('--trial_size', type=int, default=50)
    p.add_argument('--resample_rounds', type=int, default=1)
    p.add_argument('--impute_condition', type=str, default="x_t")
    p.add_argument('--y_only_model_path', type=str, default=None)
    p.add_argument('--w_num', type=float, default=0.6)
    p.add_argument('--w_cat', type=float, default=0.6)
    return p


def main():
    outer = argparse.ArgumentParser()
    outer.add_argument('--argv_json', required=True)
    outer_args, _ = outer.parse_known_args()
    inner_argv = sm_common.decode_argv(outer_args.argv_json)

    args = build_inner_parser().parse_args(inner_argv)

    # SageMaker jobs never have wandb creds; always log locally.
    args.no_wandb = True
    # The launcher cannot know the instance's GPU state; decide it here.
    device, gpu_index = sm_common.autodetect_device()
    args.device = device
    args.gpu = gpu_index
    print(f"[sm] entry_train: mode={args.mode} dataname={args.dataname} "
          f"exp_name={args.exp_name} device={device}", flush=True)

    # Stage inputs into TabDiff's expected relative layout.
    sm_common.stage_channel(sm_common.CHANNEL_DATA, f"data/{args.dataname}")
    sm_common.stage_channel(sm_common.CHANNEL_SYNTHETIC, f"synthetic/{args.dataname}")
    # test / impute modes need a pre-trained checkpoint mounted at the standard path.
    exp_for_ckpt = args.exp_name or (
        'non_learnable_schedule' if args.non_learnable_schedule else 'learnable_schedule')
    sm_common.stage_channel(sm_common.CHANNEL_MODEL,
                            f"tabdiff/ckpt/{args.dataname}/{exp_for_ckpt}")

    from tabdiff.main import main as tabdiff_main
    tabdiff_main(args)

    # Ship everything the run may have produced. The launcher maps these back to the
    # matching local dirs (ckpt -> tabdiff/ckpt, result -> tabdiff/result, etc.).
    sm_common.copy_out("tabdiff/ckpt", "ckpt")
    sm_common.copy_out("tabdiff/result", "result")
    sm_common.copy_out("eval/report_runs", "report_runs")
    sm_common.copy_out("impute", "impute")


if __name__ == "__main__":
    main()
