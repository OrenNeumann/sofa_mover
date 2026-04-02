"""PPO training script for the sofa moving problem."""

import argparse
import warnings

from tqdm import TqdmExperimentalWarning

from sofa_mover.training.config import TrainingConfig
from sofa_mover.training.loop import run_training

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for training."""
    parser = argparse.ArgumentParser(
        description="PPO training for the sofa moving problem."
    )
    parser.add_argument(
        "--obs-mode",
        default="aggressive",
        choices=["baseline", "safe", "aggressive"],
    )
    parser.add_argument("--num-envs", default="auto")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_training(
        TrainingConfig(
            obs_mode=args.obs_mode,
            num_envs=args.num_envs if args.num_envs == "auto" else int(args.num_envs),
        )
    )
