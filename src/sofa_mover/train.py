"""PPO training script for the sofa moving problem."""

import argparse
import warnings

from tqdm import TqdmExperimentalWarning

from sofa_mover.training.config import TrainingConfig
from sofa_mover.training.loop import run_training

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

if __name__ == "__main__":
    default_num_envs = TrainingConfig().num_envs
    parser = argparse.ArgumentParser(
        description="PPO training for the sofa moving problem."
    )
    parser.add_argument("--num-envs", type=int, default=default_num_envs)
    args = parser.parse_args()
    run_training(TrainingConfig(num_envs=args.num_envs))
