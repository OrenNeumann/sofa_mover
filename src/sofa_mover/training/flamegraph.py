"""Generate a flame graph for the default training entrypoint."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the default training run with py-spy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/default_training_flamegraph.svg"),
        help="Where to write the SVG flame graph.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1200,
        help="Seconds to sample the training process.",
    )
    parser.add_argument("--rate", type=int, default=100, help="Samples per second.")
    parser.add_argument(
        "--native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include native stack frames from C/C++ extensions.",
    )
    parser.add_argument(
        "--nonblocking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use py-spy's lower-overhead nonblocking mode.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default="offline",
        help="WANDB_MODE value for the profiled training run.",
    )
    args = parser.parse_args()
    if args.native and args.nonblocking:
        raise ValueError(
            "py-spy cannot collect native stacks while running in nonblocking mode."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    command = ["py-spy", "record", "-o", str(args.output)]
    command += ["-d", str(args.duration), "-r", str(args.rate)]
    if args.native:
        command.append("--native")
    if args.nonblocking:
        command.append("--nonblocking")
    command += ["--", sys.executable, "-m", "sofa_mover.training.train"]

    print("Running:", " ".join(command))
    subprocess.run(
        command, check=True, env={**os.environ, "WANDB_MODE": args.wandb_mode}
    )
    print(f"Wrote flame graph to {args.output}")


if __name__ == "__main__":
    main()
