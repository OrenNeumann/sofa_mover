"""Generate a flame graph for the default training entrypoint."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_OUTPUT_PATH = Path("output/default_training_flamegraph.svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the default training run with py-spy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the SVG flame graph.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1200,
        help="Seconds to sample the training process.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=100,
        help="Samples per second.",
    )
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
    parser.add_argument(
        "--py-spy-bin",
        default="py-spy",
        help="Profiler executable to invoke.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch training.",
    )
    return parser


def build_py_spy_command(
    *,
    py_spy_bin: str,
    output: Path,
    duration: int,
    rate: int,
    native: bool,
    nonblocking: bool,
    python_bin: str,
) -> list[str]:
    if native and nonblocking:
        raise ValueError(
            "py-spy cannot collect native stacks while running in nonblocking mode."
        )

    command = [
        py_spy_bin,
        "record",
        "-o",
        str(output),
        "-d",
        str(duration),
        "-r",
        str(rate),
    ]
    if native:
        command.append("--native")
    if nonblocking:
        command.append("--nonblocking")
    command.extend(
        [
            "--",
            python_bin,
            "-m",
            "sofa_mover.training.train",
        ]
    )
    return command


def build_training_env(*, wandb_mode: str) -> dict[str, str]:
    env = dict(os.environ)
    env["WANDB_MODE"] = wandb_mode
    return env


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    command = build_py_spy_command(
        py_spy_bin=args.py_spy_bin,
        output=args.output,
        duration=args.duration,
        rate=args.rate,
        native=args.native,
        nonblocking=args.nonblocking,
        python_bin=args.python_bin,
    )
    env = build_training_env(wandb_mode=args.wandb_mode)

    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)
    print(f"Wrote flame graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
