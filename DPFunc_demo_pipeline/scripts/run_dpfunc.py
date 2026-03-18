#!/usr/bin/env python3
"""Run original DPFunc scripts in an isolated demo workspace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    # .../DPFunc_demo_pipeline/scripts/run_dpfunc.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPFunc training/prediction in demo workspace")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Call DPFunc_main.py")
    train.add_argument("--workspace", type=Path, required=True)
    train.add_argument("--ontology", choices=["mf", "bp", "cc"], required=True)
    train.add_argument("--gpu", type=int, default=0)
    train.add_argument("--epoch", type=int, default=15)
    train.add_argument("--pre-name", type=str, default="demo_model")
    train.add_argument("--python", type=str, default=sys.executable)

    pred = sub.add_parser("predict", help="Call DPFunc_pred.py")
    pred.add_argument("--workspace", type=Path, required=True)
    pred.add_argument("--ontology", choices=["mf", "bp", "cc"], required=True)
    pred.add_argument("--gpu", type=int, default=0)
    pred.add_argument("--pre-name", type=str, default="demo_model")
    pred.add_argument("--python", type=str, default=sys.executable)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    workspace = args.workspace.resolve()

    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    if args.mode == "train":
        cmd = [
            args.python,
            str(root / "DPFunc_main.py"),
            "-d",
            args.ontology,
            "-n",
            str(args.gpu),
            "-e",
            str(args.epoch),
            "-p",
            args.pre_name,
        ]
        run_cmd(cmd, cwd=workspace)
        return

    cmd = [
        args.python,
        str(root / "DPFunc_pred.py"),
        "-d",
        args.ontology,
        "-n",
        str(args.gpu),
        "-p",
        args.pre_name,
    ]
    run_cmd(cmd, cwd=workspace)


if __name__ == "__main__":
    main()
