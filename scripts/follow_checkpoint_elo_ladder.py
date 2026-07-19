"""Keep the resumable checkpoint Elo ladder current through a target iteration."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATTERN = re.compile(r"plump_v4_iter_(\d+)\.pt")
VOLUME_NAME = "plump-checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="v9_8m_wideppo_seed1")
    parser.add_argument("--target-iteration", type=int, default=6000)
    parser.add_argument("--wait-for-pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def volume_iterations(run_name: str) -> set[int]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "ls",
            VOLUME_NAME,
            f"{run_name}/",
        ],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return {int(value) for value in CHECKPOINT_PATTERN.findall(result.stdout)}


def plotted_iterations(run_name: str) -> set[int]:
    csv_path = REPO_DIR / "checkpoints" / "ladder" / run_name / "elo.csv"
    if not csv_path.exists():
        return set()
    with csv_path.open() as handle:
        return {int(row["iteration"]) for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    out_dir = REPO_DIR / "checkpoints" / "ladder" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "dense_elo.log"

    if args.wait_for_pid is not None:
        while process_exists(args.wait_for_pid):
            time.sleep(args.poll_seconds)

    while True:
        with log_path.open("ab", buffering=0) as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "examples/checkpoint_elo_ladder.py",
                    "--run-name",
                    args.run_name,
                ],
                cwd=REPO_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0:
            try:
                volume = volume_iterations(args.run_name)
                plotted = plotted_iterations(args.run_name)
            except subprocess.CalledProcessError:
                volume = set()
                plotted = set()
            if (
                volume
                and max(volume) >= args.target_iteration
                and volume <= plotted
            ):
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
