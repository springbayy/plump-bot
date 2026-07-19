"""One-shot migration of a local training run to the Modal volume.

Uploads the newest checkpoint, the league snapshot files it references,
best.pt, run_config.json, and copies of metrics.csv / events.jsonl truncated
to the resume iteration (so the resumed run does not append duplicate rows
after already-logged iterations).

Run AFTER stopping local training:
    .venv/bin/python scripts/modal_migrate_v9.py
    .venv/bin/python scripts/modal_migrate_v9.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VOLUME_NAME = "plump-checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "v9_8m_wideppo_seed1",
    )
    parser.add_argument(
        "--remote-run-name",
        default=None,
        help="Directory name on the volume (default: local run dir name).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _truncate_metrics(source: Path, target: Path, resume_iteration: int) -> int:
    kept = 0
    with source.open() as infile, target.open("w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(next(reader))
        for row in reader:
            if int(row[0]) <= resume_iteration:
                writer.writerow(row)
                kept += 1
    return kept


def _truncate_events(source: Path, target: Path, resume_iteration: int) -> int:
    kept = 0
    with source.open() as infile, target.open("w") as outfile:
        for line in infile:
            if not line.strip():
                continue
            if json.loads(line).get("iteration", 0) <= resume_iteration:
                outfile.write(line)
                kept += 1
    return kept


def main() -> None:
    import torch

    args = parse_args()
    run_dir = args.run_dir
    run_name = args.remote_run_name or run_dir.name

    checkpoints = sorted(
        run_dir.glob("plump_v4_iter_*.pt"),
        key=lambda file: int(file.stem.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        raise SystemExit(f"No plump_v4_iter_*.pt checkpoints in {run_dir}")
    latest = checkpoints[-1]
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    resume_iteration = int(payload["iteration"])
    snapshot_names = [
        Path(stored).name
        for stored in payload.get("league", {}).get("snapshot_paths", [])
    ]
    print(f"resume checkpoint: {latest.name} (iteration {resume_iteration})")
    print(f"league snapshots referenced: {snapshot_names}")

    uploads: list[tuple[Path, str]] = [(latest, latest.name)]
    for name in snapshot_names:
        source = run_dir / name
        if not source.exists():
            raise SystemExit(f"League snapshot missing locally: {source}")
        if source != latest:
            uploads.append((source, name))
    for optional in ("best.pt", "run_config.json"):
        source = run_dir / optional
        if source.exists():
            uploads.append((source, optional))

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        metrics = run_dir / "metrics.csv"
        if metrics.exists():
            truncated = staging / "metrics.csv"
            kept = _truncate_metrics(metrics, truncated, resume_iteration)
            uploads.append((truncated, "metrics.csv"))
            print(f"metrics.csv: kept {kept} rows <= iteration {resume_iteration}")
        events = run_dir / "events.jsonl"
        if events.exists():
            truncated = staging / "events.jsonl"
            kept = _truncate_events(events, truncated, resume_iteration)
            uploads.append((truncated, "events.jsonl"))
            print(f"events.jsonl: kept {kept} rows <= iteration {resume_iteration}")

        total_mb = sum(source.stat().st_size for source, _ in uploads) / 1e6
        for source, remote_name in uploads:
            print(
                f"  {remote_name:<28} {source.stat().st_size / 1e6:8.1f} MB"
                f" -> {run_name}/{remote_name}"
            )
        print(f"total: {total_mb:.0f} MB -> volume '{VOLUME_NAME}'")
        if args.dry_run:
            print("dry run: nothing uploaded")
            return

        import modal

        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        with volume.batch_upload(force=True) as batch:
            for source, remote_name in uploads:
                batch.put_file(source, f"/{run_name}/{remote_name}")
        print("upload complete")


if __name__ == "__main__":
    main()
