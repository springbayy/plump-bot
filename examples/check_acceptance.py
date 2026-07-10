"""Require three independently trained v4 checkpoints to beat one legacy policy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.evaluation import DealBank, evaluate_paired
from plump.modeling import SCHEMA_VERSION
from plump.policies import HeuristicPolicy, ModelPolicy
from plump.rounds import rules_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the three-seed legacy acceptance bar.")
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("acceptance.json"))
    parser.add_argument("--deals-per-configuration", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.candidate) != 3:
        raise SystemExit("Exactly three independently trained --candidate checkpoints are required.")
    bank = DealBank.generate(
        deals_per_configuration=args.deals_per_configuration,
        seed=args.seed + 10_000,
    )
    reports = []
    for index, path in enumerate(args.candidate):
        candidate = ModelPolicy.from_checkpoint(path, device=args.device, name=f"seed-{index + 1}")
        if not isinstance(candidate, ModelPolicy):
            raise SystemExit(f"Candidate {path} is not a schema-v4 observation model.")
        legacy = ModelPolicy.from_checkpoint(args.legacy, device=args.device, name="legacy")
        report = evaluate_paired(
            candidate,
            legacy,
            HeuristicPolicy(),
            bank,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index,
            batch_size=args.batch_size,
        )
        reports.append(report)
        print(
            f"candidate={path} delta={report.macro_relative_reward_delta:.4f} "
            f"ci=[{report.ci_low:.4f},{report.ci_high:.4f}]"
        )
    passed = all(report.ci_low > 0.0 for report in reports)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rules_fingerprint": rules_fingerprint(),
        "legacy": str(args.legacy.resolve()),
        "candidates": [str(path.resolve()) for path in args.candidate],
        "passed": passed,
        "criterion": "positive paired relative-reward 95% lower bound for all three training seeds",
        "reports": [asdict(report) for report in reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"passed={passed} report={args.output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
