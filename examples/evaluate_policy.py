"""Evaluate a checkpoint over deterministic round banks and full schedules."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.evaluation import (
    DealBank,
    check_round_policy_compatibility,
    evaluate_full_games,
    evaluate_paired,
    evaluate_policy,
)
from plump.modeling import SCHEMA_VERSION
from plump.policies import HeuristicPolicy, ModelPolicy, RandomPolicy
from plump.rounds import rules_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run position-controlled and full-game evaluation.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--comparison", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--deals-per-configuration", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--full-games", type=int, default=32)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = ModelPolicy.from_checkpoint(args.checkpoint, device=args.device, name="candidate")
    opponent = HeuristicPolicy() if args.opponent == "heuristic" else RandomPolicy(args.seed + 500)
    bank = DealBank.generate(
        deals_per_configuration=args.deals_per_configuration,
        seed=args.seed + 1_000,
    )
    report = evaluate_policy(
        candidate,
        opponent,
        bank,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    comparisons = {}
    baselines = {
        "random": RandomPolicy(args.seed + 2_000),
        "heuristic": HeuristicPolicy(),
    }
    for value in args.comparison:
        if "=" not in value:
            raise SystemExit("--comparison must use NAME=CHECKPOINT.")
        name, raw_path = value.split("=", maxsplit=1)
        baselines[name] = ModelPolicy.from_checkpoint(raw_path, device=args.device, name=name)
    for name, baseline in baselines.items():
        comparisons[name] = asdict(
            evaluate_paired(
                candidate,
                baseline,
                opponent,
                bank,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                batch_size=args.batch_size,
            )
        )
    full_games = {
        str(players): asdict(
            evaluate_full_games(
                candidate,
                opponent,
                num_players=players,
                games=args.full_games,
                seed=args.seed + players * 10_000,
            )
        )
        for players in (3, 4, 5)
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rules_fingerprint": rules_fingerprint(),
        "checkpoint": str(args.checkpoint.resolve()),
        "round_policy_compatibility": asdict(
            check_round_policy_compatibility(
                candidate,
                opponent,
                seed=args.seed + 50_000,
            )
        ),
        "position_controlled_round_evaluation": asdict(report),
        "paired_comparisons": comparisons,
        "full_game_performance": full_games,
        "claim": "Measured game performance; not proof of game-optimal play.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"macro_relative_reward={report.macro_relative_reward:.4f} "
        f"ci=[{report.relative_reward_ci_low:.4f},{report.relative_reward_ci_high:.4f}]"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
