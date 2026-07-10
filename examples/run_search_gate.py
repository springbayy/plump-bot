"""Sweep root-search breadth and write the schema-v4 Gate 0 report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.evaluation import DealBank, evaluate_paired
from plump.modeling import SCHEMA_VERSION
from plump.policies import HeuristicPolicy, ModelPolicy, RandomPolicy
from plump.rounds import rules_fingerprint
from plump.search import RootSearchPolicy, SearchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled schema-v4 search gate.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("search_gate.json"))
    parser.add_argument("--player-counts", default="3,4,5")
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--deals-per-configuration", type=int, default=32)
    parser.add_argument("--breadths", default="4,8,16,32")
    parser.add_argument("--forward-pass-budget", type=int, default=2_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--max-cell-regression", type=float, default=0.5)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument(
        "--cross-opponent",
        action="append",
        choices=("heuristic", "random"),
        default=[],
        help="Also measure the selected breadth against this opponent model.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _opponent(name: str, seed: int):
    return HeuristicPolicy() if name == "heuristic" else RandomPolicy(seed)


def main() -> None:
    args = parse_args()
    breadths = sorted(set(_csv_ints(args.breadths)))
    if not breadths or breadths[0] < 1:
        raise SystemExit("--breadths must contain positive integers.")
    bank = DealBank.generate(
        player_counts=_csv_ints(args.player_counts),
        hand_sizes=_csv_ints(args.hand_sizes),
        deals_per_configuration=args.deals_per_configuration,
        seed=args.seed + 1_000,
    )
    opponent = _opponent(args.opponent, args.seed + 2_000)
    reports = []
    for breadth in breadths:
        base = ModelPolicy.from_checkpoint(
            args.checkpoint,
            device=args.device,
            name=f"raw-d{breadth}",
        )
        baseline = ModelPolicy.from_checkpoint(
            args.checkpoint,
            device=args.device,
            name="raw",
        )
        if not isinstance(base, ModelPolicy):
            raise SystemExit("Search Gate 0 requires an observation-model checkpoint.")
        search = RootSearchPolicy(
            base,
            opponent,
            config=SearchConfig(
                min_determinizations=breadth,
                max_determinizations=breadth,
                batch_determinizations=breadth,
                forward_pass_budget=args.forward_pass_budget,
                seed=args.seed,
            ),
            name=f"search-d{breadth}",
        )
        paired = evaluate_paired(
            search,
            baseline,
            opponent,
            bank,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + breadth,
            batch_size=args.evaluation_batch_size,
        )
        reports.append((breadth, paired))
        print(
            f"breadth={breadth} delta={paired.macro_relative_reward_delta:.4f} "
            f"ci=[{paired.ci_low:.4f},{paired.ci_high:.4f}] "
            f"worst_cell={paired.worst_cell_delta:.4f}"
        )

    peak = max(report.macro_relative_reward_delta for _, report in reports)
    tolerance = 0.05 * max(abs(peak), 1e-9)
    selected_breadth, selected = next(
        (breadth, report)
        for breadth, report in reports
        if report.macro_relative_reward_delta >= peak - tolerance
    )
    gate_passed = selected.passes_gate(max_cell_regression=args.max_cell_regression)
    cross_opponent_reports = {}
    for index, opponent_name in enumerate(dict.fromkeys(args.cross_opponent)):
        if opponent_name == args.opponent:
            continue
        cross_opponent = _opponent(opponent_name, args.seed + 30_000 + index)
        cross_base = ModelPolicy.from_checkpoint(
            args.checkpoint,
            device=args.device,
            name=f"raw-cross-{opponent_name}",
        )
        cross_baseline = ModelPolicy.from_checkpoint(
            args.checkpoint,
            device=args.device,
            name="raw",
        )
        if not isinstance(cross_base, ModelPolicy):
            raise SystemExit("Cross-opponent search evaluation requires an observation model.")
        cross_search = RootSearchPolicy(
            cross_base,
            cross_opponent,
            config=SearchConfig(
                min_determinizations=selected_breadth,
                max_determinizations=selected_breadth,
                batch_determinizations=selected_breadth,
                forward_pass_budget=args.forward_pass_budget,
                seed=args.seed,
            ),
        )
        cross_opponent_reports[opponent_name] = asdict(
            evaluate_paired(
                cross_search,
                cross_baseline,
                cross_opponent,
                bank,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 40_000 + index,
                batch_size=args.evaluation_batch_size,
            )
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rules_fingerprint": rules_fingerprint(),
        "checkpoint": str(args.checkpoint.resolve()),
        "opponent": opponent.name,
        "deal_bank_seed": bank.seed,
        "deals_per_configuration": args.deals_per_configuration,
        "forward_pass_budget": args.forward_pass_budget,
        "selection_rule": "smallest breadth within 5% of peak macro relative-reward delta",
        "selected_breadth": selected_breadth,
        "gate_passed": gate_passed,
        "max_cell_regression": args.max_cell_regression,
        "selected": asdict(selected),
        "short_round_cell_deltas": {
            key: value
            for key, value in selected.cell_deltas.items()
            if "-3c-" in key
        },
        "cross_opponent_reports": cross_opponent_reports,
        "sweep": [
            {"breadth": breadth, "paired": asdict(report)}
            for breadth, report in reports
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"selected_breadth={selected_breadth} gate_passed={gate_passed} "
        f"report={args.output}"
    )


if __name__ == "__main__":
    main()
