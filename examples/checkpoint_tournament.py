"""One-shot randomized tournament over training checkpoints: replay everything.

Unlike the incremental ladder, every invocation REPLAYS all games: no cached
matchup is ever reused, the checkpoint set is frozen at launch (up to the most
recent checkpoint on the volume, starting at --min-iteration), and a fresh
seed randomizes both the deal bank and the matchup draw, so each rerun yields
new, independent scores. Policies can be non-transitively strong — beating one
opponent says little about another — so instead of a fixed opponent structure
every checkpoint meets --rounds uniformly random opponents (one random perfect
matching per round: everyone plays exactly one matchup per round), each played
in both seats back-to-back so the focal-seat advantage cancels per pair. With
many randomized pairings the law of large numbers evens out matchup luck. A
weighted least-squares fit with a global focal-bias term pools every margin
into one rating per checkpoint, centered so 0 = the population mean, i.e. a
rating reads "expected points per round vs a uniformly random checkpoint".
Self-play probes on a second bank track bid behavior exactly as in the ladder.

    .venv/bin/python examples/checkpoint_tournament.py            # full fresh run

Outputs (rewritten continuously): tournament.csv, the plot elo_ladder.png, and
the raw stream tournament_results.jsonl (truncated at launch; diagnostic only,
never read back as a cache).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_elo_ladder import (
    HEURISTIC_KEY,
    MIGRATION_ITERATION,
    PolicyCache,
    _macro_cell_mean,
    _spread_order,
    checkpoint_iterations,
    fit_margin_rating,
    pull_checkpoint,
    volume_iterations,
)

from plump.evaluation import DealBank, evaluate_policy

REPO_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh randomized all-vs-all tournament of training checkpoints.",
    )
    parser.add_argument("--run-name", default="v9_8m_wideppo_seed1")
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=100,
        help="Ignore checkpoints trained for fewer iterations than this.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=12,
        help=(
            "Random-matching rounds: every checkpoint plays one uniformly "
            "random opponent per round, in both seats."
        ),
    )
    parser.add_argument(
        "--deals-per-configuration",
        type=int,
        default=3,
        help="Deals per (players, hand size) cell of the shared bank.",
    )
    parser.add_argument("--probe-deals", type=int, default=2)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Tournament seed randomizing the deal bank and the matchup draw. "
            "Defaults to the current time: every rerun is a fresh sample."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--publish-every",
        type=int,
        default=4,
        help="Rewrite tournament.csv and the plot after this many matchups.",
    )
    return parser.parse_args()


def _bisection_tree(count: int) -> list[tuple[int, int]]:
    """Spanning-tree edges over indices 0..count-1, coarse-to-fine.

    The first edge joins the two endpoints; every later edge attaches an
    interval midpoint to an already-connected index. Every prefix of the list
    is therefore one connected graph spanning the full index range, so the
    rating fit — which only rates the component connected to the earliest
    checkpoint — identifies every checkpoint played so far from the very
    first publish, at a resolution that refines as edges land.
    """

    if count < 2:
        return []
    edges = [(0, count - 1)]
    intervals = deque([(0, count - 1)])
    while intervals:
        low, high = intervals.popleft()
        middle = (low + high) // 2
        if middle in (low, high):
            continue
        edges.append((low, middle))
        intervals.append((low, middle))
        intervals.append((middle, high))
    return edges


def build_schedule(
    keys: list[str],
    rounds: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    """A bisection-tree bootstrap with interleaved probes, then random rounds.

    Work items are ("matchup", focal, table) or ("probe", key). Ratings are
    only identified within a connected matchup graph, and random matchings
    alone take thousands of games to connect every checkpoint, so a spanning
    tree built by bisection runs first: connected at every prefix and spanning
    the full iteration range from the first edge, so intermediate publishes
    show a full-range curve that densifies instead of a stub. Behavior probes
    are woven between tree edges so both plot panels fill together.

    Then every random-matching round pairs each checkpoint with one uniformly
    random opponent (with an odd field, one randomly chosen checkpoint sits
    the round out), so all checkpoints accumulate games at the same rate
    against uniformly random opposition. Every matchup pair plays both
    directions back-to-back so the focal-seat advantage cancels per pair.
    """

    tree = _bisection_tree(len(keys))
    probe_order = [keys[index] for index in _spread_order(len(keys))]
    work: list[tuple[str, ...]] = []
    for position, (a, b) in enumerate(tree):
        work.append(("matchup", keys[a], keys[b]))
        work.append(("matchup", keys[b], keys[a]))
        if position < len(probe_order):
            work.append(("probe", probe_order[position]))
    work.extend(("probe", key) for key in probe_order[len(tree):])

    pairs: list[tuple[str, str]] = []
    for _ in range(rounds):
        shuffled = rng.sample(keys, len(keys))
        pairs.extend(zip(shuffled[0::2], shuffled[1::2]))
    rng.shuffle(pairs)
    for a, b in pairs:
        work.append(("matchup", a, b))
        work.append(("matchup", b, a))
    return work


def probe_row(key: str, policies: PolicyCache, probe_bank: DealBank, args) -> dict:
    started = time.monotonic()
    policy = policies.get(key)
    report = evaluate_policy(
        policy,
        policy,
        probe_bank,
        bootstrap_samples=8,
        seed=args.seed + 1,
        batch_size=args.batch_size,
    )
    results = report.results
    return {
        "kind": "behavior",
        "key": key,
        "rounds": len(results),
        "avg_bid": _macro_cell_mean(results, lambda r: r.bid_value),
        "bid_hit_rate": _macro_cell_mean(results, lambda r: r.bid_hit),
        "zero_bid_rate": _macro_cell_mean(results, lambda r: float(r.bid_value == 0)),
        "all_bid_rate": _macro_cell_mean(
            results,
            lambda r: float(r.bid_value == r.spec.hand_size),
        ),
        "elapsed_sec": round(time.monotonic() - started, 1),
    }


def matchup_row(
    focal: str,
    table: str,
    policies: PolicyCache,
    bank: DealBank,
    args,
) -> dict:
    started = time.monotonic()
    report = evaluate_policy(
        policies.get(focal),
        policies.get(table),
        bank,
        bootstrap_samples=8,
        seed=args.seed + 2,
        batch_size=args.batch_size,
    )
    return {
        "kind": "matchup",
        "focal": focal,
        "table": table,
        "rounds": report.rounds,
        "macro_relative_reward": report.macro_relative_reward,
        "elapsed_sec": round(time.monotonic() - started, 1),
    }


def write_outputs(
    matchup_rows: list[dict],
    behavior_rows: list[dict],
    out_dir: Path,
    header: str,
) -> None:
    rating, focal_bias = fit_margin_rating(matchup_rows)
    behavior_by_key = {row["key"]: row for row in behavior_rows}
    matchups_by_key: dict[str, int] = {}
    opponents_by_key: dict[str, set[str]] = {}
    for row in matchup_rows:
        for key in (row["focal"], row["table"]):
            matchups_by_key[key] = matchups_by_key.get(key, 0) + 1
        opponents_by_key.setdefault(row["focal"], set()).add(row["table"])
        opponents_by_key.setdefault(row["table"], set()).add(row["focal"])

    members = set(rating) | set(behavior_by_key)
    points = sorted((int(key.split("_")[1]), key) for key in members)
    if not points:
        return

    def fmt(value: float | None) -> str:
        return "" if value is None else format(value, ".4f")

    csv_path = out_dir / "tournament.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w") as handle:
        handle.write(
            "iteration,margin_rating,avg_bid,bid_hit_rate,zero_bid_rate,"
            "all_bid_rate,matchups,opponents\n"
        )
        for iteration, key in points:
            behavior = behavior_by_key.get(key, {})
            handle.write(
                f"{iteration},{fmt(rating.get(key))},"
                f"{fmt(behavior.get('avg_bid'))},"
                f"{fmt(behavior.get('bid_hit_rate'))},"
                f"{fmt(behavior.get('zero_bid_rate'))},"
                f"{fmt(behavior.get('all_bid_rate'))},"
                f"{matchups_by_key.get(key, 0)},"
                f"{len(opponents_by_key.get(key, set()))}\n"
            )
    csv_tmp.replace(csv_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    behavior_points = [
        (iteration, behavior_by_key[key])
        for iteration, key in points
        if key in behavior_by_key
    ]
    handles = []
    if behavior_points:
        bid_x = [iteration for iteration, _ in behavior_points]
        twin = top.twinx()
        (line_bid,) = top.plot(
            bid_x,
            [row["avg_bid"] for _, row in behavior_points],
            marker="o",
            markersize=2.5,
            color="tab:blue",
            label="avg trick bid",
        )
        handles = [line_bid]
        for field, color, label in (
            ("bid_hit_rate", "tab:green", "bid hit %"),
            ("zero_bid_rate", "tab:orange", "0-bid %"),
            ("all_bid_rate", "tab:red", "bid-all %"),
        ):
            (line,) = twin.plot(
                bid_x,
                [100.0 * row[field] for _, row in behavior_points],
                marker="o",
                markersize=2.5,
                color=color,
                label=label,
            )
            handles.append(line)
        top.set_ylabel("average trick bid", color="tab:blue")
        twin.set_ylabel("% of bids (hit / zero / all)")
        twin.set_ylim(bottom=0)
    top.axvline(MIGRATION_ITERATION, color="gray", linestyle=":", linewidth=1)
    top.set_title(header)
    if handles:
        top.legend(handles=handles, loc="upper right", fontsize=8, ncols=2)
    top.grid(alpha=0.3)

    rating_points = [
        (iteration, rating[key])
        for iteration, key in points
        if key in rating
    ]
    if rating_points:
        bottom.plot(
            *zip(*rating_points),
            marker="o",
            markersize=3,
            color="tab:blue",
            label="points rating (vs uniform random checkpoint)",
        )
    bottom.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    bottom.axvline(MIGRATION_ITERATION, color="gray", linestyle=":", linewidth=1)
    bottom.text(
        0.01,
        0.03,
        f"focal-seat bias removed: {focal_bias:+.3f} pts/round",
        transform=bottom.transAxes,
        fontsize=7,
        color="gray",
    )
    bottom.set_ylabel("expected points/round vs uniform ckpt (mean = 0)")
    bottom.set_xlabel("training iteration")
    if rating_points:
        bottom.legend(loc="lower right", fontsize=8)
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    # The plot publishes to the same path the old incremental ladder used —
    # it is the benchmark image; the ladder that once owned it is retired.
    png_path = out_dir / "elo_ladder.png"
    png_tmp = png_path.with_suffix(".png.tmp")
    figure.savefig(png_tmp, dpi=150, format="png")
    plt.close(figure)
    png_tmp.replace(png_path)


def main() -> None:
    args = parse_args()
    if args.seed is None:
        args.seed = int(time.time())
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    archive_dir = REPO_DIR / "checkpoints" / args.run_name
    out_dir = REPO_DIR / "checkpoints" / "ladder" / args.run_name
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    local_paths = checkpoint_iterations(archive_dir)
    pulled_paths = checkpoint_iterations(ckpt_dir)
    volume_iters = volume_iterations(args.run_name)
    volume_set = set(volume_iters)

    # The field is frozen now: checkpoints landing during the tournament are
    # picked up by the next rerun, which replays everything anyway.
    ladder = sorted(
        iteration
        for iteration in set(local_paths) | set(pulled_paths) | set(volume_iters)
        if iteration >= args.min_iteration
    )
    if len(ladder) < 2:
        raise SystemExit("Need at least two checkpoints above --min-iteration.")
    keys = [f"iter_{iteration:05d}" for iteration in ladder]
    print(
        f"tournament: {len(keys)} checkpoints, iters {ladder[0]}-{ladder[-1]}, "
        f"seed {args.seed}, {args.rounds} random-matching rounds"
    )

    paths: dict[str, Path] = {}
    for iteration in ladder:
        key = f"iter_{iteration:05d}"
        if iteration in local_paths:
            paths[key] = local_paths[iteration]
        elif iteration in pulled_paths:
            paths[key] = pulled_paths[iteration]
        else:
            paths[key] = ckpt_dir / f"plump_v4_iter_{iteration:05d}.pt"

    def fetch_missing(key: str, dest: Path) -> None:
        iteration = int(key.split("_")[1])
        if iteration not in volume_set:
            raise FileNotFoundError(f"iter {iteration} absent from the volume")
        pull_checkpoint(args.run_name, iteration, dest)

    rng = random.Random(args.seed)
    schedule = build_schedule(keys, args.rounds, rng)
    bank = DealBank.generate(
        player_counts=(3, 4, 5),
        hand_sizes=tuple(range(3, 11)),
        deals_per_configuration=args.deals_per_configuration,
        seed=args.seed,
    )
    probe_bank = DealBank.generate(
        player_counts=(3, 4, 5),
        hand_sizes=tuple(range(3, 11)),
        deals_per_configuration=args.probe_deals,
        seed=args.seed + 1,
    )
    total_matchups = sum(1 for item in schedule if item[0] == "matchup")
    rounds_per_matchup = sum(deal.spec.num_players**2 for deal in bank.deals)
    print(
        f"schedule: {total_matchups} directed matchups x {rounds_per_matchup} "
        f"rounds = {total_matchups * rounds_per_matchup:,} rounds, plus "
        f"{len(keys)} behavior probes interleaved with the bootstrap tree",
        flush=True,
    )

    policies = PolicyCache(paths, args.device, capacity=6, fetch_missing=fetch_missing)
    header = (
        f"Randomized tournament, seed {args.seed}: {len(keys)} checkpoints, "
        f"{args.rounds} rounds of random matchings"
    )
    results_path = out_dir / "tournament_results.jsonl"
    matchup_rows: list[dict] = []
    behavior_rows: list[dict] = []
    probes_done = 0
    matchups_done = 0
    with results_path.open("w") as stream:
        for position, item in enumerate(schedule, start=1):
            if item[0] == "probe":
                key = item[1]
                row = probe_row(key, policies, probe_bank, args)
                behavior_rows.append(row)
                probes_done += 1
                progress = (
                    f"[probe {probes_done}/{len(keys)}] {key}: "
                    f"avg_bid {row['avg_bid']:.2f} "
                    f"hit {100 * row['bid_hit_rate']:.1f}% "
                    f"zero {100 * row['zero_bid_rate']:.1f}% "
                    f"all {100 * row['all_bid_rate']:.1f}%"
                )
            else:
                _, focal, table = item
                row = matchup_row(focal, table, policies, bank, args)
                matchup_rows.append(row)
                matchups_done += 1
                progress = (
                    f"[{matchups_done}/{total_matchups}] {focal} vs {table}: "
                    f"reward {row['macro_relative_reward']:+.3f}"
                )
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if position % args.publish_every == 0 or position == len(schedule):
                write_outputs(matchup_rows, behavior_rows, out_dir, header)
            print(f"{progress} ({row['elapsed_sec']}s)", flush=True)
    print(f"tournament complete: {out_dir / 'elo_ladder.png'}")


if __name__ == "__main__":
    main()
