"""Dense checkpoint ladder: score-margin rating and bid behavior over training.

Every checkpoint is measured identically: it plays the checkpoints at fixed
rung offsets (1, 5, 25, 60, 120, 240 by default — short links for local
precision, long links for cross-era stiffness), in BOTH seats of each matchup,
on one shared deal bank where policies play every hand from every bidding
position. No randomness in the schedule, so no checkpoint gets a luckier
opponent mix than another. A weighted least-squares rating in the game's own
units (my points minus the table mean per round) is fitted over the matchup
graph together with a global focal-seat bias term, and ratings are centered on
the population mean: a rating reads "expected points per round against an
opponent drawn uniformly from the run's checkpoints 0..N". Fast self-play
probes on a second fixed bank additionally track bid behavior per checkpoint:
average bid, bid hit rate, zero-bid rate, and bid-everything rate.

Designed to run on the Mac while training continues on Modal:
    .venv/bin/python examples/checkpoint_elo_ladder.py            # sync + play + plot
    .venv/bin/python examples/checkpoint_elo_ladder.py --no-sync  # offline re-fit/plot

Finished directed matchups are cached in pairings.jsonl, and elo.csv plus the
plot are republished after every result. Random opponents are append-stable, so
re-runs only play matchups introduced by new checkpoints; interrupting the
backlog loses at most the currently active matchup.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from plump.evaluation import DealBank, evaluate_policy
from plump.policies import HeuristicPolicy, ModelPolicy

REPO_DIR = Path(__file__).resolve().parents[1]
VOLUME_NAME = "plump-checkpoints"
CHECKPOINT_PATTERN = re.compile(r"plump_v4_iter_(\d+)\.pt")
HEURISTIC_KEY = "heuristic"
# Iteration where the run migrated from the Mac to Modal (plot annotation).
MIGRATION_ITERATION = 3301


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-robin ladder of training checkpoints with a Bradley-Terry Elo fit.",
    )
    parser.add_argument("--run-name", default="v9_8m_wideppo_seed1")
    parser.add_argument(
        "--min-iter-gap",
        type=int,
        default=1,
        help="Minimum iteration spacing between ladder rungs (1 = every checkpoint).",
    )
    parser.add_argument("--deals-per-configuration", type=int, default=6)
    parser.add_argument("--bank-seed", type=int, default=424242)
    parser.add_argument("--eval-seed", type=int, default=99)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--offsets",
        default="1,5,25,60,120,240",
        help=(
            "Comma-separated rung distances every checkpoint is paired at, "
            "in both directions: identical comparison structure for all."
        ),
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate both policies as the focal policy for every matchup.",
    )
    parser.add_argument(
        "--heuristic-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also play every checkpoint against the heuristic baseline.",
    )
    parser.add_argument(
        "--probe-deals",
        type=int,
        default=2,
        help="Deals per configuration for the self-play bid-behavior probes.",
    )
    parser.add_argument(
        "--publish-every",
        type=int,
        default=1,
        help="Rewrite elo.csv and the plot after this many new directed matchups.",
    )
    parser.add_argument(
        "--max-pairings",
        type=int,
        default=None,
        help="Play at most this many new pairings this invocation (0 = fit/plot only).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the Modal volume listing/download; use cached checkpoints only.",
    )
    return parser.parse_args()


def checkpoint_iterations(directory: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def volume_iterations(run_name: str) -> list[int]:
    listing = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "ls", VOLUME_NAME, f"{run_name}/"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(
        int(match.group(1))
        for match in CHECKPOINT_PATTERN.finditer(listing.stdout)
    )


def pull_checkpoint(run_name: str, iteration: int, dest: Path) -> None:
    name = f"plump_v4_iter_{iteration:05d}.pt"
    print(f"  pulling {name} from volume ...", flush=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    partial.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "modal", "volume", "get", "--force",
            VOLUME_NAME, f"{run_name}/{name}", str(partial),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    partial.replace(dest)


def select_ladder(iterations: list[int], min_gap: int) -> list[int]:
    """Greedy stride from the start: deterministic and stable under appends."""

    ladder: list[int] = []
    for iteration in sorted(iterations):
        if not ladder or iteration - ladder[-1] >= min_gap:
            ladder.append(iteration)
    # The newest checkpoint always joins so the curve tracks the live run.
    if iterations and ladder[-1] != max(iterations):
        ladder.append(max(iterations))
    return ladder


def _spread_order(count: int) -> list[int]:
    """Progressive bisection: endpoints first, then midpoints, breadth-first."""

    if count <= 0:
        return []
    order = [0]
    seen = {0}
    if count > 1 and (count - 1) not in seen:
        order.append(count - 1)
        seen.add(count - 1)
    intervals = deque([(0, count - 1)])
    while intervals:
        low, high = intervals.popleft()
        middle = (low + high) // 2
        if middle in (low, high):
            continue
        if middle not in seen:
            seen.add(middle)
            order.append(middle)
        intervals.append((low, middle))
        intervals.append((middle, high))
    return order


def build_pairings(
    ladder: list[int],
    offsets: tuple[int, ...],
    *,
    bidirectional: bool = True,
    heuristic_anchors: bool = False,
) -> list[tuple[str, str]]:
    if not offsets or any(offset < 1 for offset in offsets):
        raise ValueError("offsets must be positive rung distances.")
    keys = [f"iter_{iteration:05d}" for iteration in ladder]

    # Every rung is paired at the same fixed offsets, so away from the run's
    # boundaries all checkpoints share an identical comparison structure.
    # Short offsets give local precision; long ones tie eras together so the
    # fit does not rely on long transitive chains. Offset 1 runs first (it
    # alone connects every rung to the rated component), then longer scales,
    # each in bisection order over rungs so coverage spans the full range
    # almost immediately.
    spread = _spread_order(len(keys))
    anchor_edges = (
        [(keys[index], HEURISTIC_KEY) for index in spread]
        if heuristic_anchors
        else []
    )
    model_edges: list[tuple[str, str]] = []
    seen_model_edges: set[tuple[str, str]] = set()
    for offset in sorted(set(offsets)):
        for index in spread:
            if index + offset < len(keys):
                pair = (keys[index], keys[index + offset])
                if pair not in seen_model_edges:
                    seen_model_edges.add(pair)
                    model_edges.append(pair)

    # The two directions of an edge run back-to-back: the lone focal seat
    # against a table of opponent clones carries a systematic advantage
    # (~+0.1 pts/round), and only pairs measured in both directions let the
    # rating fit separate that seat bias from true strength differences.
    pairings: list[tuple[str, str]] = []
    for focal, table in anchor_edges + model_edges:
        pairings.append((focal, table))
        if bidirectional:
            pairings.append((table, focal))
    return pairings


class PolicyCache:
    """LRU of loaded checkpoint policies; the heuristic anchor is free."""

    def __init__(
        self,
        paths: dict[str, Path],
        device: str | None,
        capacity: int = 4,
        fetch_missing: Callable[[str, Path], None] | None = None,
    ):
        self.paths = paths
        self.device = device
        self.capacity = capacity
        self.fetch_missing = fetch_missing
        self._cache: OrderedDict[str, ModelPolicy] = OrderedDict()

    def get(self, key: str):
        if key == HEURISTIC_KEY:
            return HeuristicPolicy()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.paths[key]
        if not path.exists():
            if self.fetch_missing is None:
                raise FileNotFoundError(path)
            self.fetch_missing(key, path)
        try:
            policy = ModelPolicy.from_checkpoint(
                path,
                device=self.device,
                greedy=False,
                event_length_buckets=(8, 16, 32, 64),
                batch_packing="numpy",
                lean_action_forward=True,
                name=key,
            )
        except (EOFError, OSError, RuntimeError, ValueError):
            # An interrupted Modal download can leave a plausible-looking but
            # truncated .pt file. Re-fetch it atomically when possible.
            if self.fetch_missing is None:
                raise
            print(f"  cached {path.name} is invalid; pulling it again", flush=True)
            path.unlink(missing_ok=True)
            self.fetch_missing(key, path)
            policy = ModelPolicy.from_checkpoint(
                path,
                device=self.device,
                greedy=False,
                event_length_buckets=(8, 16, 32, 64),
                batch_packing="numpy",
                lean_action_forward=True,
                name=key,
            )
        self._cache[key] = policy
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return policy


def load_cache_rows(cache_path: Path) -> list[dict]:
    rows: list[dict] = []
    if cache_path.exists():
        with cache_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def bank_fingerprint(args: argparse.Namespace) -> dict:
    return {
        "matchup_outcome_schema": 2,
        "bank_seed": args.bank_seed,
        "deals_per_configuration": args.deals_per_configuration,
        "eval_seed": args.eval_seed,
        "player_counts": [3, 4, 5],
        "hand_sizes": list(range(3, 11)),
    }


def behavior_fingerprint(args: argparse.Namespace) -> dict:
    return {
        "behavior_schema": 1,
        "probe_seed": args.bank_seed + 1,
        "probe_deals_per_configuration": args.probe_deals,
        "eval_seed": args.eval_seed,
        "player_counts": [3, 4, 5],
        "hand_sizes": list(range(3, 11)),
    }


def _macro_cell_mean(results, value) -> float:
    """Uniform mean over (players, hand size) cells: fair across configurations."""

    by_cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for result in results:
        by_cell[(result.spec.num_players, result.spec.hand_size)].append(value(result))
    return sum(
        sum(values) / len(values) for values in by_cell.values()
    ) / len(by_cell)


def probe_behavior(
    key: str,
    policies: "PolicyCache",
    probe_bank: DealBank,
    args: argparse.Namespace,
) -> dict:
    """Self-play probe: every seat runs the checkpoint, so all bids are its own."""

    started = time.monotonic()
    policy = policies.get(key)
    report = evaluate_policy(
        policy,
        policy,
        probe_bank,
        bootstrap_samples=8,
        seed=args.eval_seed,
        batch_size=args.batch_size,
    )
    results = report.results
    return {
        "kind": "behavior",
        "key": key,
        **behavior_fingerprint(args),
        "rounds": len(results),
        "avg_bid": _macro_cell_mean(results, lambda r: r.bid_value),
        "bid_hit_rate": _macro_cell_mean(results, lambda r: r.bid_hit),
        "zero_bid_rate": _macro_cell_mean(results, lambda r: float(r.bid_value == 0)),
        "all_bid_rate": _macro_cell_mean(
            results,
            lambda r: float(r.bid_value == r.spec.hand_size),
        ),
        "elapsed_sec": round(time.monotonic() - started, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def play_pairing(
    focal_key: str,
    table_key: str,
    policies: PolicyCache,
    bank: DealBank,
    args: argparse.Namespace,
) -> dict:
    started = time.monotonic()
    report = evaluate_policy(
        policies.get(focal_key),
        policies.get(table_key),
        bank,
        bootstrap_samples=8,
        seed=args.eval_seed,
        batch_size=args.batch_size,
    )
    wins = sum(1 for result in report.results if result.relative_reward > 0)
    draws = sum(1 for result in report.results if result.relative_reward == 0)
    losses = len(report.results) - wins - draws
    return {
        "focal": focal_key,
        "table": table_key,
        **bank_fingerprint(args),
        "rounds": len(report.results),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "macro_relative_reward": report.macro_relative_reward,
        "ci_low": report.relative_reward_ci_low,
        "ci_high": report.relative_reward_ci_high,
        "elapsed_sec": round(time.monotonic() - started, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def fit_margin_rating(rows: list[dict]) -> tuple[dict[str, float], float]:
    """Weighted least-squares points rating from checkpoint-vs-checkpoint games.

    Rating gaps predict per-round score margins (my points minus the table
    mean), so opponent strength is corrected across the whole matchup graph.
    Ratings are centered on the population mean: under the fitted model,
    E[margin vs an opponent drawn uniformly from members] = R_n - mean(R),
    so a rating reads "expected points per round against a uniformly random
    checkpoint of the run" on the shared bank.

    A directed matchup seats one focal policy against a table of opponent
    clones, and the lone focal seat carries a systematic advantage of about
    +0.1 pts/round regardless of who sits in it (measured bidirectionally,
    e.g. iter 6968 focal vs 4523 was +0.20 while 4523 focal vs 6968 was
    +0.02). A fitted global intercept absorbs it; without one, chains of
    same-direction matchups telescope the seat bias into a fake rating trend.
    Returns (ratings, fitted focal bias).
    """

    model_rows = [
        row
        for row in rows
        if HEURISTIC_KEY not in (row["focal"], row["table"])
    ]
    if not model_rows:
        return {}, 0.0
    # Ratings are only identified within a connected component of the matchup
    # graph. Fresh checkpoints whose matchups so far are all against other
    # fresh checkpoints form an island whose absolute level is arbitrary (it
    # floats at the optimizer init), so rate only the component containing
    # the earliest checkpoint and leave islands unrated until a matchup
    # connects them.
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in model_rows:
        adjacency[row["focal"]].add(row["table"])
        adjacency[row["table"]].add(row["focal"])
    component = {min(adjacency)}
    frontier = list(component)
    while frontier:
        for neighbor in adjacency[frontier.pop()]:
            if neighbor not in component:
                component.add(neighbor)
                frontier.append(neighbor)
    model_rows = [row for row in model_rows if row["focal"] in component]
    members = sorted(component)
    if len(members) < 2:
        return {}, 0.0
    index = {key: position for position, key in enumerate(members)}
    focal_tensor = torch.tensor([index[row["focal"]] for row in model_rows])
    table_tensor = torch.tensor([index[row["table"]] for row in model_rows])
    margin_tensor = torch.tensor(
        [row["macro_relative_reward"] for row in model_rows],
        dtype=torch.float64,
    )
    weight_tensor = torch.tensor(
        [row["rounds"] for row in model_rows],
        dtype=torch.float64,
    )
    weight_tensor = weight_tensor / weight_tensor.sum()

    free = torch.zeros(len(members) - 1, dtype=torch.float64, requires_grad=True)
    pinned = torch.zeros(1, dtype=torch.float64)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([free, bias], lr=0.05)
    for _ in range(4000):
        optimizer.zero_grad()
        ratings = torch.cat([pinned, free])
        predicted = bias + ratings[focal_tensor] - ratings[table_tensor]
        loss = (weight_tensor * (margin_tensor - predicted) ** 2).sum()
        # While no pair has been played in both directions yet, the bias is
        # confounded with a uniform tilt of the ratings; the tiny ridge picks
        # the bias-zero solution there and is negligible once data decides.
        loss = loss + 1e-4 * bias.pow(2).sum()
        loss.backward()
        optimizer.step()
    final = torch.cat([pinned, free.detach()])
    # Only rating differences are identified; re-gauge so 0 = population mean
    # and each value is the expected margin vs a uniformly random checkpoint.
    final = final - final.mean()
    return (
        {key: float(final[position]) for key, position in index.items()},
        float(bias.detach()),
    )


def write_outputs(
    matchup_rows: list[dict],
    behavior_rows: list[dict],
    out_dir: Path,
    *,
    announce: bool = True,
) -> None:
    rating, focal_bias = fit_margin_rating(matchup_rows)
    behavior_by_key = {row["key"]: row for row in behavior_rows}
    rounds_by_member: dict[str, int] = {}
    opponents_by_member: dict[str, set[str]] = {}
    for row in matchup_rows:
        if HEURISTIC_KEY in (row["focal"], row["table"]):
            continue
        for key in (row["focal"], row["table"]):
            rounds_by_member[key] = rounds_by_member.get(key, 0) + row["rounds"]
        opponents_by_member.setdefault(row["focal"], set()).add(row["table"])
        opponents_by_member.setdefault(row["table"], set()).add(row["focal"])

    members = set(rating) | set(behavior_by_key)
    points = sorted((int(key.split("_")[1]), key) for key in members)
    if not points:
        return

    def fmt(value: float | None, spec: str = ".4f") -> str:
        return "" if value is None else format(value, spec)

    csv_path = out_dir / "elo.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w") as handle:
        handle.write(
            "iteration,margin_rating,avg_bid,bid_hit_rate,zero_bid_rate,"
            "all_bid_rate,matchup_rounds,opponents\n"
        )
        for iteration, key in points:
            behavior = behavior_by_key.get(key, {})
            handle.write(
                f"{iteration},{fmt(rating.get(key))},"
                f"{fmt(behavior.get('avg_bid'))},"
                f"{fmt(behavior.get('bid_hit_rate'))},"
                f"{fmt(behavior.get('zero_bid_rate'))},"
                f"{fmt(behavior.get('all_bid_rate'))},"
                f"{rounds_by_member.get(key, 0)},"
                f"{len(opponents_by_member.get(key, set()))}\n"
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
        percent_series = (
            ("bid_hit_rate", "tab:green", "bid hit %"),
            ("zero_bid_rate", "tab:orange", "0-bid %"),
            ("all_bid_rate", "tab:red", "bid-all %"),
        )
        handles = [line_bid]
        for field, color, label in percent_series:
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
    top.set_title(
        "Bid behavior and score-margin rating over training "
        f"({len(points)} checkpoints, {len(matchup_rows)} directed matchups, "
        f"{len(behavior_points)} behavior probes)"
    )
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
        bottom.annotate(
            f"latest: {rating_points[-1][1]:+.2f} pts/round vs avg ckpt",
            rating_points[-1],
            textcoords="offset points",
            xytext=(-8, 8),
            ha="right",
            fontsize=8,
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
    png_path = out_dir / "elo_ladder.png"
    png_tmp = png_path.with_suffix(".png.tmp")
    figure.savefig(png_tmp, dpi=150, format="png")
    plt.close(figure)
    png_tmp.replace(png_path)
    if announce:
        print(f"wrote {csv_path} and {png_path}")


def main() -> None:
    args = parse_args()
    if args.min_iter_gap < 1:
        raise SystemExit("--min-iter-gap must be at least 1")
    try:
        offsets = tuple(int(part) for part in args.offsets.split(","))
    except ValueError:
        raise SystemExit("--offsets must be comma-separated integers")
    if not offsets or any(offset < 1 for offset in offsets):
        raise SystemExit("--offsets must be positive rung distances")
    if args.publish_every < 1:
        raise SystemExit("--publish-every must be at least 1")
    archive_dir = REPO_DIR / "checkpoints" / args.run_name
    out_dir = REPO_DIR / "checkpoints" / "ladder" / args.run_name
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    local_paths = checkpoint_iterations(archive_dir)
    pulled_paths = checkpoint_iterations(ckpt_dir)
    volume_iters = [] if args.no_sync else volume_iterations(args.run_name)
    volume_set = set(volume_iters)

    all_iterations = sorted(set(local_paths) | set(pulled_paths) | set(volume_iters))
    if not all_iterations:
        raise SystemExit("No checkpoints found locally or on the volume.")
    ladder = select_ladder(all_iterations, args.min_iter_gap)
    print(f"ladder: {len(ladder)} checkpoints, iters {ladder[0]}-{ladder[-1]}")

    paths: dict[str, Path] = {}
    for iteration in ladder:
        key = f"iter_{iteration:05d}"
        if iteration in local_paths:
            paths[key] = local_paths[iteration]
        elif iteration in pulled_paths:
            paths[key] = pulled_paths[iteration]
        elif not args.no_sync:
            # Pull lazily when the first matchup needs this model. That lets
            # Elo points and the plot appear while the remote backlog syncs.
            paths[key] = ckpt_dir / f"plump_v4_iter_{iteration:05d}.pt"
        else:
            raise SystemExit(f"iter {iteration} not cached; rerun without --no-sync")

    def fetch_missing(key: str, dest: Path) -> None:
        iteration = int(key.split("_")[1])
        if iteration not in volume_set:
            raise FileNotFoundError(
                f"iter {iteration} is not cached and is absent from the volume listing"
            )
        pull_checkpoint(args.run_name, iteration, dest)

    cache_path = out_dir / "pairings.jsonl"
    fingerprint = bank_fingerprint(args)
    matching_rows = [
        row
        for row in load_cache_rows(cache_path)
        if all(row.get(key) == value for key, value in fingerprint.items())
    ]
    # Directed identities matter: A as the lone focal policy against a table
    # of B policies is not equivalent to B as focal against a table of A.
    rows_by_pair = {
        (row["focal"], row["table"]): row
        for row in matching_rows
    }
    rows = list(rows_by_pair.values())
    done = set(rows_by_pair)

    behavior_path = out_dir / "behavior.jsonl"
    behavior_print = behavior_fingerprint(args)
    behavior_rows = [
        row
        for row in load_cache_rows(behavior_path)
        if all(row.get(key) == value for key, value in behavior_print.items())
    ]
    behavior_done = {row["key"] for row in behavior_rows}
    ladder_keys = [f"iter_{iteration:05d}" for iteration in ladder]
    probe_todo = [
        ladder_keys[index]
        for index in _spread_order(len(ladder_keys))
        if ladder_keys[index] not in behavior_done
    ]
    if args.max_pairings == 0:
        probe_todo = []

    pairings = build_pairings(
        ladder,
        offsets,
        bidirectional=args.bidirectional,
        heuristic_anchors=args.heuristic_anchors,
    )
    todo = [pair for pair in pairings if pair not in done]
    if args.max_pairings is not None:
        todo = todo[: args.max_pairings]
    print(
        f"pairings: {len(pairings)} scheduled, {len(done)} cached, "
        f"{len(todo)} to play; behavior probes: {len(probe_todo)} to run"
    )

    policies = None
    if probe_todo or todo:
        policies = PolicyCache(
            paths,
            args.device,
            fetch_missing=None if args.no_sync else fetch_missing,
        )

    if probe_todo:
        probe_bank = DealBank.generate(
            player_counts=(3, 4, 5),
            hand_sizes=tuple(range(3, 11)),
            deals_per_configuration=args.probe_deals,
            seed=args.bank_seed + 1,
        )
        with behavior_path.open("a") as handle:
            for position, key in enumerate(probe_todo, start=1):
                row = probe_behavior(key, policies, probe_bank, args)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                behavior_rows.append(row)
                write_outputs(rows, behavior_rows, out_dir, announce=False)
                print(
                    f"[probe {position}/{len(probe_todo)}] {key}: "
                    f"avg_bid {row['avg_bid']:.2f} "
                    f"hit {100 * row['bid_hit_rate']:.1f}% "
                    f"zero {100 * row['zero_bid_rate']:.1f}% "
                    f"all {100 * row['all_bid_rate']:.1f}% "
                    f"({row['elapsed_sec']}s)",
                    flush=True,
                )

    if todo:
        bank = DealBank.generate(
            player_counts=(3, 4, 5),
            hand_sizes=tuple(range(3, 11)),
            deals_per_configuration=args.deals_per_configuration,
            seed=args.bank_seed,
        )
        rounds_per_pairing = sum(
            deal.spec.num_players * deal.spec.num_players
            for deal in bank.deals
        )
        print(
            f"evaluation: {rounds_per_pairing} rounds/pairing, "
            f"{rounds_per_pairing * len(todo):,} new rounds total",
            flush=True,
        )
        with cache_path.open("a") as handle:
            for position, (focal, table) in enumerate(todo, start=1):
                row = play_pairing(focal, table, policies, bank, args)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)
                if position % args.publish_every == 0 or position == len(todo):
                    write_outputs(rows, behavior_rows, out_dir, announce=False)
                print(
                    f"[{position}/{len(todo)}] {focal} vs {table}: "
                    f"reward {row['macro_relative_reward']:+.3f} "
                    f"w/d/l {row['wins']}/{row['draws']}/{row['losses']} "
                    f"({row['elapsed_sec']}s)",
                    flush=True,
                )

    if rows or behavior_rows:
        if not todo and not probe_todo:
            write_outputs(rows, behavior_rows, out_dir)
    else:
        print("no results yet; nothing to fit")


if __name__ == "__main__":
    main()
