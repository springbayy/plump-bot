"""Plot key metrics for one Plump PPO training run."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a metrics dashboard for a Plump training run.")
    parser.add_argument("run_dir", type=Path, help="Training run directory containing metrics.csv.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to RUN_DIR/metrics.png.")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window for noisy scalar lines.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.csv found at {metrics_path}")

    rows = _read_metrics(metrics_path)
    if not rows:
        raise SystemExit(f"{metrics_path} has no metric rows yet.")

    output_path = args.output or (args.run_dir / "metrics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    fig.suptitle(f"Plump PPO Training: {args.run_dir.name}", fontsize=16)

    x = _series(rows, "iteration")
    smooth = max(args.smooth, 1)

    # Row 1: direct progress — is the policy actually getting better?
    _plot_evaluation(axes[0][0], rows, x)
    _plot_training_reward(axes[0][1], rows, x, smooth)
    _plot_bid_success(axes[0][2], rows, x, smooth)
    _plot_explained_variance(axes[0][3], rows, x, smooth)

    # Row 2: indirect progress — is the model still learning, and healthily?
    _plot_prediction_heads(axes[1][0], rows, x, smooth)
    _plot_policy_sharpness(axes[1][1], rows, x, smooth)
    _plot_ppo_stability(axes[1][2], rows, x, smooth)
    _plot_throughput(axes[1][3], rows, x, smooth)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("iteration")

    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    last_iteration = int(x[-1]) if x else 0
    print(f"wrote {output_path}")
    print(f"rows={len(rows)} last_iteration={last_iteration}")
    _print_latest_metrics(rows)


def _read_metrics(metrics_path: Path) -> list[dict[str, float | str]]:
    with metrics_path.open(newline="") as file:
        reader = csv.DictReader(file)
        parsed = [{key: _parse_value(value) for key, value in row.items()} for row in reader]

    # A resumed run can append lower iteration numbers after unsaved rows from
    # an abandoned trajectory. Roll back those later rows before adding the
    # resumed branch.
    by_iteration: dict[int, dict[str, float | str]] = {}
    previous_iteration = -1
    for row in parsed:
        iteration = row.get("iteration", math.nan)
        if isinstance(iteration, float) and not math.isnan(iteration):
            current_iteration = int(iteration)
            if current_iteration <= previous_iteration:
                by_iteration = {
                    saved_iteration: saved_row
                    for saved_iteration, saved_row in by_iteration.items()
                    if saved_iteration < current_iteration
                }
            by_iteration[current_iteration] = row
            previous_iteration = current_iteration
    return [by_iteration[iteration] for iteration in sorted(by_iteration)]


def _parse_value(value: str) -> float | str:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return value


def _series(rows: list[dict[str, float | str]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(name, math.nan)
        values.append(float(value) if isinstance(value, float) else math.nan)
    return values


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = [value for value in values[start : index + 1] if not math.isnan(value)]
        result.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return result


def _has_data(values: list[float]) -> bool:
    return any(not math.isnan(value) for value in values)


def _plot_bid_success(ax, rows: list[dict[str, float | str]], x: list[float], smooth: int) -> None:
    plotted = False
    rollout_hit = _series(rows, "rollout_bid_hit_rate")
    if _has_data(rollout_hit):
        ax.plot(x, _smooth(rollout_hit, smooth), label="rollout bid hit rate", linewidth=2)
        plotted = True

    all_player_hit = _series(rows, "rollout_all_player_bid_hit_rate")
    if _has_data(all_player_hit):
        ax.plot(
            x,
            _smooth(all_player_hit, smooth),
            label="all-player bid hit rate",
            linewidth=1.5,
            alpha=0.7,
        )
        plotted = True

    eval_hit = _series(rows, "eval_macro_bid_hit_rate")
    if _has_data(eval_hit):
        ax.plot(x, eval_hit, label="eval bid hit rate", marker="o", markersize=4, linewidth=2)
        plotted = True

    abs_error = _series(rows, "rollout_bid_abs_error_mean")
    if _has_data(abs_error):
        smoothed_error = _smooth(abs_error, smooth)
        error_limit = _robust_axis_max(smoothed_error)
        twin = ax.twinx()
        twin.plot(
            x,
            _clip_to_axis(smoothed_error, error_limit),
            label="avg bid error",
            color="tab:red",
            alpha=0.7,
        )
        _mark_clipped_points(
            twin,
            x,
            smoothed_error,
            error_limit,
            color="tab:red",
        )
        twin.set_ylabel("average absolute bid error")
        twin.set_ylim(0.0, error_limit)
        twin.legend(fontsize=8, loc="lower right")

    ax.set_ylim(0.0, 1.0)
    ax.set_title("Bid Success Rate")
    ax.set_ylabel("players hitting bid")
    if plotted:
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no bid-hit data yet", ha="center", va="center", transform=ax.transAxes)


def _plot_evaluation(ax, rows: list[dict[str, float | str]], x: list[float]) -> None:
    reward = _series(rows, "eval_macro_relative_reward")
    low = _series(rows, "eval_ci_low")
    high = _series(rows, "eval_ci_high")
    if not _has_data(reward):
        ax.set_title("Controlled Evaluation")
        ax.text(0.5, 0.5, "no evaluation yet", ha="center", va="center", transform=ax.transAxes)
        return
    ax.plot(x, reward, label="macro relative reward", marker="o", markersize=4)
    finite = [
        (x_value, lo, hi)
        for x_value, lo, hi in zip(x, low, high)
        if not any(math.isnan(value) for value in (x_value, lo, hi))
    ]
    if finite:
        fx, flo, fhi = zip(*finite)
        ax.fill_between(fx, flo, fhi, alpha=0.2, label="95% bootstrap CI")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Controlled Evaluation")
    ax.legend(fontsize=8)


def _plot_training_reward(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    reward = _series(rows, "rollout_heuristic_relative_reward")
    label = "training rounds vs heuristic"
    title = "Training Reward vs Heuristic"
    sparse_evaluation = False
    if not _has_data(reward):
        reward = _series(rows, "eval_macro_relative_reward")
        label = "controlled eval vs heuristic"
        title = "Reward vs Heuristic"
        sparse_evaluation = True
    if not _has_data(reward):
        ax.set_title(title)
        ax.text(0.5, 0.5, "no reward data yet", ha="center", va="center", transform=ax.transAxes)
        return
    if sparse_evaluation:
        finite = [
            (iteration, value)
            for iteration, value in zip(x, reward)
            if not math.isnan(value)
        ]
        plot_x, plot_reward = zip(*finite)
        ax.plot(
            plot_x,
            plot_reward,
            label=label,
            linewidth=2,
            marker="o",
            markersize=5,
        )
    else:
        ax.plot(
            x,
            _smooth(reward, smooth),
            label=label,
            linewidth=2,
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("mean focal-player relative score")
    ax.legend(fontsize=8)


def _plot_ppo_stability(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    kl = _smooth(_series(rows, "approx_kl"), smooth)
    clip = _smooth(_series(rows, "clip_fraction"), smooth)
    plotted = False

    if _has_data(kl):
        kl_limit = _robust_axis_max(kl)
        ax.plot(x, _clip_to_axis(kl, kl_limit), label="approx KL", color="tab:blue")
        _mark_clipped_points(ax, x, kl, kl_limit, color="tab:blue")
        ax.set_ylabel("approx KL", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_ylim(0.0, kl_limit)
        plotted = True

    if _has_data(clip):
        clip_limit = _robust_axis_max(clip)
        twin = ax.twinx()
        twin.plot(
            x,
            _clip_to_axis(clip, clip_limit),
            label="clip fraction",
            color="tab:orange",
        )
        _mark_clipped_points(
            twin,
            x,
            clip,
            clip_limit,
            color="tab:orange",
        )
        twin.set_ylabel("clip fraction", color="tab:orange")
        twin.tick_params(axis="y", labelcolor="tab:orange")
        twin.set_ylim(0.0, clip_limit)
        plotted = True

    ax.set_title("PPO Stability")
    if not plotted:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)


def _robust_axis_max(values: list[float]) -> float:
    finite = sorted(value for value in values if not math.isnan(value))
    if not finite:
        return 1e-6
    percentile_index = min(math.ceil(0.95 * len(finite)) - 1, len(finite) - 1)
    return max(finite[percentile_index] * 4.0 / 3.0, 1e-6)


def _clip_to_axis(values: list[float], axis_max: float) -> list[float]:
    return [
        min(value, axis_max)
        if not math.isnan(value)
        else math.nan
        for value in values
    ]


def _mark_clipped_points(
    ax,
    x: list[float],
    values: list[float],
    axis_max: float,
    *,
    color: str,
) -> None:
    clipped_x = [
        iteration
        for iteration, value in zip(x, values)
        if not math.isnan(value) and value > axis_max
    ]
    if clipped_x:
        ax.scatter(
            clipped_x,
            [axis_max * 0.985] * len(clipped_x),
            marker="^",
            s=18,
            color=color,
            zorder=4,
        )


def _plot_explained_variance(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """How much of the return the model explains — should climb, then hold."""

    series = (
        ("pred_value_explained_variance", "value EV", "tab:green", "-"),
        ("pred_bid_value_explained_variance", "bid EV", "tab:blue", "-"),
        ("pred_play_value_explained_variance", "play EV", "tab:orange", "-"),
        (
            "pred_trick_implied_value_explained_variance",
            "trick-implied EV",
            "tab:purple",
            "--",
        ),
    )
    plotted = False
    for name, label, color, style in series:
        values = _series(rows, name)
        if not _has_data(values):
            continue
        ax.plot(
            x,
            _smooth(values, smooth),
            label=label,
            color=color,
            linestyle=style,
            alpha=0.8,
        )
        plotted = True
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_ylim(-0.25, 1.0)
    ax.set_title("Value Explained Variance")
    if plotted:
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)


def _plot_prediction_heads(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """Belief-head quality: accuracy/probability up, Brier scores down."""

    series = (
        ("pred_trick_count_accuracy", "trick-count accuracy", "tab:blue", "-"),
        ("pred_trick_count_true_prob", "trick-count true prob", "tab:cyan", "-"),
        ("pred_hit_prob_brier", "hit-prob Brier (down)", "tab:red", "--"),
        (
            "pred_owner_opponent_accuracy",
            "owner opp. accuracy",
            "tab:green",
            "-",
        ),
        (
            "pred_owner_opponent_true_prob",
            "owner opp. true prob",
            "tab:olive",
            "-",
        ),
        ("pred_owner_brier", "owner Brier (down)", "tab:brown", "--"),
    )
    plotted = False
    for name, label, color, style in series:
        values = _series(rows, name)
        if not _has_data(values):
            continue
        ax.plot(
            x,
            _smooth(values, smooth),
            label=label,
            color=color,
            linestyle=style,
            alpha=0.8,
        )
        plotted = True
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Belief Heads (tricks + hidden cards)")
    if plotted:
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    else:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)


def _plot_policy_sharpness(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """Entropy should decay as reward rises; collapse with flat reward is bad."""

    plotted = False
    for name, label, color in (
        ("pred_bid_entropy", "bid entropy", "tab:blue"),
        ("pred_play_entropy", "play entropy", "tab:orange"),
    ):
        values = _series(rows, name)
        if not _has_data(values):
            continue
        ax.plot(x, _smooth(values, smooth), label=label, color=color)
        plotted = True
    ax.set_ylabel("entropy (nats)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Policy Sharpness")
    if plotted:
        ax.legend(fontsize=8, loc="upper left")

    twin = None
    for name, label, color in (
        ("pred_bid_max_prob", "bid max prob", "tab:cyan"),
        ("pred_play_max_prob", "play max prob", "tab:red"),
    ):
        values = _series(rows, name)
        if not _has_data(values):
            continue
        if twin is None:
            twin = ax.twinx()
            twin.set_ylim(0.0, 1.0)
            twin.set_ylabel("max action probability")
        twin.plot(
            x,
            _smooth(values, smooth),
            label=label,
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        plotted = True
    if twin is not None:
        twin.legend(fontsize=8, loc="lower right")
    if not plotted:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)


def _plot_throughput(
    ax,
    rows: list[dict[str, float | str]],
    x: list[float],
    smooth: int,
) -> None:
    """Progress per wall-clock: decisions/second."""

    samples = _series(rows, "samples")
    iteration_seconds = _series(rows, "iteration_sec")
    rate = [
        sample_count / seconds
        if not math.isnan(sample_count)
        and not math.isnan(seconds)
        and seconds > 0
        else math.nan
        for sample_count, seconds in zip(samples, iteration_seconds)
    ]
    plotted = False
    if _has_data(rate):
        ax.plot(
            x,
            _smooth(rate, smooth),
            label="decisions / second",
            color="tab:blue",
        )
        ax.set_ylabel("decisions / second")
        ax.set_ylim(bottom=0.0)
        plotted = True

    ax.set_title("Throughput")
    if not plotted:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)


def _print_latest_metrics(rows: list[dict[str, float | str]]) -> None:
    latest = rows[-1]
    print(
        "latest "
        f"bid_hit={_format(latest, 'rollout_bid_hit_rate')} "
        f"bid_error={_format(latest, 'rollout_bid_abs_error_mean')} "
        f"policy_loss={_format(latest, 'loss_policy')} "
        f"value_loss={_format(latest, 'loss_value')} "
        f"kl={_format(latest, 'approx_kl')} "
        f"clip={_format(latest, 'clip_fraction')}"
    )

    prediction = _latest_row_with(rows, "pred_samples")
    if prediction is not None:
        print(
            "prediction "
            f"iteration={int(float(prediction['iteration']))} "
            f"value_explained={_format(prediction, 'pred_value_explained_variance')} "
            f"trick_implied_explained={_format(prediction, 'pred_trick_implied_value_explained_variance')} "
            f"trick_accuracy={_format(prediction, 'pred_trick_count_accuracy')} "
            f"owner_accuracy={_format(prediction, 'pred_owner_accuracy')}"
            f" owner_opponent_accuracy={_format(prediction, 'pred_owner_opponent_accuracy')}"
            f" owner_capacity_mae={_format(prediction, 'pred_owner_capacity_mae')}"
            f" owner_raw_capacity_mae={_format(prediction, 'pred_owner_raw_capacity_mae')}"
        )

    evaluation = _latest_row_with(rows, "eval_macro_relative_reward")
    if evaluation is not None:
        print(
            "evaluation "
            f"iteration={int(float(evaluation['iteration']))} "
            f"relative_reward={_format(evaluation, 'eval_macro_relative_reward')} "
            f"ci=[{_format(evaluation, 'eval_ci_low')},{_format(evaluation, 'eval_ci_high')}] "
            f"bid_hit={_format(evaluation, 'eval_macro_bid_hit_rate')} "
            f"elo_delta={_format(evaluation, 'eval_elo_delta')}"
        )

    search = _latest_row_with(rows, "search_bid_probed")
    if search is not None:
        print(
            "search "
            f"iteration={int(float(search['iteration']))} "
            f"bid_gate={_format(search, 'search_bid_gate_passed')} "
            f"bid_accept={_format(search, 'search_bid_accepted_rate')} "
            f"play_gate={_format(search, 'search_play_gate_passed')} "
            f"play_accept={_format(search, 'search_play_accepted_rate')}"
        )


def _latest_row_with(
    rows: list[dict[str, float | str]],
    name: str,
) -> dict[str, float | str] | None:
    for row in reversed(rows):
        value = row.get(name, math.nan)
        if isinstance(value, float) and not math.isnan(value):
            return row
    return None


def _format(row: dict[str, float | str], name: str) -> str:
    value = row.get(name, math.nan)
    if not isinstance(value, float) or math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
