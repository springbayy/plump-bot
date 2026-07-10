"""Plot reward, search quality, and prediction metrics for a v5 run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()
    rows = _read(args.run_dir / "metrics.csv")
    if not rows:
        raise SystemExit("No v5 metric rows found.")
    output = args.output or args.run_dir / "metrics.png"
    x = _series(rows, "cycle")
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(22, 10),
        constrained_layout=True,
    )
    figure.suptitle(
        f"Plump v5 Expert Iteration: {args.run_dir.name}",
        fontsize=16,
    )
    _plot(
        axes[0, 0],
        rows,
        x,
        [
            "rollout_heuristic_relative_reward",
            "rollout_self_relative_reward",
            "rollout_mixed_relative_reward",
        ],
        "Training Relative Reward",
        args.smooth,
        zero=True,
    )
    _plot(
        axes[0, 1],
        rows,
        x,
        [
            "eval_raw_macro_relative_reward",
            "eval_teacher_macro_relative_reward",
        ],
        "Controlled Evaluation",
        1,
        zero=True,
    )
    _plot(
        axes[0, 2],
        rows,
        x,
        [
            "search_accepted_rate",
            "search_bid_accepted_rate",
            "search_play_accepted_rate",
            "search_split_half_agreement",
        ],
        "Search Label Quality",
        args.smooth,
        limits=(0.0, 1.0),
    )
    _plot(
        axes[0, 3],
        rows,
        x,
        ["loss_policy_ce", "loss_q", "loss_value", "loss_owner"],
        "Training Losses",
        args.smooth,
    )
    _plot(
        axes[1, 0],
        rows,
        x,
        [
            "pred_q_explained_variance",
            "pred_q_rank_correlation",
            "pred_q_mae",
        ],
        "Q Calibration",
        args.smooth,
    )
    _plot(
        axes[1, 1],
        rows,
        x,
        [
            "pred_value_explained_variance",
            "pred_bid_value_explained_variance",
            "pred_play_value_explained_variance",
        ],
        "Value Explained Variance",
        args.smooth,
        limits=(-0.25, 1.0),
    )
    _plot(
        axes[1, 2],
        rows,
        x,
        [
            "pred_owner_brier",
            "pred_owner_uniform_brier",
            "owner_belief_weight",
        ],
        "Belief Calibration",
        args.smooth,
    )
    _plot(
        axes[1, 3],
        rows,
        x,
        [
            "search_mean_depth",
            "search_mean_determinizations",
            "search_leaf_rollout_fraction",
        ],
        "Search Schedule",
        args.smooth,
    )
    for axis in axes.flat:
        axis.set_xlabel("cycle")
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    latest = rows[-1]
    print(f"wrote {output}")
    print(
        f"cycle={int(float(latest['cycle']))} "
        f"accepted={_format(latest, 'search_accepted_rate')} "
        f"policy_ce={_format(latest, 'loss_policy_ce')} "
        f"q_loss={_format(latest, 'loss_q')} "
        f"value_ev={_format(latest, 'pred_value_explained_variance')}"
    )


def _read(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return [
            {
                key: _parse(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]


def _parse(value: str) -> float | str:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return value


def _series(
    rows: list[dict[str, float | str]],
    name: str,
) -> list[float]:
    return [
        float(row.get(name, math.nan))
        if isinstance(row.get(name), float)
        else math.nan
        for row in rows
    ]


def _smooth(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        chunk = [
            value
            for value in values[max(0, index - window + 1) : index + 1]
            if not math.isnan(value)
        ]
        result.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return result


def _plot(
    axis,
    rows,
    x,
    fields,
    title,
    smooth,
    *,
    limits=None,
    zero=False,
) -> None:
    plotted = False
    for field in fields:
        values = _series(rows, field)
        if not any(not math.isnan(value) for value in values):
            continue
        axis.plot(
            x,
            _smooth(values, max(smooth, 1)),
            label=field.replace("_", " "),
        )
        plotted = True
    if zero:
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    if limits is not None:
        axis.set_ylim(*limits)
    axis.set_title(title)
    if plotted:
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "no data yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )


def _format(row, field) -> str:
    value = row.get(field, math.nan)
    return (
        f"{value:.4f}"
        if isinstance(value, float) and not math.isnan(value)
        else "n/a"
    )


if __name__ == "__main__":
    main()
