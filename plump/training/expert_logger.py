"""Structured metrics for schema-v5 expert iteration."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from plump.evaluation import EvaluationReport
from plump.rounds import rules_fingerprint

from .expert_iteration import (
    CHECKPOINT_SCHEMA_VERSION,
    ExpertCycle,
    ExpertDiagnostics,
    ExpertUpdateStats,
)


METRIC_FIELDS = [
    "cycle",
    "timestamp_utc",
    "schema_version",
    "rules_fingerprint",
    "rollout_rounds",
    "new_states",
    "replay_states",
    "rollout_bid_hit_rate",
    "rollout_bid_abs_error",
    "rollout_heuristic_relative_reward",
    "rollout_self_relative_reward",
    "rollout_mixed_relative_reward",
    "rollout_historical_relative_reward",
    "updates",
    "update_samples",
    "loss_total",
    "loss_policy_ce",
    "loss_q",
    "loss_value",
    "loss_trick",
    "loss_owner",
    "loss_owner_ce",
    "loss_owner_capacity",
    "loss_entropy_floor",
    "grad_norm",
    "search_accepted_rate",
    "search_bid_accepted_rate",
    "search_play_accepted_rate",
    "search_split_half_agreement",
    "search_median_js",
    "search_mean_nodes",
    "search_mean_depth",
    "search_mean_determinizations",
    "search_leaf_rollout_fraction",
    "teacher_student_kl",
    "pred_q_mae",
    "pred_q_explained_variance",
    "pred_q_rank_correlation",
    "pred_value_mae",
    "pred_value_explained_variance",
    "pred_bid_value_explained_variance",
    "pred_play_value_explained_variance",
    "pred_owner_brier",
    "pred_owner_uniform_brier",
    "owner_belief_weight",
    "bid_leaf_value_weight",
    "play_leaf_value_weight",
    "bid_search_temperature",
    "play_search_temperature",
    "sampler_infeasible_rejection_rate",
    "sampler_failed_draw_rate",
    "eval_raw_macro_relative_reward",
    "eval_raw_ci_low",
    "eval_raw_ci_high",
    "eval_raw_bid_hit_rate",
    "eval_teacher_macro_relative_reward",
    "eval_teacher_ci_low",
    "eval_teacher_ci_high",
    "eval_teacher_bid_hit_rate",
    "checkpoint_path",
]


class ExpertRunLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.csv"
        self.events_path = self.log_dir / "events.jsonl"
        self.config_path = self.log_dir / "run_config.json"
        self.latest_path = self.log_dir / "latest.json"
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="") as file:
                csv.DictWriter(
                    file,
                    fieldnames=METRIC_FIELDS,
                ).writeheader()

    def write_config(self, payload: dict) -> None:
        data = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "observation_schema_version": 4,
            "training_algorithm": "information_set_expert_iteration",
            "rules_fingerprint": rules_fingerprint(),
            **payload,
        }
        self.config_path.write_text(
            json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
        )

    def log_cycle(
        self,
        *,
        cycle_index: int,
        cycle: ExpertCycle,
        replay_states: int,
        update: ExpertUpdateStats,
        diagnostics: ExpertDiagnostics,
        raw_evaluation: EvaluationReport | None,
        teacher_evaluation: EvaluationReport | None,
        checkpoint_path: Path | None,
    ) -> None:
        row = self._row(
            cycle_index,
            cycle,
            replay_states,
            update,
            diagnostics,
            raw_evaluation,
            teacher_evaluation,
            checkpoint_path,
        )
        with self.metrics_path.open("a", newline="") as file:
            csv.DictWriter(
                file,
                fieldnames=METRIC_FIELDS,
            ).writerow(row)
        event = {
            "type": "cycle",
            "cycle": cycle_index,
            "timestamp_utc": row["timestamp_utc"],
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            "rollout": {
                "rounds": len(cycle.outcomes),
                "states": len(cycle.samples),
                "outcomes": [asdict(item) for item in cycle.outcomes],
            },
            "update": asdict(update),
            "diagnostics": asdict(diagnostics),
            "raw_evaluation": (
                asdict(raw_evaluation)
                if raw_evaluation is not None
                else None
            ),
            "teacher_evaluation": (
                asdict(teacher_evaluation)
                if teacher_evaluation is not None
                else None
            ),
            "checkpoint_path": (
                str(checkpoint_path) if checkpoint_path else ""
            ),
        }
        serialized = json.dumps(event, sort_keys=True, default=str)
        with self.events_path.open("a") as file:
            file.write(serialized + "\n")
        self.latest_path.write_text(
            json.dumps(event, indent=2, sort_keys=True, default=str) + "\n"
        )

    def _row(
        self,
        cycle_index: int,
        cycle: ExpertCycle,
        replay_states: int,
        update: ExpertUpdateStats,
        diagnostics: ExpertDiagnostics,
        raw_evaluation: EvaluationReport | None,
        teacher_evaluation: EvaluationReport | None,
        checkpoint_path: Path | None,
    ) -> dict:
        by_arm = {
            arm: [
                outcome.focal_reward
                for outcome in cycle.outcomes
                if outcome.opponent_arm == arm
            ]
            for arm in ("self", "heuristic", "mixed", "historical")
        }
        row = {
            "cycle": cycle_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            "rollout_rounds": len(cycle.outcomes),
            "new_states": len(cycle.samples),
            "replay_states": replay_states,
            "rollout_bid_hit_rate": _mean(
                outcome.focal_bid_hit for outcome in cycle.outcomes
            ),
            "rollout_bid_abs_error": _mean(
                outcome.focal_bid_error for outcome in cycle.outcomes
            ),
            "rollout_heuristic_relative_reward": _mean(
                by_arm["heuristic"]
            ),
            "rollout_self_relative_reward": _mean(by_arm["self"]),
            "rollout_mixed_relative_reward": _mean(by_arm["mixed"]),
            "rollout_historical_relative_reward": _mean(
                by_arm["historical"]
            ),
            "updates": update.updates,
            "update_samples": update.samples,
            "loss_total": update.total_loss,
            "loss_policy_ce": update.policy_loss,
            "loss_q": update.q_loss,
            "loss_value": update.value_loss,
            "loss_trick": update.trick_loss,
            "loss_owner": update.owner_loss,
            "loss_owner_ce": update.owner_ce_loss,
            "loss_owner_capacity": update.owner_capacity_loss,
            "loss_entropy_floor": update.entropy_floor_loss,
            "grad_norm": update.grad_norm,
            "search_accepted_rate": diagnostics.accepted_rate,
            "search_bid_accepted_rate": diagnostics.bid_accepted_rate,
            "search_play_accepted_rate": diagnostics.play_accepted_rate,
            "search_split_half_agreement": (
                diagnostics.split_half_agreement
            ),
            "search_median_js": diagnostics.median_js,
            "search_mean_nodes": diagnostics.mean_nodes,
            "search_mean_depth": diagnostics.mean_depth,
            "search_mean_determinizations": (
                diagnostics.mean_determinizations
            ),
            "search_leaf_rollout_fraction": (
                diagnostics.leaf_rollout_fraction
            ),
            "teacher_student_kl": diagnostics.teacher_student_kl,
            "pred_q_mae": diagnostics.q_mae,
            "pred_q_explained_variance": (
                diagnostics.q_explained_variance
            ),
            "pred_q_rank_correlation": diagnostics.q_rank_correlation,
            "pred_value_mae": diagnostics.value_mae,
            "pred_value_explained_variance": (
                diagnostics.value_explained_variance
            ),
            "pred_bid_value_explained_variance": (
                diagnostics.bid_value_explained_variance
            ),
            "pred_play_value_explained_variance": (
                diagnostics.play_value_explained_variance
            ),
            "pred_owner_brier": diagnostics.owner_brier,
            "pred_owner_uniform_brier": diagnostics.owner_uniform_brier,
            "owner_belief_weight": diagnostics.owner_belief_weight,
            "bid_leaf_value_weight": diagnostics.bid_leaf_value_weight,
            "play_leaf_value_weight": diagnostics.play_leaf_value_weight,
            "bid_search_temperature": (
                diagnostics.bid_search_temperature
            ),
            "play_search_temperature": (
                diagnostics.play_search_temperature
            ),
            "sampler_infeasible_rejection_rate": (
                diagnostics.sampler_infeasible_rejection_rate
            ),
            "sampler_failed_draw_rate": (
                diagnostics.sampler_failed_draw_rate
            ),
            "checkpoint_path": (
                str(checkpoint_path) if checkpoint_path else ""
            ),
        }
        if raw_evaluation is not None:
            row.update(
                {
                    "eval_raw_macro_relative_reward": (
                        raw_evaluation.macro_relative_reward
                    ),
                    "eval_raw_ci_low": (
                        raw_evaluation.relative_reward_ci_low
                    ),
                    "eval_raw_ci_high": (
                        raw_evaluation.relative_reward_ci_high
                    ),
                    "eval_raw_bid_hit_rate": (
                        raw_evaluation.macro_bid_hit_rate
                    ),
                }
            )
        if teacher_evaluation is not None:
            row.update(
                {
                    "eval_teacher_macro_relative_reward": (
                        teacher_evaluation.macro_relative_reward
                    ),
                    "eval_teacher_ci_low": (
                        teacher_evaluation.relative_reward_ci_low
                    ),
                    "eval_teacher_ci_high": (
                        teacher_evaluation.relative_reward_ci_high
                    ),
                    "eval_teacher_bid_hit_rate": (
                        teacher_evaluation.macro_bid_hit_rate
                    ),
                }
            )
        return {
            field: row.get(field, "")
            for field in METRIC_FIELDS
        }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
