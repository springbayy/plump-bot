"""Balanced schema-v4 PPO for round-local and full-game Plump policies."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from plump.cards import Card, make_deck
from plump.env import PlumpEnv
from plump.modeling import (
    EncodedObservation,
    ModelConfig,
    SCHEMA_VERSION,
    card_from_id,
    card_id,
    encode_observation,
)
from plump.modeling.encoding import NUM_CARDS
from plump.modeling.torch_model import (
    PlumpTransformerModel,
    best_torch_device,
    combined_action_logits,
    encoded_observations_to_batch,
    index_model_batch,
    load_v2_weights,
    load_v3_weights,
    model_autocast,
)
from plump.policies import ActionPolicy, HeuristicPolicy, ModelPolicy
from plump.rounds import (
    RoundSpec,
    descending_ascending_schedule,
    round_game_config,
    rules_fingerprint,
)
from plump.state import (
    Bid,
    BidAction,
    GameConfig,
    Observation,
    Phase,
    PlayCardAction,
    RoundState,
    Trick,
)
from .common import (
    PositionBaseline,
    allocate_opponent_arms,
    compute_relative_rewards,
    final_bids_relative as _final_bids_relative,
    final_tricks_relative as _final_tricks_relative,
    owner_targets_relative as _owner_targets_relative,
)


SamplePhase = Literal["bid", "play"]
OpponentArm = Literal["self", "heuristic", "mixed", "historical"]
TrainingMode = Literal["round", "game"]
PositionKey = tuple[int, int, int]


@dataclass
class BatchSettings:
    """Unpickle-only schema-v1 checkpoint payload shim."""

    num_players: int = 0
    hand_size: int = 0


@dataclass
class EvaluationStats:
    """Unpickle-only schema-v1 checkpoint payload shim."""

    rounds: int = 0


@dataclass
class TrainingConfig:
    """Hyperparameters for the balanced round-local deployment objective."""

    player_counts: tuple[int, ...] = (3, 4, 5)
    hand_sizes: tuple[int, ...] = tuple(range(3, 11))
    rounds_per_configuration: int = 8
    games_per_player_seat: int = 4
    num_envs: int = 32
    ppo_epochs: int = 4
    minibatch_size: int = 256
    microbatch_size: int | None = None
    learning_rate: float = 3e-4
    ppo_clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    trick_coef: float = 0.1
    owner_coef: float = 0.05
    owner_capacity_coef: float = 0.1
    max_grad_norm: float = 1.0
    position_baseline_decay: float = 0.98
    self_play_fraction: float = 0.3
    heuristic_fraction: float = 0.3
    mixed_fraction: float = 0.3
    historical_fraction: float = 0.1
    historical_checkpoint_paths: tuple[str, ...] = ()
    historical_max_snapshots: int = 4
    include_game_context: bool = False
    trick_baseline: bool = False
    training_mode: TrainingMode = "round"
    game_schedule: tuple[int, ...] = ()
    min_cards: int = 3
    max_cards: int = 10
    gae_gamma: float = 1.0
    round_gae_lambda: float = 0.95
    game_gae_lambda: float = 0.97
    precision: str = "fp32"
    seed: int = 1
    device: str | None = None
    model_config: ModelConfig = field(default_factory=ModelConfig)

    @property
    def specs(self) -> tuple[RoundSpec, ...]:
        return tuple(RoundSpec(players, hand) for players in self.player_counts for hand in self.hand_sizes)

    @property
    def rounds_per_batch(self) -> int:
        return len(self.specs) * self.rounds_per_configuration

    @property
    def resolved_game_schedule(self) -> tuple[int, ...]:
        return self.game_schedule or tuple(
            descending_ascending_schedule(
                min_cards=self.min_cards,
                max_cards=self.max_cards,
            )
        )


@dataclass
class RolloutSample:
    encoded: EncodedObservation
    phase: SamplePhase
    action_index: int
    old_logprob: float
    old_value: float
    old_residual_value: float
    old_entropy: float
    position_intercept: float
    acting_player: int
    episode_id: int
    round_id: int
    spec: RoundSpec
    bidding_position: int
    opponent_arm: OpponentArm
    return_target: float | None = None
    value_target: float | None = None
    final_trick_targets: list[int] | None = None
    final_bid_targets: list[int] | None = None
    owner_targets: list[int] | None = None
    round_weight: float = 0.0
    advantage_target: float | None = None
    observation: object | None = None
    ppo_policy_enabled: bool = True
    search_target_probabilities: list[float] | None = None
    trick_position: int = -1


@dataclass
class RoundOutcome:
    episode_id: int
    spec: RoundSpec
    opponent_arm: OpponentArm
    focal_reward: float | None
    bid_hit_count: int
    bid_player_count: int
    bid_abs_error_mean: float
    focal_bid_hit: int
    focal_bid_abs_error: float
    position_rewards: dict[int, float]


@dataclass
class RolloutBuffer:
    samples: list[RolloutSample] = field(default_factory=list)
    round_outcomes: list[RoundOutcome] = field(default_factory=list)

    def extend(self, samples: Iterable[RolloutSample]) -> None:
        self.samples.extend(samples)

    def ready_samples(self) -> list[RolloutSample]:
        for sample in self.samples:
            if (
                sample.return_target is None
                or sample.value_target is None
                or sample.advantage_target is None
                or sample.final_trick_targets is None
                or sample.final_bid_targets is None
                or sample.owner_targets is None
            ):
                raise RuntimeError("RolloutBuffer contains samples without terminal targets.")
        return self.samples

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class RolloutStats:
    rounds: int
    configurations: int
    samples: int
    bid_samples: int
    play_samples: int
    bid_hit_rate: float
    bid_abs_error_mean: float
    all_player_bid_hit_rate: float
    all_player_bid_abs_error_mean: float
    heuristic_relative_reward: float
    self_play_rounds: int
    heuristic_rounds: int
    mixed_rounds: int
    historical_rounds: int
    return_mean: float
    return_std: float
    old_value_mean: float
    advantage_std: float
    old_bid_entropy_mean: float
    old_play_entropy_mean: float
    bid_advantage_mean: float
    bid_advantage_std: float
    bid_advantage_abs_p50: float
    bid_advantage_abs_p90: float
    play_advantage_mean: float
    play_advantage_std: float
    play_advantage_abs_p50: float
    play_advantage_abs_p90: float


@dataclass
class PredictionStats:
    samples: int
    value_mae: float
    value_mse: float
    value_explained_variance: float
    bid_value_explained_variance: float
    play_value_explained_variance: float
    trick_implied_value_explained_variance: float
    bid_trick_implied_value_explained_variance: float
    play_trick_implied_value_explained_variance: float
    trick_count_accuracy: float
    trick_count_true_prob: float
    owner_accuracy: float
    owner_true_prob: float
    owner_brier: float
    owner_opponent_accuracy: float
    owner_opponent_true_prob: float
    owner_capacity_mae: float
    owner_capacity_max_error: float
    owner_raw_capacity_mae: float
    hit_prob_brier: float
    bid_entropy: float
    play_entropy: float
    bid_max_prob: float
    play_max_prob: float


@dataclass
class UpdateStats:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    auxiliary_loss: float
    trick_loss: float
    owner_loss: float
    owner_ce_loss: float
    owner_capacity_loss: float
    approx_kl: float
    clip_fraction: float
    samples: int
    configurations: int


@dataclass
class _ActiveEpisode:
    env: PlumpEnv
    spec: RoundSpec
    episode_id: int
    opponent_arm: OpponentArm
    trainable_players: frozenset[int]
    opponent_policies: dict[int, ActionPolicy | None]
    focal_player: int
    game_schedule: tuple[int, ...] = ()
    completed_rounds: int = 0
    samples: list[RolloutSample] = field(default_factory=list)


class PPOTrainer:
    """Collect a balanced 24-cell curriculum and optimize a shared policy."""

    def __init__(
        self,
        model: PlumpTransformerModel | None = None,
        config: TrainingConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self._validate_config()
        self.device = torch.device(self.config.device) if self.config.device else best_torch_device()
        torch.manual_seed(self.config.seed)
        self.model = model or PlumpTransformerModel(self.config.model_config)
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.rng = random.Random(self.config.seed)
        self.position_baseline = PositionBaseline(self.config.position_baseline_decay)
        self.heuristic_policy = HeuristicPolicy()
        self.historical_policies: list[ActionPolicy] = [
            ModelPolicy.from_checkpoint(path, device=self.device, greedy=False)
            for path in self.config.historical_checkpoint_paths
        ]

    def balanced_round_specs(self) -> list[RoundSpec]:
        specs = [
            spec
            for spec in self.config.specs
            for _ in range(self.config.rounds_per_configuration)
        ]
        self.rng.shuffle(specs)
        return specs

    def balanced_round_schedule(self) -> list[tuple[RoundSpec, OpponentArm]]:
        fractions = self._effective_arm_fractions()
        schedule = [
            (spec, arm)
            for spec in self.config.specs
            for arm in self._allocate_opponent_arms(
                self.config.rounds_per_configuration,
                fractions,
            )
        ]
        self.rng.shuffle(schedule)
        return schedule

    def collect_rollouts(self, *, iteration: int = 1) -> RolloutBuffer:
        if self.config.training_mode == "game":
            return self._collect_game_rollouts(iteration=iteration)
        return self._collect_round_rollouts(iteration=iteration)

    def _collect_round_rollouts(self, *, iteration: int = 1) -> RolloutBuffer:
        buffer = RolloutBuffer()
        schedule = self.balanced_round_schedule()
        next_episode_id = 0
        active: list[_ActiveEpisode] = []
        pending_baseline_updates: list[tuple[PositionKey, float]] = []

        for _ in range(min(self.config.num_envs, len(schedule))):
            active.append(
                self._new_active_episode(
                    *schedule[next_episode_id],
                    next_episode_id,
                    iteration,
                )
            )
            next_episode_id += 1

        self.model.eval()
        with torch.no_grad():
            while active:
                model_controlled = [
                    episode
                    for episode in active
                    if self._uses_current_policy(
                        episode,
                        episode.env.current_player(),
                    )
                ]
                sampled = (
                    self._sample_batched_actions(model_controlled)
                    if model_controlled
                    else {}
                )
                frozen = [
                    episode
                    for episode in active
                    if not self._uses_current_policy(
                        episode,
                        episode.env.current_player(),
                    )
                ]
                opponent_actions = self._sample_batched_opponent_actions(frozen)
                next_active: list[_ActiveEpisode] = []

                for episode in active:
                    acting_player = episode.env.current_player()
                    if episode.episode_id in sampled:
                        sample, action = sampled[episode.episode_id]
                        if acting_player in episode.trainable_players:
                            episode.samples.append(sample)
                    else:
                        action = opponent_actions[episode.episode_id]
                    episode.env.step(action)

                    if episode.env.is_done():
                        updates = self._assign_terminal_targets(episode)
                        pending_baseline_updates.extend(updates)
                        buffer.extend(episode.samples)
                        buffer.round_outcomes.append(self._round_outcome(episode))
                        if next_episode_id < len(schedule):
                            next_active.append(
                                self._new_active_episode(
                                    *schedule[next_episode_id],
                                    next_episode_id,
                                    iteration,
                                )
                            )
                            next_episode_id += 1
                    else:
                        next_active.append(episode)
                active = next_active

        self._assign_round_weights(buffer)
        self.position_baseline.update_many(pending_baseline_updates)
        if len(buffer.round_outcomes) != self.config.rounds_per_batch:
            raise RuntimeError("Balanced collection did not complete the requested schedule.")
        return buffer

    def _collect_game_rollouts(self, *, iteration: int = 1) -> RolloutBuffer:
        buffer = RolloutBuffer()
        schedule = self._balanced_game_schedule()
        next_episode_id = 0
        active: list[_ActiveEpisode] = []

        for _ in range(min(self.config.num_envs, len(schedule))):
            num_players, focal_player, arm = schedule[next_episode_id]
            active.append(
                self._new_active_game(
                    num_players,
                    focal_player,
                    arm,
                    next_episode_id,
                    iteration,
                )
            )
            next_episode_id += 1

        self.model.eval()
        with torch.no_grad():
            while active:
                model_controlled = [
                    episode
                    for episode in active
                    if self._uses_current_policy(
                        episode,
                        episode.env.current_player(),
                    )
                ]
                sampled = (
                    self._sample_batched_actions(model_controlled)
                    if model_controlled
                    else {}
                )
                frozen = [
                    episode
                    for episode in active
                    if not self._uses_current_policy(
                        episode,
                        episode.env.current_player(),
                    )
                ]
                opponent_actions = self._sample_batched_opponent_actions(frozen)
                next_active: list[_ActiveEpisode] = []

                for episode in active:
                    acting_player = episode.env.current_player()
                    if episode.episode_id in sampled:
                        sample, action = sampled[episode.episode_id]
                        if acting_player in episode.trainable_players:
                            episode.samples.append(sample)
                    else:
                        action = opponent_actions[episode.episode_id]
                    result = episode.env.step(action)

                    if result.info.get("round_ended"):
                        completed = episode.env.state.rounds[episode.completed_rounds]
                        buffer.round_outcomes.append(
                            self._round_outcome_for_state(episode, completed)
                        )
                        episode.completed_rounds += 1

                    if episode.env.is_done():
                        self._assign_game_terminal_targets(episode)
                        buffer.extend(episode.samples)
                        if next_episode_id < len(schedule):
                            num_players, focal_player, arm = schedule[next_episode_id]
                            next_active.append(
                                self._new_active_game(
                                    num_players,
                                    focal_player,
                                    arm,
                                    next_episode_id,
                                    iteration,
                                )
                            )
                            next_episode_id += 1
                    else:
                        next_active.append(episode)
                active = next_active

        self._assign_game_weights(buffer)
        return buffer

    def update(self, buffer: RolloutBuffer) -> UpdateStats:
        samples = buffer.ready_samples()
        if not samples:
            raise ValueError("Cannot update from an empty rollout buffer.")

        returns = torch.tensor([sample.return_target for sample in samples], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([sample.old_value for sample in samples], dtype=torch.float32, device=self.device)
        raw_advantages = torch.tensor(
            [sample.advantage_target for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        policy_enabled = torch.tensor(
            [sample.ppo_policy_enabled for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        enabled_advantages = raw_advantages[policy_enabled.bool()]
        if len(enabled_advantages) > 0:
            advantages = (
                raw_advantages - enabled_advantages.mean()
            ) / (enabled_advantages.std(unbiased=False) + 1e-8)
        else:
            advantages = torch.zeros_like(raw_advantages)
        staged_batch = encoded_observations_to_batch(
            [sample.encoded for sample in samples],
            device=self.device,
        )
        old_logprobs = torch.tensor(
            [sample.old_logprob for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        round_weights = torch.tensor(
            [sample.round_weight for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        residual_targets = torch.tensor(
            [sample.value_target for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        trick_targets = torch.tensor(
            [sample.final_trick_targets for sample in samples],
            dtype=torch.long,
            device=self.device,
        )
        owner_targets = torch.tensor(
            [sample.owner_targets for sample in samples],
            dtype=torch.long,
            device=self.device,
        )

        metric_values: dict[str, list[Tensor]] = defaultdict(list)
        indices = list(range(len(samples)))
        previous_training = self.model.training
        self.model.eval()
        try:
            for _ in range(self.config.ppo_epochs):
                self.rng.shuffle(indices)
                for start in range(0, len(indices), self.config.minibatch_size):
                    logical_selected = indices[start : start + self.config.minibatch_size]
                    logical_tensor = torch.tensor(
                        logical_selected,
                        dtype=torch.long,
                        device=self.device,
                    )
                    logical_count = len(logical_selected)
                    logical_trick_targets = trick_targets.index_select(0, logical_tensor)
                    logical_owner_targets = owner_targets.index_select(0, logical_tensor)
                    trick_label_count = (logical_trick_targets != -100).sum().clamp_min(1)
                    owner_label_count = (logical_owner_targets != -100).sum().clamp_min(1)
                    logical_owner_capacities = (
                        staged_batch.owner_capacities.index_select(
                            0,
                            logical_tensor,
                        )
                    )
                    owner_capacity_count = (
                        logical_owner_capacities > 0.0
                    ).sum().clamp_min(1)
                    step_values: dict[str, Tensor] = {}
                    self.optimizer.zero_grad(set_to_none=True)
                    microbatch_size = self.config.microbatch_size or logical_count
                    for micro_start in range(0, logical_count, microbatch_size):
                        selected = logical_selected[micro_start : micro_start + microbatch_size]
                        selected_tensor = torch.tensor(
                            selected,
                            dtype=torch.long,
                            device=self.device,
                        )
                        mb_samples = [samples[index] for index in selected]
                        mb_advantages = advantages.index_select(0, selected_tensor)
                        mb_old_logprobs = old_logprobs.index_select(0, selected_tensor)
                        weights = round_weights.index_select(0, selected_tensor)
                        mb_policy_enabled = policy_enabled.index_select(0, selected_tensor)
                        batch = index_model_batch(staged_batch, selected_tensor)
                        with model_autocast(self.device, self.config.precision):
                            output = self.model(batch)
                        new_logprobs, entropy_by_sample = self._logprobs_and_entropy(output, mb_samples)
                        log_ratio = new_logprobs - mb_old_logprobs
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.config.ppo_clip_eps,
                            1.0 + self.config.ppo_clip_eps,
                        )
                        policy_terms = -torch.min(
                            ratio * mb_advantages,
                            clipped_ratio * mb_advantages,
                        )
                        # Uniform over configurations, mean over rounds, sum over decisions.
                        objective_scale = len(samples) * weights
                        policy_scale = objective_scale * mb_policy_enabled
                        policy_loss = (policy_scale * policy_terms).sum() / logical_count
                        entropy = (policy_scale * entropy_by_sample).sum() / logical_count

                        mb_residual_targets = residual_targets.index_select(0, selected_tensor)
                        value_terms = F.smooth_l1_loss(
                            output.value.squeeze(-1).float(),
                            mb_residual_targets,
                            reduction="none",
                        )
                        value_loss = (objective_scale * value_terms).sum() / logical_count
                        mb_trick_targets = trick_targets.index_select(0, selected_tensor)
                        trick_loss = F.cross_entropy(
                            output.masked_trick_count_logits.float().reshape(
                                -1,
                                self.config.model_config.bid_count,
                            ),
                            mb_trick_targets.reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        ) / trick_label_count
                        mb_owner_targets = owner_targets.index_select(0, selected_tensor)
                        active_owner_targets = mb_owner_targets != -100
                        safe_owner_targets = mb_owner_targets.clamp_min(0)
                        owner_true_probabilities = output.owner_probs.gather(
                            -1,
                            safe_owner_targets.unsqueeze(-1),
                        ).squeeze(-1)
                        owner_ce_loss = -torch.log(
                            owner_true_probabilities.clamp_min(1e-12)
                        )[active_owner_targets].sum() / owner_label_count
                        owner_capacities = batch.owner_capacities.float()
                        active_capacities = owner_capacities > 0.0
                        raw_expected_counts = (
                            output.owner_pre_sinkhorn_probs.sum(dim=1)
                        )
                        hidden_card_counts = owner_capacities.sum(
                            dim=-1,
                            keepdim=True,
                        ).clamp_min(1.0)
                        owner_capacity_loss = (
                            (
                                (
                                    raw_expected_counts
                                    - owner_capacities
                                )
                                / hidden_card_counts
                            ).pow(2)[active_capacities]
                            .sum()
                            / owner_capacity_count
                        )
                        owner_loss = (
                            owner_ce_loss
                            + self.config.owner_capacity_coef
                            * owner_capacity_loss
                        )
                        auxiliary_loss = (
                            self.config.trick_coef * trick_loss
                            + self.config.owner_coef * owner_loss
                        )
                        total_loss = (
                            policy_loss
                            + self.config.value_coef * value_loss
                            - self.config.entropy_coef * entropy
                            + auxiliary_loss
                        )
                        total_loss.backward()

                        with torch.no_grad():
                            enabled_count = mb_policy_enabled.sum().clamp_min(1.0)
                            approx_kl = (
                                mb_policy_enabled
                                * ((ratio - 1.0) - log_ratio)
                            ).sum() / enabled_count
                            clip_fraction = (
                                mb_policy_enabled
                                * (
                                    (ratio - 1.0).abs()
                                    > self.config.ppo_clip_eps
                                ).float()
                            ).sum() / enabled_count
                        values = {
                            "total_loss": total_loss,
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                            "entropy": entropy,
                            "auxiliary_loss": auxiliary_loss,
                            "trick_loss": trick_loss,
                            "owner_loss": owner_loss,
                            "owner_ce_loss": owner_ce_loss,
                            "owner_capacity_loss": owner_capacity_loss,
                            "approx_kl": approx_kl,
                            "clip_fraction": clip_fraction,
                        }
                        for key, value in values.items():
                            detached = value.detach()
                            step_values[key] = step_values.get(key, detached.new_zeros(())) + detached

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                    for key, value in step_values.items():
                        metric_values[key].append(value)
        finally:
            if previous_training:
                self.model.train()

        totals = {
            key: float(torch.stack(values).mean().cpu())
            for key, values in metric_values.items()
        }
        return UpdateStats(
            total_loss=totals["total_loss"],
            policy_loss=totals["policy_loss"],
            value_loss=totals["value_loss"],
            entropy=totals["entropy"],
            auxiliary_loss=totals["auxiliary_loss"],
            trick_loss=totals["trick_loss"],
            owner_loss=totals["owner_loss"],
            owner_ce_loss=totals["owner_ce_loss"],
            owner_capacity_loss=totals["owner_capacity_loss"],
            approx_kl=totals["approx_kl"],
            clip_fraction=totals["clip_fraction"],
            samples=len(samples),
            configurations=(
                len(self.config.player_counts)
                if self.config.training_mode == "game"
                else len(self.config.specs)
            ),
        )

    def summarize_rollout(self, buffer: RolloutBuffer) -> RolloutStats:
        samples = buffer.ready_samples()
        returns = [float(sample.return_target) for sample in samples]
        old_values = [sample.old_value for sample in samples]
        bid_entropies = [sample.old_entropy for sample in samples if sample.phase == "bid"]
        play_entropies = [sample.old_entropy for sample in samples if sample.phase == "play"]
        hit_count = sum(outcome.bid_hit_count for outcome in buffer.round_outcomes)
        player_count = sum(outcome.bid_player_count for outcome in buffer.round_outcomes)
        bid_advantages = [
            float(sample.advantage_target)
            for sample in samples
            if sample.phase == "bid"
        ]
        play_advantages = [
            float(sample.advantage_target)
            for sample in samples
            if sample.phase == "play"
        ]
        return RolloutStats(
            rounds=len(buffer.round_outcomes),
            configurations=len({outcome.spec for outcome in buffer.round_outcomes}),
            samples=len(samples),
            bid_samples=sum(sample.phase == "bid" for sample in samples),
            play_samples=sum(sample.phase == "play" for sample in samples),
            bid_hit_rate=_mean(
                [float(outcome.focal_bid_hit) for outcome in buffer.round_outcomes]
            ),
            bid_abs_error_mean=_mean(
                [outcome.focal_bid_abs_error for outcome in buffer.round_outcomes]
            ),
            all_player_bid_hit_rate=hit_count / max(player_count, 1),
            all_player_bid_abs_error_mean=_mean(
                [outcome.bid_abs_error_mean for outcome in buffer.round_outcomes]
            ),
            heuristic_relative_reward=_mean(
                [
                    float(outcome.focal_reward)
                    for outcome in buffer.round_outcomes
                    if outcome.opponent_arm == "heuristic"
                    and outcome.focal_reward is not None
                ]
            ),
            self_play_rounds=sum(
                outcome.opponent_arm == "self"
                for outcome in buffer.round_outcomes
            ),
            heuristic_rounds=sum(
                outcome.opponent_arm == "heuristic"
                for outcome in buffer.round_outcomes
            ),
            mixed_rounds=sum(
                outcome.opponent_arm == "mixed"
                for outcome in buffer.round_outcomes
            ),
            historical_rounds=sum(
                outcome.opponent_arm == "historical"
                for outcome in buffer.round_outcomes
            ),
            return_mean=_mean(returns),
            return_std=_std(returns),
            old_value_mean=_mean(old_values),
            advantage_std=_std(
                [float(sample.advantage_target) for sample in samples]
            ),
            old_bid_entropy_mean=_mean(bid_entropies),
            old_play_entropy_mean=_mean(play_entropies),
            bid_advantage_mean=_mean(bid_advantages),
            bid_advantage_std=_std(bid_advantages),
            bid_advantage_abs_p50=_percentile_abs(bid_advantages, 0.50),
            bid_advantage_abs_p90=_percentile_abs(bid_advantages, 0.90),
            play_advantage_mean=_mean(play_advantages),
            play_advantage_std=_std(play_advantages),
            play_advantage_abs_p50=_percentile_abs(play_advantages, 0.50),
            play_advantage_abs_p90=_percentile_abs(play_advantages, 0.90),
        )

    def compute_prediction_stats(
        self,
        buffer: RolloutBuffer,
        *,
        max_samples: int = 2048,
        minibatch_size: int | None = None,
    ) -> PredictionStats:
        samples = _evenly_spaced_samples(buffer.ready_samples(), max_samples)
        minibatch_size = minibatch_size or self.config.minibatch_size
        values: list[float] = []
        targets: list[float] = []
        bid_values: list[float] = []
        bid_targets: list[float] = []
        play_values: list[float] = []
        play_targets: list[float] = []
        trick_implied_values: list[float] = []
        bid_trick_implied_values: list[float] = []
        play_trick_implied_values: list[float] = []
        trick_correct = trick_count = owner_correct = owner_count = 0
        trick_true_probs: list[float] = []
        owner_true_probs: list[float] = []
        owner_brier: list[float] = []
        owner_opponent_correct = owner_opponent_count = 0
        owner_opponent_true_probs: list[float] = []
        owner_capacity_errors: list[float] = []
        owner_raw_capacity_errors: list[float] = []
        hit_brier: list[float] = []
        bid_entropies: list[float] = []
        play_entropies: list[float] = []
        bid_max_probs: list[float] = []
        play_max_probs: list[float] = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(samples), minibatch_size):
                mb = samples[start : start + minibatch_size]
                batch = encoded_observations_to_batch([sample.encoded for sample in mb], device=self.device)
                with model_autocast(self.device, self.config.precision):
                    output = self.model(batch)
                intercepts = torch.tensor(
                    [sample.position_intercept for sample in mb],
                    dtype=torch.float32,
                    device=self.device,
                )
                full_values = output.value.squeeze(-1).float() + intercepts
                full_value_rows = full_values.cpu().tolist()
                target_rows = [float(sample.return_target) for sample in mb]
                values.extend(full_value_rows)
                targets.extend(target_rows)
                for sample, value, target in zip(mb, full_value_rows, target_rows):
                    if sample.phase == "bid":
                        bid_values.append(value)
                        bid_targets.append(target)
                    else:
                        play_values.append(value)
                        play_targets.append(target)

                self._collect_policy_stats(
                    output,
                    mb,
                    bid_entropies,
                    play_entropies,
                    bid_max_probs,
                    play_max_probs,
                )
                trick_targets = torch.tensor(
                    [sample.final_trick_targets for sample in mb],
                    dtype=torch.long,
                    device=self.device,
                )
                active_tricks = trick_targets >= 0
                trick_probs = torch.softmax(
                    output.masked_trick_count_logits.float(),
                    dim=-1,
                )
                final_bid_targets = torch.tensor(
                    [sample.final_bid_targets for sample in mb],
                    dtype=torch.long,
                    device=self.device,
                )
                implied_relative_values = _trick_implied_relative_values(
                    trick_probs,
                    final_bid_targets,
                    batch.active_player_mask,
                )
                implied_rows = (
                    implied_relative_values.cpu().tolist()
                )
                trick_implied_values.extend(implied_rows)
                for sample, implied in zip(mb, implied_rows):
                    if sample.phase == "bid":
                        bid_trick_implied_values.append(implied)
                    else:
                        play_trick_implied_values.append(implied)
                trick_predictions = output.masked_trick_count_logits.argmax(dim=-1)
                trick_correct += int(((trick_predictions == trick_targets) & active_tricks).sum().cpu())
                trick_count += int(active_tricks.sum().cpu())
                safe_tricks = trick_targets.clamp(min=0, max=self.config.model_config.bid_count - 1)
                gathered_tricks = trick_probs.gather(-1, safe_tricks.unsqueeze(-1)).squeeze(-1)
                trick_true_probs.extend(gathered_tricks[active_tricks].cpu().tolist())

                owner_targets = torch.tensor(
                    [sample.owner_targets for sample in mb],
                    dtype=torch.long,
                    device=self.device,
                )
                active_owners = owner_targets >= 0
                owner_predictions = output.owner_probs.argmax(dim=-1)
                owner_correct += int(((owner_predictions == owner_targets) & active_owners).sum().cpu())
                owner_count += int(active_owners.sum().cpu())
                safe_owners = owner_targets.clamp(min=0)
                owner_probs = output.owner_probs.float()
                gathered_owners = owner_probs.gather(
                    -1,
                    safe_owners.unsqueeze(-1),
                ).squeeze(-1)
                owner_true_probs.extend(gathered_owners[active_owners].cpu().tolist())
                opponent_owners = (
                    active_owners
                    & (
                        owner_targets
                        != self.config.model_config.undealt_owner_class
                    )
                )
                owner_opponent_correct += int(
                    (
                        (owner_predictions == owner_targets)
                        & opponent_owners
                    )
                    .sum()
                    .cpu()
                )
                owner_opponent_count += int(opponent_owners.sum().cpu())
                owner_opponent_true_probs.extend(
                    gathered_owners[opponent_owners].cpu().tolist()
                )
                if active_owners.any():
                    one_hot = F.one_hot(
                        safe_owners, num_classes=self.config.model_config.owner_class_count
                    ).float()
                    squared = ((owner_probs - one_hot) ** 2).sum(dim=-1)
                    owner_brier.extend(squared[active_owners].cpu().tolist())
                capacities = batch.owner_capacities.float()
                active_capacities = capacities > 0.0
                projected_count_errors = (
                    output.owner_probs.sum(dim=1) - capacities
                ).abs()
                raw_count_errors = (
                    output.owner_pre_sinkhorn_probs.sum(dim=1)
                    - capacities
                ).abs()
                owner_capacity_errors.extend(
                    projected_count_errors[active_capacities]
                    .cpu()
                    .tolist()
                )
                owner_raw_capacity_errors.extend(
                    raw_count_errors[active_capacities]
                    .cpu()
                    .tolist()
                )

                has_bid = (batch.bid_values >= 0) & batch.active_player_mask & active_tricks
                actual_hit = trick_targets == batch.bid_values
                if has_bid.any():
                    hit_brier.extend(
                        ((output.hit_bid_probs.float()[has_bid] - actual_hit[has_bid].float()) ** 2)
                        .cpu()
                        .tolist()
                    )

        return PredictionStats(
            samples=len(samples),
            value_mae=_mae(values, targets),
            value_mse=_mse(values, targets),
            value_explained_variance=_explained_variance(values, targets),
            bid_value_explained_variance=_explained_variance(
                bid_values,
                bid_targets,
            ),
            play_value_explained_variance=_explained_variance(
                play_values,
                play_targets,
            ),
            trick_implied_value_explained_variance=(
                _explained_variance(
                    trick_implied_values,
                    targets,
                )
            ),
            bid_trick_implied_value_explained_variance=(
                _explained_variance(
                    bid_trick_implied_values,
                    bid_targets,
                )
            ),
            play_trick_implied_value_explained_variance=(
                _explained_variance(
                    play_trick_implied_values,
                    play_targets,
                )
            ),
            trick_count_accuracy=trick_correct / max(trick_count, 1),
            trick_count_true_prob=_mean(trick_true_probs),
            owner_accuracy=owner_correct / max(owner_count, 1),
            owner_true_prob=_mean(owner_true_probs),
            owner_brier=_mean(owner_brier),
            owner_opponent_accuracy=(
                owner_opponent_correct
                / max(owner_opponent_count, 1)
            ),
            owner_opponent_true_prob=_mean(
                owner_opponent_true_probs
            ),
            owner_capacity_mae=_mean(owner_capacity_errors),
            owner_capacity_max_error=max(
                owner_capacity_errors,
                default=0.0,
            ),
            owner_raw_capacity_mae=_mean(owner_raw_capacity_errors),
            hit_prob_brier=_mean(hit_brier),
            bid_entropy=_mean(bid_entropies),
            play_entropy=_mean(play_entropies),
            bid_max_prob=_mean(bid_max_probs),
            play_max_prob=_mean(play_max_probs),
        )

    def save_checkpoint(self, path: str | Path, *, iteration: int, extra: dict | None = None) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": asdict(self.config.model_config),
            "training_config": asdict(self.config),
            "position_baseline": self.position_baseline.state_dict(),
            "include_game_context": (
                self.config.training_mode == "game"
                or self.config.include_game_context
            ),
            "precision": self.config.precision,
            "rules_fingerprint": rules_fingerprint(),
            "extra": extra or {},
        }
        torch.save(payload, Path(path))

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = False) -> dict[str, object]:
        payload = _load_checkpoint_payload(path, self.device)
        if payload.get("schema_version", 1) != SCHEMA_VERSION:
            raise ValueError(
                "Only schema-v4 checkpoints can resume training; older schemas "
                "are initialization/evaluation-only."
            )
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("Checkpoint rules fingerprint does not match the active rules.")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.position_baseline.load_state_dict(payload.get("position_baseline", {}))
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return {
            "path": str(path),
            "iteration": payload.get("iteration"),
            "optimizer_loaded": load_optimizer,
            "schema_version": payload["schema_version"],
        }

    def warm_start_v2(self, path: str | Path) -> dict[str, object]:
        """Import compatible v2 trunk weights with a fresh v4 owner head."""

        payload = _load_checkpoint_payload(path, self.device)
        if payload.get("schema_version") != 2:
            raise ValueError("warm_start_v2 requires a schema-v2 checkpoint.")
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("V2 checkpoint rules fingerprint does not match.")
        migration = load_v2_weights(
            self.model,
            payload["model_state_dict"],
        )
        self.position_baseline.load_state_dict(
            payload.get("position_baseline", {})
        )
        return {
            "path": str(path),
            "source_iteration": payload.get("iteration"),
            "source_schema_version": 2,
            "loaded_tensors": len(migration["loaded"]),
            "fresh_tensors": migration["fresh"],
            "optimizer_loaded": False,
        }

    def warm_start_v3(self, path: str | Path) -> dict[str, object]:
        """Import compatible v3 trunk weights with a fresh v4 owner head."""

        payload = _load_checkpoint_payload(path, self.device)
        if payload.get("schema_version") != 3:
            raise ValueError("warm_start_v3 requires a schema-v3 checkpoint.")
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("V3 checkpoint rules fingerprint does not match.")
        migration = load_v3_weights(
            self.model,
            payload["model_state_dict"],
        )
        self.position_baseline.load_state_dict(
            payload.get("position_baseline", {})
        )
        return {
            "path": str(path),
            "source_iteration": payload.get("iteration"),
            "source_schema_version": 3,
            "loaded_tensors": len(migration["loaded"]),
            "fresh_tensors": migration["fresh"],
            "dropped_tensors": migration["dropped"],
            "optimizer_loaded": False,
        }

    def add_historical_checkpoint(self, path: str | Path) -> None:
        self.historical_policies.append(
            ModelPolicy.from_checkpoint(path, device=self.device, greedy=False)
        )
        if len(self.historical_policies) > self.config.historical_max_snapshots:
            self.historical_policies = self.historical_policies[
                -self.config.historical_max_snapshots :
            ]

    def _effective_arm_fractions(self) -> dict[OpponentArm, float]:
        fractions: dict[OpponentArm, float] = {
            "self": self.config.self_play_fraction,
            "heuristic": self.config.heuristic_fraction,
            "mixed": self.config.mixed_fraction,
            "historical": self.config.historical_fraction,
        }
        if not self.historical_policies:
            fractions["self"] += fractions["historical"]
            fractions["historical"] = 0.0
        return fractions

    def _allocate_opponent_arms(
        self,
        rounds: int,
        fractions: dict[OpponentArm, float],
    ) -> list[OpponentArm]:
        return allocate_opponent_arms(
            rounds,
            fractions,
            self.rng,
        )

    def _new_active_episode(
        self,
        spec: RoundSpec,
        arm: OpponentArm,
        episode_id: int,
        iteration: int,
    ) -> _ActiveEpisode:
        start_player = self.rng.randrange(spec.num_players)
        env = PlumpEnv(round_game_config(spec, bidding_start_player=start_player), seed=self.rng.randrange(2**31))
        env.reset()
        del iteration
        focal_player = episode_id % spec.num_players
        opponent_policies = self._opponent_policies(
            spec.num_players,
            focal_player,
            arm,
        )
        return _ActiveEpisode(
            env=env,
            spec=spec,
            episode_id=episode_id,
            opponent_arm=arm,
            trainable_players=_current_policy_players(
                focal_player,
                opponent_policies,
            ),
            opponent_policies=opponent_policies,
            focal_player=focal_player,
        )

    def _balanced_game_schedule(
        self,
    ) -> list[tuple[int, int, OpponentArm]]:
        fractions = self._effective_arm_fractions()
        schedule = []
        for num_players in self.config.player_counts:
            for focal_player in range(num_players):
                for arm in self._allocate_opponent_arms(
                    self.config.games_per_player_seat,
                    fractions,
                ):
                    schedule.append((num_players, focal_player, arm))
        self.rng.shuffle(schedule)
        return schedule

    def _new_active_game(
        self,
        num_players: int,
        focal_player: int,
        arm: OpponentArm,
        episode_id: int,
        iteration: int,
    ) -> _ActiveEpisode:
        del iteration
        game_schedule = self.config.resolved_game_schedule
        env = PlumpEnv(
            GameConfig(
                num_players=num_players,
                hand_sizes=list(game_schedule),
            ),
            seed=self.rng.randrange(2**31),
        )
        env.reset()
        spec = RoundSpec(num_players, game_schedule[0])
        opponent_policies = self._opponent_policies(
            num_players,
            focal_player,
            arm,
        )
        return _ActiveEpisode(
            env=env,
            spec=spec,
            episode_id=episode_id,
            opponent_arm=arm,
            trainable_players=_current_policy_players(
                focal_player,
                opponent_policies,
            ),
            opponent_policies=opponent_policies,
            focal_player=focal_player,
            game_schedule=game_schedule,
        )

    def _opponent_policies(
        self,
        num_players: int,
        focal_player: int,
        arm: OpponentArm,
    ) -> dict[int, ActionPolicy | None]:
        opponent_policies: dict[int, ActionPolicy | None] = {}
        for player in range(num_players):
            if player == focal_player:
                continue
            if arm == "self":
                policy = None
            elif arm == "heuristic":
                policy = self.heuristic_policy
            elif arm == "historical":
                policy = self.rng.choice(self.historical_policies)
            else:
                category = self.rng.choice(("self", "heuristic", "historical"))
                if category == "self":
                    policy = None
                elif category == "heuristic":
                    policy = self.heuristic_policy
                else:
                    policy = (
                        self.rng.choice(self.historical_policies)
                        if self.historical_policies
                        else None
                    )
            opponent_policies[player] = policy
        return opponent_policies

    @staticmethod
    def _uses_current_policy(
        episode: _ActiveEpisode,
        player: int,
    ) -> bool:
        return (
            player in episode.trainable_players
            or episode.opponent_policies.get(player) is None
        )

    def _sample_batched_opponent_actions(
        self,
        episodes: list[_ActiveEpisode],
    ) -> dict[int, BidAction | PlayCardAction]:
        grouped: dict[int, list[_ActiveEpisode]] = defaultdict(list)
        policies: dict[int, ActionPolicy] = {}
        for episode in episodes:
            player = episode.env.current_player()
            policy = episode.opponent_policies.get(player)
            if policy is None:
                raise RuntimeError("Non-trainable turn has no frozen opponent policy.")
            key = id(policy)
            grouped[key].append(episode)
            policies[key] = policy

        actions: dict[int, BidAction | PlayCardAction] = {}
        for key, rows in grouped.items():
            policy = policies[key]
            if isinstance(policy, ModelPolicy):
                selected = policy.act_many(
                    [episode.env for episode in rows],
                    rngs=[self.rng] * len(rows),
                )
            else:
                selected = [
                    policy.act(episode.env, rng=self.rng)
                    for episode in rows
                ]
            actions.update(
                (episode.episode_id, action)
                for episode, action in zip(rows, selected)
            )
        return actions

    def _sample_batched_actions(
        self,
        episodes: list[_ActiveEpisode],
    ) -> dict[int, tuple[RolloutSample, BidAction | PlayCardAction]]:
        encoded: list[EncodedObservation] = []
        owner_targets: list[list[int]] = []
        phases: list[SamplePhase] = []
        players: list[int] = []
        observations: list[Observation] = []
        for episode in episodes:
            player = episode.env.current_player()
            observation = episode.env.get_observation(player)
            observations.append(observation)
            encoded_observation = encode_observation(
                observation,
                self.config.model_config,
                include_game_context=(
                    self.config.training_mode == "game"
                    or self.config.include_game_context
                ),
            )
            encoded.append(encoded_observation)
            owner_targets.append(
                _owner_targets_relative(
                    episode.env,
                    player,
                    encoded_observation.owner_valid_mask,
                    self.config.model_config,
                )
            )
            phases.append("bid" if episode.env.phase() == Phase.BIDDING else "play")
            players.append(player)

        batch = encoded_observations_to_batch(encoded, device=self.device)
        with model_autocast(self.device, self.config.precision):
            output = self.model(batch, need_owner=False)
        implied_values: list[float] | None = None
        if self.config.trick_baseline and self.config.training_mode != "game":
            # Trick-head-implied expected relative score acts as a per-state
            # potential; folding it into the intercept residualizes both the
            # value target and the advantage without biasing the optimum.
            implied_values = _trick_implied_relative_values(
                torch.softmax(output.masked_trick_count_logits.float(), dim=-1),
                batch.bid_values,
                batch.active_player_mask,
            ).cpu().tolist()
        bid_mask = torch.tensor(
            [phase == "bid" for phase in phases],
            dtype=torch.bool,
            device=self.device,
        )
        distribution = Categorical(logits=combined_action_logits(output, bid_mask))
        action_indices = distribution.sample()
        rollout_rows = torch.stack(
            (
                action_indices.float(),
                output.value.squeeze(-1).float(),
                distribution.log_prob(action_indices),
                distribution.entropy(),
            ),
            dim=-1,
        ).cpu().tolist()
        results: dict[int, tuple[RolloutSample, BidAction | PlayCardAction]] = {}
        for index, episode in enumerate(episodes):
            phase = phases[index]
            action_index_value, residual, old_logprob, old_entropy = rollout_rows[index]
            action_index = int(action_index_value)
            position = encoded[index].bidding_position
            current_round = episode.env.state.current_round
            current_spec = RoundSpec(
                episode.env.config.num_players,
                current_round.hand_size,
            )
            key = (current_spec.num_players, current_spec.hand_size, position)
            intercept = (
                0.0
                if self.config.training_mode == "game"
                else self.position_baseline.get(key)
            )
            if implied_values is not None:
                intercept += implied_values[index]
            player = players[index]
            action: BidAction | PlayCardAction
            if phase == "bid":
                action = BidAction(player, action_index)
            else:
                action = PlayCardAction(player, card_from_id(action_index))
            sample = RolloutSample(
                encoded=encoded[index],
                phase=phase,
                action_index=action_index,
                old_logprob=old_logprob,
                old_value=intercept + residual,
                old_residual_value=residual,
                old_entropy=old_entropy,
                position_intercept=intercept,
                acting_player=player,
                episode_id=episode.episode_id,
                round_id=current_round.round_index,
                spec=current_spec,
                bidding_position=position,
                opponent_arm=episode.opponent_arm,
                owner_targets=owner_targets[index],
                observation=_snapshot_observation(observations[index]),
                trick_position=(
                    len(observations[index].current_trick.plays)
                    if observations[index].current_trick is not None
                    else -1
                ),
            )
            results[episode.episode_id] = (sample, action)
        return results

    def _assign_terminal_targets(
        self,
        episode: _ActiveEpisode,
    ) -> list[tuple[PositionKey, float]]:
        round_state = episode.env.state.current_round
        rewards = compute_relative_rewards(round_state.round_scores)
        bid_positions = {bid.player: bid.position for bid in round_state.bids}
        for player in episode.trainable_players:
            self._assign_gae(
                [
                    sample
                    for sample in episode.samples
                    if sample.acting_player == player
                ],
                terminal_reward=rewards[player],
                gae_lambda=self.config.round_gae_lambda,
            )
        for sample in episode.samples:
            sample.final_trick_targets = _final_tricks_relative(
                round_state.tricks_won,
                sample.acting_player,
                episode.env.config.num_players,
                self.config.model_config,
            )
            sample.final_bid_targets = _final_bids_relative(
                round_state.bids,
                sample.acting_player,
                episode.env.config.num_players,
                self.config.model_config,
            )
        return [
            (
                (episode.spec.num_players, episode.spec.hand_size, bid_positions[player]),
                rewards[player],
            )
            for player in episode.trainable_players
        ]

    def _round_outcome(self, episode: _ActiveEpisode) -> RoundOutcome:
        return self._round_outcome_for_state(
            episode,
            episode.env.state.current_round,
        )

    def _round_outcome_for_state(
        self,
        episode: _ActiveEpisode,
        round_state: RoundState,
    ) -> RoundOutcome:
        bids = {bid.player: bid for bid in round_state.bids}
        rewards = compute_relative_rewards(round_state.round_scores)
        focal_player = episode.focal_player
        focal_bid = bids[focal_player]
        return RoundOutcome(
            episode_id=episode.episode_id,
            spec=RoundSpec(episode.env.config.num_players, round_state.hand_size),
            opponent_arm=episode.opponent_arm,
            focal_reward=rewards[focal_player],
            bid_hit_count=sum(
                round_state.tricks_won[player] == bid.value
                for player, bid in bids.items()
            ),
            bid_player_count=len(bids),
            bid_abs_error_mean=_mean(
                [
                    float(abs(round_state.tricks_won[player] - bid.value))
                    for player, bid in bids.items()
                ]
            ),
            focal_bid_hit=int(
                round_state.tricks_won[focal_player] == focal_bid.value
            ),
            focal_bid_abs_error=float(
                abs(round_state.tricks_won[focal_player] - focal_bid.value)
            ),
            position_rewards={bid.position: rewards[player] for player, bid in bids.items()},
        )

    def _assign_game_terminal_targets(self, episode: _ActiveEpisode) -> None:
        rewards = compute_relative_rewards(episode.env.state.cumulative_scores)
        for player in episode.trainable_players:
            self._assign_gae(
                [
                    sample
                    for sample in episode.samples
                    if sample.acting_player == player
                ],
                terminal_reward=rewards[player],
                gae_lambda=self.config.game_gae_lambda,
            )
        rounds = {
            round_state.round_index: round_state
            for round_state in episode.env.state.rounds
        }
        for sample in episode.samples:
            round_state = rounds[sample.round_id]
            sample.final_trick_targets = _final_tricks_relative(
                round_state.tricks_won,
                sample.acting_player,
                episode.env.config.num_players,
                self.config.model_config,
            )
            sample.final_bid_targets = _final_bids_relative(
                round_state.bids,
                sample.acting_player,
                episode.env.config.num_players,
                self.config.model_config,
            )

    def _assign_gae(
        self,
        samples: list[RolloutSample],
        *,
        terminal_reward: float,
        gae_lambda: float,
    ) -> None:
        gae = 0.0
        next_value = 0.0
        for reverse_index, sample in enumerate(reversed(samples)):
            reward = terminal_reward if reverse_index == 0 else 0.0
            delta = (
                reward
                + self.config.gae_gamma * next_value
                - sample.old_value
            )
            gae = (
                delta
                + self.config.gae_gamma * gae_lambda * gae
            )
            sample.advantage_target = gae
            sample.return_target = gae + sample.old_value
            sample.value_target = (
                sample.return_target - sample.position_intercept
            )
            next_value = sample.old_value

    def _assign_round_weights(self, buffer: RolloutBuffer) -> None:
        rounds_per_cell: dict[tuple[RoundSpec, OpponentArm], int] = defaultdict(int)
        for outcome in buffer.round_outcomes:
            rounds_per_cell[(outcome.spec, outcome.opponent_arm)] += 1
        # Every trainable seat contributes one trajectory per round, so each
        # round is averaged over its seats to keep the arm balance exact.
        seats_per_episode: dict[int, set[int]] = defaultdict(set)
        for sample in buffer.samples:
            seats_per_episode[sample.episode_id].add(sample.acting_player)
        fractions = self._effective_arm_fractions()
        configuration_count = len(self.config.specs)
        for sample in buffer.samples:
            sample.round_weight = fractions[sample.opponent_arm] / (
                configuration_count
                * rounds_per_cell[(sample.spec, sample.opponent_arm)]
                * len(seats_per_episode[sample.episode_id])
            )

    def _assign_game_weights(self, buffer: RolloutBuffer) -> None:
        episodes_by_cell: dict[
            tuple[int, int, OpponentArm],
            set[int],
        ] = defaultdict(set)
        for sample in buffer.samples:
            episodes_by_cell[
                (sample.spec.num_players, sample.acting_player, sample.opponent_arm)
            ].add(sample.episode_id)
        fractions = self._effective_arm_fractions()
        player_count_options = len(self.config.player_counts)
        for sample in buffer.samples:
            key = (
                sample.spec.num_players,
                sample.acting_player,
                sample.opponent_arm,
            )
            sample.round_weight = fractions[sample.opponent_arm] / (
                player_count_options
                * sample.spec.num_players
                * len(episodes_by_cell[key])
            )

    def _logprobs_and_entropy(
        self,
        output,
        samples: list[RolloutSample],
    ) -> tuple[Tensor, Tensor]:
        logprobs = torch.empty(len(samples), dtype=torch.float32, device=self.device)
        entropies = torch.empty(len(samples), dtype=torch.float32, device=self.device)
        bid_indices = [index for index, sample in enumerate(samples) if sample.phase == "bid"]
        play_indices = [index for index, sample in enumerate(samples) if sample.phase == "play"]
        for indices, logits in (
            (bid_indices, output.masked_bid_logits),
            (play_indices, output.masked_card_logits),
        ):
            if not indices:
                continue
            actions = torch.tensor(
                [samples[index].action_index for index in indices],
                dtype=torch.long,
                device=self.device,
            )
            distribution = Categorical(logits=logits[indices].float())
            logprobs[indices] = distribution.log_prob(actions)
            entropies[indices] = distribution.entropy()
        return logprobs, entropies

    def _trick_loss(self, logits: Tensor, samples: list[RolloutSample]) -> Tensor:
        targets = torch.tensor(
            [sample.final_trick_targets for sample in samples],
            dtype=torch.long,
            device=self.device,
        )
        return F.cross_entropy(
            logits.reshape(-1, self.config.model_config.bid_count),
            targets.reshape(-1),
            ignore_index=-100,
        )

    def _collect_policy_stats(
        self,
        output,
        samples: list[RolloutSample],
        bid_entropies: list[float],
        play_entropies: list[float],
        bid_max_probs: list[float],
        play_max_probs: list[float],
    ) -> None:
        for phase, logits, entropies, max_probs in (
            ("bid", output.masked_bid_logits, bid_entropies, bid_max_probs),
            ("play", output.masked_card_logits, play_entropies, play_max_probs),
        ):
            indices = [index for index, sample in enumerate(samples) if sample.phase == phase]
            if indices:
                distribution = Categorical(logits=logits[indices].float())
                entropies.extend(distribution.entropy().cpu().tolist())
                max_probs.extend(distribution.probs.max(dim=-1).values.cpu().tolist())

    def _validate_config(self) -> None:
        if not self.config.player_counts or not self.config.hand_sizes:
            raise ValueError("player_counts and hand_sizes must not be empty.")
        for spec in self.config.specs:
            spec.validate()
            if spec.num_players > self.config.model_config.max_players:
                raise ValueError("Training player count exceeds model capacity.")
            if spec.hand_size > self.config.model_config.max_hand_size:
                raise ValueError("Training hand size exceeds model capacity.")
        if self.config.rounds_per_configuration < 1:
            raise ValueError("rounds_per_configuration must be positive.")
        if self.config.games_per_player_seat < 1:
            raise ValueError("games_per_player_seat must be positive.")
        if self.config.training_mode not in {"round", "game"}:
            raise ValueError("training_mode must be 'round' or 'game'.")
        if not 0.0 <= self.config.round_gae_lambda <= 1.0:
            raise ValueError("round_gae_lambda must be in [0, 1].")
        if not 0.0 <= self.config.game_gae_lambda <= 1.0:
            raise ValueError("game_gae_lambda must be in [0, 1].")
        if self.config.gae_gamma <= 0.0:
            raise ValueError("gae_gamma must be positive.")
        if self.config.training_mode == "game":
            game_schedule = self.config.resolved_game_schedule
            if len(game_schedule) > self.config.model_config.max_rounds:
                raise ValueError("Game schedule exceeds model max_rounds.")
            if any(
                hand_size > self.config.model_config.max_hand_size
                for hand_size in game_schedule
            ):
                raise ValueError("Game schedule exceeds model max_hand_size.")
        if self.config.num_envs < 1 or self.config.minibatch_size < 1:
            raise ValueError("num_envs and minibatch_size must be positive.")
        if self.config.microbatch_size is not None and self.config.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive when set.")
        fractions = (
            self.config.self_play_fraction,
            self.config.heuristic_fraction,
            self.config.mixed_fraction,
            self.config.historical_fraction,
        )
        if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
            raise ValueError("Opponent-arm fractions must be in [0, 1].")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError("Opponent-arm fractions must sum to 1.")
        active_arms = sum(fraction > 0.0 for fraction in fractions)
        cell_repetitions = (
            self.config.games_per_player_seat
            if self.config.training_mode == "game"
            else self.config.rounds_per_configuration
        )
        if cell_repetitions < active_arms:
            raise ValueError(
                "Each balanced cell must cover every active opponent arm."
            )
        if self.config.historical_max_snapshots < 1:
            raise ValueError("historical_max_snapshots must be positive.")
        if self.config.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be one of: fp32, bf16, fp16.")
        if self.config.owner_capacity_coef < 0.0:
            raise ValueError("owner_capacity_coef must be nonnegative.")
        if self.config.model_config.owner_sinkhorn_iterations < 1:
            raise ValueError(
                "owner_sinkhorn_iterations must be positive."
            )


def _snapshot_observation(observation: Observation) -> Observation:
    """Copy the in-progress trick, the only container the env keeps mutating."""

    trick = observation.current_trick
    if trick is None:
        return observation
    return dataclass_replace(
        observation,
        current_trick=Trick(
            trick_index=trick.trick_index,
            leader=trick.leader,
            led_suit=trick.led_suit,
            plays=list(trick.plays),
            winner=trick.winner,
        ),
    )


def _current_policy_players(
    focal_player: int,
    opponent_policies: dict[int, ActionPolicy | None],
) -> frozenset[int]:
    """Every seat the current model controls; all of them are trainable."""

    return frozenset(
        {focal_player}
        | {
            player
            for player, policy in opponent_policies.items()
            if policy is None
        }
    )


def _trick_implied_relative_values(
    trick_probabilities: Tensor,
    final_bids: Tensor,
    active_player_mask: Tensor,
) -> Tensor:
    """Convert final-trick distributions into expected relative round score."""

    safe_bids = final_bids.clamp(
        min=0,
        max=trick_probabilities.shape[-1] - 1,
    )
    hit_probabilities = trick_probabilities.gather(
        -1,
        safe_bids.unsqueeze(-1),
    ).squeeze(-1)
    hit_scores = torch.where(
        safe_bids == 0,
        torch.full_like(hit_probabilities, 5.0),
        10.0 + safe_bids.float(),
    )
    expected_scores = (
        hit_probabilities
        * hit_scores
        * active_player_mask.float()
        * (final_bids >= 0).float()
    )
    opponent_count = (
        active_player_mask[:, 1:]
        .sum(dim=-1)
        .clamp_min(1)
    )
    return (
        expected_scores[:, 0]
        - expected_scores[:, 1:].sum(dim=-1)
        / opponent_count
    )


def _evenly_spaced_samples(samples: list[RolloutSample], max_samples: int) -> list[RolloutSample]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return list(samples)
    stride = len(samples) / max_samples
    return [samples[min(int(index * stride), len(samples) - 1)] for index in range(max_samples)]


def _load_checkpoint_payload(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(Path(path), map_location=device)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values]))


def _percentile_abs(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(abs(value) for value in values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def _mae(values: list[float], targets: list[float]) -> float:
    return _mean([abs(value - target) for value, target in zip(values, targets)])


def _mse(values: list[float], targets: list[float]) -> float:
    return _mean([(value - target) ** 2 for value, target in zip(values, targets)])


def _explained_variance(values: list[float], targets: list[float]) -> float:
    if not targets:
        return 0.0
    target_mean = _mean(targets)
    target_variance = _mean([(target - target_mean) ** 2 for target in targets])
    if target_variance <= 1e-12:
        return 0.0
    errors = [
        target - value
        for value, target in zip(values, targets)
    ]
    error_mean = _mean(errors)
    error_variance = _mean(
        [(error - error_mean) ** 2 for error in errors]
    )
    return 1.0 - error_variance / target_variance


def format_update_stats(stats: UpdateStats) -> str:
    values = {
        "samples": stats.samples,
        "configs": stats.configurations,
        "loss": stats.total_loss,
        "policy": stats.policy_loss,
        "value": stats.value_loss,
        "entropy": stats.entropy,
        "aux": stats.auxiliary_loss,
        "trick": stats.trick_loss,
        "owner": stats.owner_loss,
        "owner_ce": stats.owner_ce_loss,
        "owner_cap": stats.owner_capacity_loss,
        "kl": stats.approx_kl,
        "clip": stats.clip_fraction,
    }
    return " ".join(
        f"{key}={value:.4f}" if isinstance(value, float) and math.isfinite(value) else f"{key}={value}"
        for key, value in values.items()
    )
