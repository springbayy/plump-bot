"""Balanced schema-v4 PPO for round-local and full-game Plump policies."""

from __future__ import annotations

import copy
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from plump.cards import Card, make_deck
from plump.env import PlumpEnv
from plump.evaluation import DealBank, evaluate_policy
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
from plump.training.env_workers import (
    DecisionRequest,
    EnvWorkerPool,
    EpisodeAssignment,
    RoundResult,
)
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
    suit_presence_targets_relative as _suit_presence_targets_relative,
)


SamplePhase = Literal["bid", "play"]
OpponentArm = Literal[
    "self",
    "heuristic",
    "mixed",
    "historical",
    "explore_self",
    "explore_historical",
]
# Arms whose focal seat samples from the noised behavior policy while every
# opponent seat is frozen weight playing raw — the learner explores without
# ever optimizing against noisy play.
EXPLORE_ARMS: frozenset[str] = frozenset({"explore_self", "explore_historical"})
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
    # Optional per-value sampling weights aligned with player_counts /
    # hand_sizes (normalized internally; empty = uniform). When either is
    # set, round-mode collections draw cells by the joint product weight
    # instead of enumerating every cell equally; the total rounds per
    # collection stays len(specs) * rounds_per_configuration.
    player_count_weights: tuple[float, ...] = ()
    hand_size_weights: tuple[float, ...] = ()
    rounds_per_configuration: int = 8
    games_per_player_seat: int = 4
    num_envs: int = 32
    ppo_epochs: int = 4
    target_kl: float | None = None
    pipeline_rollouts: bool = False
    env_workers: int = 0
    event_length_buckets: tuple[int, ...] = ()
    batch_packing: Literal["torch", "numpy"] = "torch"
    lean_rollout_forward: bool = False
    minibatch_size: int = 256
    microbatch_size: int | None = None
    learning_rate: float = 3e-4
    ppo_clip_eps: float = 0.2
    value_coef: float = 0.5
    oracle_critic: bool = False
    oracle_value_coef: float = 0.5
    mmd_enabled: bool = False
    mmd_coef: float = 0.05
    mmd_magnet_decay: float = 0.995
    entropy_coef: float = 0.01
    trick_coef: float = 0.1
    owner_coef: float = 0.05
    # Iterations after the owner head first becomes active during which its
    # gradients are cut off from the trunk (head-only learning), so a
    # randomly initialized head cannot disturb an already-trained policy.
    owner_warmup_iterations: int = 0
    owner_capacity_coef: float = 0.1
    suit_coef: float = 0.1
    max_grad_norm: float = 1.0
    position_baseline_decay: float = 0.98
    self_play_fraction: float = 0.3
    heuristic_fraction: float = 0.3
    mixed_fraction: float = 0.3
    historical_fraction: float = 0.1
    # Exploration arms: the focal seat samples from the tempered/eps behavior
    # policy while every opponent is frozen weight playing raw (explore_self
    # freezes the current weights, explore_historical draws a league
    # snapshot), so diverse trajectories never come from optimizing against
    # noisy opponents. Only the focal seat produces training samples.
    explore_self_fraction: float = 0.0
    explore_historical_fraction: float = 0.0
    # Behavior-policy exploration: with probability eps a rollout action is
    # drawn uniformly over LEGAL actions; old_logprob records the mixture
    # probability so PPO importance ratios stay correct.
    explore_eps_bid: float = 0.0
    explore_eps_play: float = 0.0
    # Per-arm (bid_eps, play_eps) overrides; arms not listed fall back to the
    # global values above. Lets exploration concentrate in rounds whose
    # opponents are frozen, so no learning seat has to model the noise.
    explore_eps_by_arm: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Trajectory-diversity exploration: a random explore_temperature_fraction
    # of rounds in explore_temperature_arms sample the CURRENT policy's
    # actions from softmax(logits / T) instead of the raw distribution
    # (bids and plays get separate temperatures). Pure behavior-policy
    # change: old_logprob records the tempered (and eps-mixed) probability,
    # so PPO ratios stay importance-correct and the update is standard.
    # Frozen opponent seats are unaffected (they have their own eps).
    explore_temperature_fraction: float = 0.0
    explore_temperature_bid: float = 1.0
    explore_temperature_play: float = 1.0
    explore_temperature_arms: tuple[str, ...] = ("self", "mixed")
    # At most one uniform-random action per explore round: with this
    # probability the focal seat gets exactly one decision — uniformly placed
    # among its 1+hand_size decisions — sampled uniformly over legal actions.
    # Every other decision stays tempered-only, so a round is one deliberate
    # deviation inside otherwise-plausible play, never a random walk.
    explore_uniform_round_probability: float = 0.0
    # Scale explore noise down on longer rounds: the effective temperature
    # and the uniform-round probability shrink by (min_hand+1)/(hand+1), so
    # total per-round distortion stays roughly constant instead of
    # compounding over more decisions.
    explore_noise_hand_normalized: bool = False
    historical_checkpoint_paths: tuple[str, ...] = ()
    historical_max_snapshots: int = 4
    league_temperature: float = 2.0
    league_reward_decay: float = 0.95
    # "uniform" draws league opponents uniformly from the current pool and
    # disables all payoff bookkeeping (admission fills, refreshes, meta
    # mixtures) — intended for pools resampled from the whole checkpoint
    # history, where per-member difficulty estimates never converge anyway.
    league_meta_solver: Literal["softmax_ema", "regret_matching", "uniform"] = "regret_matching"
    batched_league_sampling: bool = False
    league_probe_fraction: float = 0.10
    league_eval_every: int = 50
    league_eval_deals_per_configuration: int = 2
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

    def spec_round_quotas(self) -> dict[RoundSpec, int] | None:
        """Rounds per cell under the joint sampling weights, or None if uniform.

        Largest-remainder allocation of rounds_per_batch across cells with
        weight P(players) * P(hand_size), so every collection plays the exact
        same deterministic cell mix (no sampling noise iteration-to-iteration).
        """

        if not self.player_count_weights and not self.hand_size_weights:
            return None
        player_weights = dict(
            zip(self.player_counts, self.player_count_weights)
        ) or {count: 1.0 for count in self.player_counts}
        hand_weights = dict(
            zip(self.hand_sizes, self.hand_size_weights)
        ) or {size: 1.0 for size in self.hand_sizes}
        raw = {
            spec: player_weights[spec.num_players] * hand_weights[spec.hand_size]
            for spec in self.specs
        }
        total = self.rounds_per_batch
        scale = total / sum(raw.values())
        quotas = {spec: int(math.floor(value * scale)) for spec, value in raw.items()}
        remainder = total - sum(quotas.values())
        ranked = sorted(
            raw,
            key=lambda spec: (raw[spec] * scale - quotas[spec], raw[spec]),
            reverse=True,
        )
        for spec in ranked[:remainder]:
            quotas[spec] += 1
        return quotas

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
    suit_presence_targets: list[list[int]] | None = None
    round_weight: float = 0.0
    advantage_target: float | None = None
    observation: object | None = None
    ppo_policy_enabled: bool = True
    search_target_probabilities: list[float] | None = None
    trick_position: int = -1
    round_progress: float = 0.0
    # Logprob under the raw policy (old_logprob is the behavior mixture when
    # explore_eps is active); None means they coincide.
    old_policy_logprob: float | None = None


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
    opponent_snapshot_id: str | None = None


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
    historical_relative_reward: float
    explore_self_relative_reward: float
    explore_historical_relative_reward: float
    self_play_rounds: int
    heuristic_rounds: int
    mixed_rounds: int
    historical_rounds: int
    explore_self_rounds: int
    explore_historical_rounds: int
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
    # Focal bid-hit rate split by table size and by hand-size bucket
    # ("3_5" | "6_8" | "9_10"); groups absent from the batch are omitted.
    bid_hit_rate_by_players: dict[int, float] = field(default_factory=dict)
    bid_hit_rate_by_hand_bucket: dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionStats:
    samples: int
    value_mae: float
    value_mse: float
    value_explained_variance: float
    oracle_value_explained_variance: float
    bid_value_explained_variance: float
    play_value_explained_variance: float
    trick_implied_value_explained_variance: float
    bid_trick_implied_value_explained_variance: float
    play_trick_implied_value_explained_variance: float
    trick_count_accuracy: float
    trick_count_true_prob: float
    # Trick-count accuracy by decision stage (bid-time, then play thirds)
    # and by table size / hand-size bucket, matching the suit-stage splits.
    trick_count_accuracy_bidtime: float
    trick_count_accuracy_early: float
    trick_count_accuracy_mid: float
    trick_count_accuracy_late: float
    trick_count_accuracy_by_players: dict[int, float]
    trick_count_accuracy_by_hand_bucket: dict[str, float]
    suit_presence_accuracy: float
    suit_presence_brier: float
    suit_presence_loss_early: float
    suit_presence_loss_mid: float
    suit_presence_loss_late: float
    suit_presence_accuracy_early: float
    suit_presence_accuracy_mid: float
    suit_presence_accuracy_late: float
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
    oracle_value_loss: float
    magnet_kl: float
    entropy: float
    auxiliary_loss: float
    trick_loss: float
    owner_loss: float
    owner_ce_loss: float
    owner_capacity_loss: float
    suit_presence_loss: float
    approx_kl: float
    clip_fraction: float
    skipped_steps: int
    epochs_run: int
    samples: int
    configurations: int


@dataclass
class CollectionStats:
    total_sec: float = 0.0
    current_forward_sec: float = 0.0
    historical_forward_sec: float = 0.0
    env_step_sec: float = 0.0
    finalize_sec: float = 0.0
    current_forward_calls: int = 0
    current_forward_rows: int = 0
    historical_forward_calls: int = 0
    historical_forward_rows: int = 0
    historical_policy_count: int = 0
    valid_event_tokens: int = 0
    processed_event_tokens: int = 0
    peak_device_memory_bytes: int = 0


@dataclass
class LeagueSnapshot:
    """A frozen historical opponent with its difficulty bookkeeping key."""

    snapshot_id: str
    path: str
    policy: ActionPolicy


# The heuristic anchor participates in the league meta-game alongside the
# frozen snapshots, but is never sampled through the historical arm (it has
# its own dedicated opponent arm).
LEAGUE_HEURISTIC_MEMBER_ID = "heuristic"
LEAGUE_CURRENT_MEMBER_ID = "current"


def solve_meta_mixture(
    member_ids: list[str],
    payoffs: dict[tuple[str, str], float],
    *,
    iterations: int = 500,
) -> dict[str, float]:
    """Regret matching over the one-population meta-game.

    ``payoffs[(a, b)]`` is the focal relative reward of member ``a`` against a
    table of member ``b``; missing cells are treated as neutral (0.0). The
    average strategy approximates a coarse correlated equilibrium of the
    meta-game, so non-transitive cycles (A beats B beats C beats A) spread
    mass instead of collapsing onto one member the way a scalar difficulty
    score does.
    """

    count = len(member_ids)
    if count == 0:
        return {}
    if count == 1:
        return {member_ids[0]: 1.0}
    matrix = [
        [payoffs.get((row_id, column_id), 0.0) for column_id in member_ids]
        for row_id in member_ids
    ]
    cumulative_regret = [0.0] * count
    cumulative_strategy = [0.0] * count
    for _ in range(iterations):
        positive = [max(regret, 0.0) for regret in cumulative_regret]
        total_positive = sum(positive)
        if total_positive > 0.0:
            strategy = [value / total_positive for value in positive]
        else:
            strategy = [1.0 / count] * count
        utilities = [
            sum(matrix[row][column] * strategy[column] for column in range(count))
            for row in range(count)
        ]
        expected = sum(strategy[row] * utilities[row] for row in range(count))
        for row in range(count):
            cumulative_regret[row] += utilities[row] - expected
            cumulative_strategy[row] += strategy[row]
    total_strategy = sum(cumulative_strategy)
    return {
        member_id: cumulative_strategy[index] / total_strategy
        for index, member_id in enumerate(member_ids)
    }


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
    opponent_snapshot_id: str | None = None
    explore_tempered: bool = False
    # Focal decision index (0 = bid) that samples uniformly over legal
    # actions this round; None = no injected action.
    explore_uniform_index: int | None = None
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
        # Update owns its own RNG so a pipelined collector thread and the
        # minibatch shuffle never interleave draws on one generator.
        self._update_rng = random.Random(self.config.seed ^ 0x5EED)
        self.last_collect_sec = 0.0
        self.last_collection_stats = CollectionStats()
        self._active_collection_stats: CollectionStats | None = None
        self._active_historical_policy_ids: set[int] | None = None
        self._iteration_exploit_snapshot: LeagueSnapshot | None = None
        self._iteration_probe_snapshot: LeagueSnapshot | None = None
        self.position_baseline = PositionBaseline(self.config.position_baseline_decay)
        # First iteration at which owner_coef was active; drives the trunk-
        # detached warmup and persists across checkpoint save/resume.
        self.owner_active_since: int | None = None
        self._current_iteration = 0
        self.heuristic_policy = HeuristicPolicy()
        self.historical_snapshots: list[LeagueSnapshot] = []
        self.league_reward_ema: dict[str, float] = {}
        self.league_payoffs: dict[tuple[str, str], float] = {}
        self.league_meta_mixture: dict[str, float] = {}
        self._league_deal_bank: DealBank | None = None
        for path in self.config.historical_checkpoint_paths:
            self.add_historical_checkpoint(path)
        self.magnet_model: PlumpTransformerModel | None = None
        if self.config.mmd_enabled:
            self._reset_magnet_model()
        self.rollout_model: PlumpTransformerModel | None = None
        if self.config.pipeline_rollouts:
            self._reset_rollout_model()
        # Optional multiprocess env stepping; owned by the caller (train loop).
        self.env_pool: EnvWorkerPool | None = None

    @property
    def historical_policies(self) -> list[ActionPolicy]:
        return [snapshot.policy for snapshot in self.historical_snapshots]

    def _reset_magnet_model(self) -> None:
        self.magnet_model = copy.deepcopy(self.model)
        self.magnet_model.eval()
        self.magnet_model.requires_grad_(False)

    def _update_magnet_model(self) -> None:
        if self.magnet_model is None:
            return
        decay = self.config.mmd_magnet_decay
        with torch.no_grad():
            for magnet_param, param in zip(
                self.magnet_model.parameters(),
                self.model.parameters(),
            ):
                magnet_param.mul_(decay).add_(param, alpha=1.0 - decay)

    def _reset_rollout_model(self) -> None:
        self.rollout_model = copy.deepcopy(self.model)
        self.rollout_model.eval()
        self.rollout_model.requires_grad_(False)

    def sync_rollout_model(self) -> None:
        """Copy live weights into the frozen model used for rollout collection.

        In pipelined mode the collector thread samples from this snapshot while
        `update()` trains `self.model`; call this only at the pipeline sync
        point (no collection in flight).
        """

        if self.rollout_model is None:
            return
        self.rollout_model.load_state_dict(self.model.state_dict())

    def _collection_model(self) -> PlumpTransformerModel:
        return self.rollout_model if self.rollout_model is not None else self.model

    def balanced_round_specs(self) -> list[RoundSpec]:
        quotas = self.config.spec_round_quotas()
        if quotas is None:
            specs = [
                spec
                for spec in self.config.specs
                for _ in range(self.config.rounds_per_configuration)
            ]
        else:
            specs = [spec for spec, quota in quotas.items() for _ in range(quota)]
        self.rng.shuffle(specs)
        return specs

    def balanced_round_schedule(self) -> list[tuple[RoundSpec, OpponentArm]]:
        fractions = self._effective_arm_fractions()
        quotas = self.config.spec_round_quotas()
        if quotas is None:
            schedule = [
                (spec, arm)
                for spec in self.config.specs
                for arm in self._allocate_opponent_arms(
                    self.config.rounds_per_configuration,
                    fractions,
                )
            ]
        else:
            # Weighted cells can have quotas smaller than the number of arms,
            # where per-cell largest-remainder allocation would silently drop
            # low-fraction arms. Allocate arms once over the whole batch
            # (exact global mix) and pair them with shuffled cells instead.
            specs = [spec for spec, quota in quotas.items() for _ in range(quota)]
            self.rng.shuffle(specs)
            arms = self._allocate_opponent_arms(len(specs), fractions)
            schedule = list(zip(specs, arms))
        self.rng.shuffle(schedule)
        return schedule

    def collect_rollouts(self, *, iteration: int = 1) -> RolloutBuffer:
        started = time.perf_counter()
        self._current_iteration = iteration
        stats = CollectionStats()
        self._active_collection_stats = stats
        self._active_historical_policy_ids = set()
        self._prepare_historical_iteration_sampling()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            if self.config.training_mode == "game":
                return self._collect_game_rollouts(iteration=iteration)
            if self.env_pool is not None:
                return self._collect_round_rollouts_workers(iteration=iteration)
            return self._collect_round_rollouts(iteration=iteration)
        finally:
            self.last_collect_sec = time.perf_counter() - started
            stats.total_sec = self.last_collect_sec
            if self.device.type == "cuda":
                stats.peak_device_memory_bytes = int(
                    torch.cuda.max_memory_allocated(self.device)
                )
            elif self.device.type == "mps":
                stats.peak_device_memory_bytes = int(
                    torch.mps.driver_allocated_memory()
                )
            self.last_collection_stats = stats
            self._active_collection_stats = None
            self._active_historical_policy_ids = None

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

        self._collection_model().eval()
        with torch.inference_mode():
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
                    step_started = time.perf_counter()
                    episode.env.step(action)
                    if self._active_collection_stats is not None:
                        self._active_collection_stats.env_step_sec += (
                            time.perf_counter() - step_started
                        )

                    if episode.env.is_done():
                        finalize_started = time.perf_counter()
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
                        if self._active_collection_stats is not None:
                            self._active_collection_stats.finalize_sec += (
                                time.perf_counter() - finalize_started
                            )
                    else:
                        next_active.append(episode)
                active = next_active

        self._assign_round_weights(buffer)
        self.position_baseline.update_many(pending_baseline_updates)
        self._update_league_ema(buffer.round_outcomes)
        if len(buffer.round_outcomes) != self.config.rounds_per_batch:
            raise RuntimeError("Balanced collection did not complete the requested schedule.")
        return buffer

    def _collect_round_rollouts_workers(self, *, iteration: int = 1) -> RolloutBuffer:
        """Round collection with env stepping/encoding in worker processes.

        Workers own the envs and heuristic seats; every model decision comes
        back as a DecisionRequest that is batched per policy and answered
        centrally, so league draws, baseline reads, and all GPU inference stay
        in this process with the same semantics as the in-process path.
        """

        del iteration
        buffer = RolloutBuffer()
        schedule = self.balanced_round_schedule()
        snapshot_id_by_policy = {
            id(snapshot.policy): snapshot.snapshot_id
            for snapshot in self.historical_snapshots
        }
        snapshot_policy_by_id = {
            snapshot.snapshot_id: snapshot.policy
            for snapshot in self.historical_snapshots
        }
        assignments: list[EpisodeAssignment] = []
        episode_meta: dict[int, tuple[RoundSpec, OpponentArm, int, frozenset[int], str | None]] = {}
        for episode_id, (spec, arm) in enumerate(schedule):
            # Same rng call order as _new_active_episode so deals match the
            # in-process path for a given seed.
            start_player = self.rng.randrange(spec.num_players)
            env_seed = self.rng.randrange(2**31)
            focal_player = episode_id % spec.num_players
            opponent_policies, snapshot_id = self._opponent_policies(
                spec.num_players,
                focal_player,
                arm,
            )
            explore_tempered = self._draw_explore_tempered(arm)
            explore_uniform_index = self._draw_explore_uniform_index(
                arm,
                spec.hand_size,
            )
            seat_policy_refs: dict[int, str] = {}
            for player, policy in opponent_policies.items():
                if policy is None:
                    seat_policy_refs[player] = (
                        "current-frozen" if arm in EXPLORE_ARMS else "current"
                    )
                elif policy is self.heuristic_policy:
                    seat_policy_refs[player] = "heuristic"
                else:
                    reference = snapshot_id_by_policy.get(id(policy))
                    if reference is None or not isinstance(policy, ModelPolicy):
                        raise RuntimeError(
                            "env workers require league opponents to be "
                            "ModelPolicy snapshots."
                        )
                    seat_policy_refs[player] = reference
            trainable_players = frozenset(
                {focal_player}
                | {
                    player
                    for player, reference in seat_policy_refs.items()
                    if reference == "current"
                }
            )
            assignments.append(
                EpisodeAssignment(
                    episode_id=episode_id,
                    num_players=spec.num_players,
                    hand_size=spec.hand_size,
                    opponent_arm=arm,
                    focal_player=focal_player,
                    start_player=start_player,
                    env_seed=env_seed,
                    seat_policy_refs=seat_policy_refs,
                    opponent_snapshot_id=snapshot_id,
                    explore_tempered=explore_tempered,
                    explore_uniform_index=explore_uniform_index,
                )
            )
            episode_meta[episode_id] = (
                spec,
                arm,
                focal_player,
                trainable_players,
                snapshot_id,
            )

        samples_by_episode: dict[int, list[RolloutSample]] = defaultdict(list)
        pending_baseline_updates: list[tuple[PositionKey, float]] = []
        self._collection_model().eval()
        self.env_pool.begin_iteration(assignments)
        while True:
            requests, results, done = self.env_pool.gather_wave()
            for completion in results:
                spec, arm, focal_player, trainable_players, snapshot_id = (
                    episode_meta[completion.episode_id]
                )
                episode_samples = samples_by_episode.pop(completion.episode_id, [])
                pending_baseline_updates.extend(
                    self._finalize_round_result(
                        samples=episode_samples,
                        trainable_players=trainable_players,
                        num_players=spec.num_players,
                        result=completion.result,
                    )
                )
                buffer.extend(episode_samples)
                buffer.round_outcomes.append(
                    self._round_outcome_from_result(
                        episode_id=completion.episode_id,
                        num_players=spec.num_players,
                        opponent_arm=arm,
                        focal_player=focal_player,
                        opponent_snapshot_id=snapshot_id,
                        result=completion.result,
                    )
                )
            if done:
                break
            actions: dict[int, BidAction | PlayCardAction] = {}
            current_requests = [
                request
                for request in requests
                if request.policy_ref in ("current", "current-frozen")
            ]
            if current_requests:
                rows = self._forward_decision_rows(current_requests)
                for request, (sample, action) in zip(current_requests, rows):
                    trainable_players = episode_meta[request.episode_id][3]
                    if request.player in trainable_players:
                        samples_by_episode[request.episode_id].append(sample)
                    actions[request.episode_id] = action
            frozen_requests: dict[str, list[DecisionRequest]] = defaultdict(list)
            for request in requests:
                if request.policy_ref not in ("current", "current-frozen"):
                    frozen_requests[request.policy_ref].append(request)
            for reference, rows in frozen_requests.items():
                policy = snapshot_policy_by_id[reference]
                selected = policy.act_encoded(
                    [request.encoded for request in rows],
                    phases=[
                        Phase.BIDDING if request.phase == "bid" else Phase.PLAYING
                        for request in rows
                    ],
                    players=[request.player for request in rows],
                    rngs=[self.rng] * len(rows),
                )
                for request, action in zip(rows, selected):
                    actions[request.episode_id] = action
            self.env_pool.send_actions(actions)

        self._assign_round_weights(buffer)
        self.position_baseline.update_many(pending_baseline_updates)
        self._update_league_ema(buffer.round_outcomes)
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

        self._collection_model().eval()
        with torch.inference_mode():
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
                    step_started = time.perf_counter()
                    result = episode.env.step(action)
                    if self._active_collection_stats is not None:
                        self._active_collection_stats.env_step_sec += (
                            time.perf_counter() - step_started
                        )

                    if result.info.get("round_ended"):
                        completed = episode.env.state.rounds[episode.completed_rounds]
                        buffer.round_outcomes.append(
                            self._round_outcome_for_state(episode, completed)
                        )
                        episode.completed_rounds += 1

                    if episode.env.is_done():
                        finalize_started = time.perf_counter()
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
                        if self._active_collection_stats is not None:
                            self._active_collection_stats.finalize_sec += (
                                time.perf_counter() - finalize_started
                            )
                    else:
                        next_active.append(episode)
                active = next_active

        self._assign_game_weights(buffer)
        self._update_league_ema(buffer.round_outcomes)
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
            event_length_buckets=self.config.event_length_buckets,
            packing=self.config.batch_packing,
        )
        old_logprobs = torch.tensor(
            [sample.old_logprob for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        # The KL gate must track movement from the collecting POLICY, not from
        # the exploration mixture (whose divergence is large by construction).
        old_policy_logprobs = torch.tensor(
            [
                sample.old_policy_logprob
                if sample.old_policy_logprob is not None
                else sample.old_logprob
                for sample in samples
            ],
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
        suit_targets = torch.tensor(
            [sample.suit_presence_targets for sample in samples],
            dtype=torch.long,
            device=self.device,
        )
        train_owner = self.config.owner_coef > 0.0
        if train_owner and self.owner_active_since is None:
            self.owner_active_since = self._current_iteration
        detach_owner_trunk = train_owner and (
            self._current_iteration
            < self.owner_active_since + self.config.owner_warmup_iterations
        )
        skipped_steps = 0
        epochs_run = 0

        metric_values: dict[str, list[Tensor]] = defaultdict(list)
        indices = list(range(len(samples)))
        previous_training = self.model.training
        self.model.eval()
        try:
            for epoch_index in range(self.config.ppo_epochs):
                epoch_kl_values: list[Tensor] = []
                self._update_rng.shuffle(indices)
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
                    logical_suit_targets = suit_targets.index_select(0, logical_tensor)
                    trick_label_count = (logical_trick_targets != -100).sum().clamp_min(1)
                    owner_label_count = (logical_owner_targets != -100).sum().clamp_min(1)
                    suit_label_count = (logical_suit_targets != -100).sum().clamp_min(1)
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
                        mb_old_policy_logprobs = old_policy_logprobs.index_select(0, selected_tensor)
                        weights = round_weights.index_select(0, selected_tensor)
                        mb_policy_enabled = policy_enabled.index_select(0, selected_tensor)
                        batch = index_model_batch(staged_batch, selected_tensor)
                        mb_owner_targets = owner_targets.index_select(0, selected_tensor)
                        with model_autocast(self.device, self.config.precision):
                            output = self.model(
                                batch,
                                need_owner=train_owner,
                                detach_owner_trunk=detach_owner_trunk,
                                privileged_owner_targets=(
                                    mb_owner_targets
                                    if self.config.oracle_critic
                                    else None
                                ),
                            )
                        new_logprobs, entropy_by_sample = self._logprobs_and_entropy(output, mb_samples)
                        if self.magnet_model is not None:
                            with torch.no_grad(), model_autocast(self.device, self.config.precision):
                                magnet_output = (
                                    self.magnet_model.forward_policy(batch)
                                    if self.config.lean_rollout_forward
                                    else self.magnet_model(batch, need_owner=False)
                                )
                            magnet_kl_by_sample = self._magnet_kl_terms(
                                output,
                                magnet_output,
                                mb_samples,
                            )
                        else:
                            magnet_kl_by_sample = None
                        # Off-policy PPO: clip the policy-to-policy ratio rho
                        # (starts at 1), and correct for the exploration
                        # behavior policy with the fixed importance weight
                        # w = pi_old/b OUTSIDE the min. Clipping pi_new/b
                        # instead centers the trust region on the behavior
                        # mixture: under strong tempering a favored action
                        # starts past the clip and can only ever be pushed
                        # down (and rare boosted actions only up) — a
                        # systematic flattening force that measurably
                        # degraded the policy. Both surrogate terms are
                        # computed in overflow-safe forms: w*rho IS the
                        # behavior ratio pi_new/b (finite because b actually
                        # sampled the action), and the clipped term bounds
                        # the log before exponentiating so w=0, rho=inf rare
                        # actions never produce 0*inf.
                        policy_log_ratio_grad = new_logprobs - mb_old_policy_logprobs
                        behavior_ratio = torch.exp(new_logprobs - mb_old_logprobs)
                        behavior_weight = torch.exp(
                            mb_old_policy_logprobs - mb_old_logprobs
                        )
                        clipped_ratio = torch.clamp(
                            torch.exp(policy_log_ratio_grad.clamp(-20.0, 20.0)),
                            1.0 - self.config.ppo_clip_eps,
                            1.0 + self.config.ppo_clip_eps,
                        )
                        policy_terms = -torch.min(
                            behavior_ratio * mb_advantages,
                            behavior_weight * clipped_ratio * mb_advantages,
                        )
                        # The behavior-policy ratio pi_new/b, used by the
                        # importance-weighted KL identity below.
                        ratio = behavior_ratio
                        # Uniform over configurations, mean over rounds, sum over decisions.
                        objective_scale = len(samples) * weights
                        policy_scale = objective_scale * mb_policy_enabled
                        policy_loss = (policy_scale * policy_terms).sum() / logical_count
                        entropy = (policy_scale * entropy_by_sample).sum() / logical_count
                        if magnet_kl_by_sample is not None:
                            magnet_kl = (
                                (policy_scale * magnet_kl_by_sample).sum() / logical_count
                            )
                        else:
                            magnet_kl = torch.zeros_like(policy_loss)

                        mb_residual_targets = residual_targets.index_select(0, selected_tensor)
                        value_terms = F.smooth_l1_loss(
                            output.value.squeeze(-1).float(),
                            mb_residual_targets,
                            reduction="none",
                        )
                        value_loss = (objective_scale * value_terms).sum() / logical_count
                        if output.oracle_value is not None:
                            oracle_terms = F.smooth_l1_loss(
                                output.oracle_value.squeeze(-1).float(),
                                mb_residual_targets,
                                reduction="none",
                            )
                            oracle_value_loss = (
                                (objective_scale * oracle_terms).sum() / logical_count
                            )
                        else:
                            oracle_value_loss = torch.zeros_like(value_loss)
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
                        if train_owner:
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
                        else:
                            owner_ce_loss = torch.zeros_like(trick_loss)
                            owner_capacity_loss = torch.zeros_like(trick_loss)
                            owner_loss = torch.zeros_like(trick_loss)
                        if output.suit_presence_logits is not None:
                            mb_suit_targets = suit_targets.index_select(0, selected_tensor)
                            active_suit_targets = mb_suit_targets != -100
                            suit_bce = F.binary_cross_entropy_with_logits(
                                output.suit_presence_logits.float(),
                                mb_suit_targets.clamp_min(0).float(),
                                reduction="none",
                            )
                            suit_presence_loss = (
                                suit_bce[active_suit_targets].sum() / suit_label_count
                            )
                        else:
                            suit_presence_loss = torch.zeros_like(trick_loss)
                        auxiliary_loss = (
                            self.config.trick_coef * trick_loss
                            + self.config.owner_coef * owner_loss
                            + self.config.suit_coef * suit_presence_loss
                        )
                        total_loss = (
                            policy_loss
                            + self.config.value_coef * value_loss
                            + self.config.oracle_value_coef * oracle_value_loss
                            + self.config.mmd_coef * magnet_kl
                            - self.config.entropy_coef * entropy
                            + auxiliary_loss
                        )
                        total_loss.backward()

                        # KL/clip are summed here and normalized by the total
                        # enabled count after the microbatch loop; summing
                        # per-microbatch means would inflate them by the
                        # number of gradient-accumulation microbatches.
                        with torch.no_grad():
                            policy_log_ratio = new_logprobs - mb_old_policy_logprobs
                            # Actions were sampled from the epsilon-mixture
                            # behavior policy. Reweight its samples back to
                            # the collecting policy before estimating the
                            # policy-to-policy KL used by the early-stop gate.
                            policy_sampling_weight = torch.exp(
                                mb_old_policy_logprobs - mb_old_logprobs
                            )
                            # Algebraically this is
                            #   w * (exp(policy_log_ratio) - 1 - policy_log_ratio)
                            # but w*exp(policy_log_ratio) is exactly the PPO
                            # behavior ratio. Using that identity avoids 0*inf
                            # when uniform exploration selects an action whose
                            # collecting-policy probability is vanishingly small.
                            importance_weighted_kl = ratio - (
                                policy_sampling_weight
                                * (1.0 + policy_log_ratio)
                            )
                            approx_kl = (
                                mb_policy_enabled
                                * importance_weighted_kl
                            ).sum()
                            clip_fraction = (
                                mb_policy_enabled
                                * (
                                    (
                                        policy_log_ratio_grad
                                        > math.log1p(self.config.ppo_clip_eps)
                                    )
                                    | (
                                        policy_log_ratio_grad
                                        < math.log(1.0 - self.config.ppo_clip_eps)
                                    )
                                ).float()
                            ).sum()
                        values = {
                            "enabled_count": mb_policy_enabled.sum(),
                            "total_loss": total_loss,
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                            "oracle_value_loss": oracle_value_loss,
                            "magnet_kl": magnet_kl,
                            "entropy": entropy,
                            "auxiliary_loss": auxiliary_loss,
                            "trick_loss": trick_loss,
                            "owner_loss": owner_loss,
                            "owner_ce_loss": owner_ce_loss,
                            "owner_capacity_loss": owner_capacity_loss,
                            "suit_presence_loss": suit_presence_loss,
                            "approx_kl": approx_kl,
                            "clip_fraction": clip_fraction,
                        }
                        for key, value in values.items():
                            detached = value.detach()
                            step_values[key] = step_values.get(key, detached.new_zeros(())) + detached

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    # A single non-finite gradient would poison the weights
                    # (clipping scales every grad by the NaN norm), so drop
                    # the step instead of applying it.
                    if bool(torch.isfinite(grad_norm)):
                        self.optimizer.step()
                    else:
                        skipped_steps += 1

                    enabled_total = step_values.pop("enabled_count").clamp_min(1.0)
                    step_values["approx_kl"] = step_values["approx_kl"] / enabled_total
                    step_values["clip_fraction"] = step_values["clip_fraction"] / enabled_total
                    epoch_kl_values.append(step_values["approx_kl"])
                    for key, value in step_values.items():
                        metric_values[key].append(value)

                epochs_run = epoch_index + 1
                if (
                    self.config.target_kl is not None
                    and epoch_kl_values
                    and float(torch.stack(epoch_kl_values).mean()) > self.config.target_kl
                ):
                    break
        finally:
            if previous_training:
                self.model.train()

        totals = {
            key: float(torch.stack(values).mean().cpu())
            for key, values in metric_values.items()
        }
        self._update_magnet_model()
        return UpdateStats(
            total_loss=totals["total_loss"],
            policy_loss=totals["policy_loss"],
            value_loss=totals["value_loss"],
            oracle_value_loss=totals["oracle_value_loss"],
            magnet_kl=totals["magnet_kl"],
            entropy=totals["entropy"],
            auxiliary_loss=totals["auxiliary_loss"],
            trick_loss=totals["trick_loss"],
            owner_loss=totals["owner_loss"],
            owner_ce_loss=totals["owner_ce_loss"],
            owner_capacity_loss=totals["owner_capacity_loss"],
            suit_presence_loss=totals["suit_presence_loss"],
            approx_kl=totals["approx_kl"],
            clip_fraction=totals["clip_fraction"],
            skipped_steps=skipped_steps,
            epochs_run=epochs_run,
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
        # Quality metrics (bid hit, entropy) are measured on CLEAN arms only:
        # explore-arm rounds have a noised focal seat visiting noise-reached
        # states, which would depress the levels without meaning anything.
        # Explore arms keep only their reward-trend and round-count fields.
        clean_samples = [
            sample
            for sample in samples
            if sample.opponent_arm not in EXPLORE_ARMS
        ] or samples
        clean_outcomes = [
            outcome
            for outcome in buffer.round_outcomes
            if outcome.opponent_arm not in EXPLORE_ARMS
        ] or buffer.round_outcomes
        bid_entropies = [
            sample.old_entropy
            for sample in clean_samples
            if sample.phase == "bid"
        ]
        play_entropies = [
            sample.old_entropy
            for sample in clean_samples
            if sample.phase == "play"
        ]
        hit_count = sum(outcome.bid_hit_count for outcome in clean_outcomes)
        player_count = sum(outcome.bid_player_count for outcome in clean_outcomes)
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

        def arm_reward(arm: OpponentArm) -> float:
            return _mean(
                [
                    float(outcome.focal_reward)
                    for outcome in buffer.round_outcomes
                    if outcome.opponent_arm == arm
                    and outcome.focal_reward is not None
                ]
            )

        def arm_rounds(arm: OpponentArm) -> int:
            return sum(
                outcome.opponent_arm == arm
                for outcome in buffer.round_outcomes
            )

        hits_by_players: dict[int, list[float]] = defaultdict(list)
        hits_by_bucket: dict[str, list[float]] = defaultdict(list)
        for outcome in clean_outcomes:
            hits_by_players[outcome.spec.num_players].append(
                float(outcome.focal_bid_hit)
            )
            hits_by_bucket[_hand_bucket_label(outcome.spec.hand_size)].append(
                float(outcome.focal_bid_hit)
            )
        return RolloutStats(
            rounds=len(buffer.round_outcomes),
            configurations=len({outcome.spec for outcome in buffer.round_outcomes}),
            samples=len(samples),
            bid_samples=sum(sample.phase == "bid" for sample in samples),
            play_samples=sum(sample.phase == "play" for sample in samples),
            bid_hit_rate=_mean(
                [float(outcome.focal_bid_hit) for outcome in clean_outcomes]
            ),
            bid_abs_error_mean=_mean(
                [outcome.focal_bid_abs_error for outcome in clean_outcomes]
            ),
            all_player_bid_hit_rate=hit_count / max(player_count, 1),
            all_player_bid_abs_error_mean=_mean(
                [outcome.bid_abs_error_mean for outcome in clean_outcomes]
            ),
            heuristic_relative_reward=arm_reward("heuristic"),
            historical_relative_reward=arm_reward("historical"),
            explore_self_relative_reward=arm_reward("explore_self"),
            explore_historical_relative_reward=arm_reward("explore_historical"),
            self_play_rounds=arm_rounds("self"),
            heuristic_rounds=arm_rounds("heuristic"),
            mixed_rounds=arm_rounds("mixed"),
            historical_rounds=arm_rounds("historical"),
            explore_self_rounds=arm_rounds("explore_self"),
            explore_historical_rounds=arm_rounds("explore_historical"),
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
            bid_hit_rate_by_players={
                players: _mean(hits)
                for players, hits in sorted(hits_by_players.items())
            },
            bid_hit_rate_by_hand_bucket={
                bucket: _mean(hits)
                for bucket, hits in sorted(hits_by_bucket.items())
            },
        )

    def compute_prediction_stats(
        self,
        buffer: RolloutBuffer,
        *,
        max_samples: int = 2048,
        minibatch_size: int | None = None,
    ) -> PredictionStats:
        # Diagnostics measure the raw policy and belief heads on CLEAN-play
        # states only; explore-arm states are reached through deliberate
        # noise, and mixing them in would shift every pred_* level with the
        # noise schedule instead of with learning.
        clean_samples = [
            sample
            for sample in buffer.ready_samples()
            if sample.opponent_arm not in EXPLORE_ARMS
        ] or buffer.ready_samples()
        samples = _evenly_spaced_samples(clean_samples, max_samples)
        minibatch_size = minibatch_size or self.config.minibatch_size
        values: list[float] = []
        oracle_values: list[float] = []
        targets: list[float] = []
        bid_values: list[float] = []
        bid_targets: list[float] = []
        play_values: list[float] = []
        play_targets: list[float] = []
        trick_implied_values: list[float] = []
        bid_trick_implied_values: list[float] = []
        play_trick_implied_values: list[float] = []
        trick_correct = trick_count = owner_correct = owner_count = 0
        # Stage 0 is bid-time; stages 1..3 are play thirds by round_progress.
        trick_stage_correct = [0, 0, 0, 0]
        trick_stage_count = [0, 0, 0, 0]
        trick_players_correct: dict[int, int] = defaultdict(int)
        trick_players_count: dict[int, int] = defaultdict(int)
        trick_bucket_correct: dict[str, int] = defaultdict(int)
        trick_bucket_count: dict[str, int] = defaultdict(int)
        suit_correct = suit_count = 0
        suit_brier: list[float] = []
        suit_stage_bce = [0.0, 0.0, 0.0]
        suit_stage_correct = [0, 0, 0]
        suit_stage_count = [0, 0, 0]
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
                batch = encoded_observations_to_batch(
                    [sample.encoded for sample in mb],
                    device=self.device,
                    event_length_buckets=self.config.event_length_buckets,
                    packing=self.config.batch_packing,
                )
                privileged_owner_targets = (
                    torch.tensor(
                        [sample.owner_targets for sample in mb],
                        dtype=torch.long,
                        device=self.device,
                    )
                    if self.config.oracle_critic
                    else None
                )
                with model_autocast(self.device, self.config.precision):
                    output = self.model(
                        batch,
                        privileged_owner_targets=privileged_owner_targets,
                    )
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
                if output.oracle_value is not None:
                    oracle_values.extend(
                        (output.oracle_value.squeeze(-1).float() + intercepts)
                        .cpu()
                        .tolist()
                    )
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
                trick_hits = (trick_predictions == trick_targets) & active_tricks
                trick_correct += int(trick_hits.sum().cpu())
                trick_count += int(active_tricks.sum().cpu())
                row_trick_correct = trick_hits.sum(dim=-1).cpu().tolist()
                row_trick_count = active_tricks.sum(dim=-1).cpu().tolist()
                for sample, correct_n, count_n in zip(
                    mb,
                    row_trick_correct,
                    row_trick_count,
                ):
                    stage = (
                        0
                        if sample.phase == "bid"
                        else 1 + _round_stage_index(sample.round_progress)
                    )
                    trick_stage_correct[stage] += int(correct_n)
                    trick_stage_count[stage] += int(count_n)
                    trick_players_correct[sample.spec.num_players] += int(correct_n)
                    trick_players_count[sample.spec.num_players] += int(count_n)
                    bucket = _hand_bucket_label(sample.spec.hand_size)
                    trick_bucket_correct[bucket] += int(correct_n)
                    trick_bucket_count[bucket] += int(count_n)
                safe_tricks = trick_targets.clamp(min=0, max=self.config.model_config.bid_count - 1)
                gathered_tricks = trick_probs.gather(-1, safe_tricks.unsqueeze(-1)).squeeze(-1)
                trick_true_probs.extend(gathered_tricks[active_tricks].cpu().tolist())

                if output.suit_presence_logits is not None:
                    suit_targets = torch.tensor(
                        [sample.suit_presence_targets for sample in mb],
                        dtype=torch.long,
                        device=self.device,
                    )
                    active_suits = suit_targets != -100
                    suit_probs = torch.sigmoid(output.suit_presence_logits.float())
                    suit_labels = suit_targets.clamp_min(0).float()
                    suit_hits = (suit_probs >= 0.5).float() == suit_labels
                    suit_correct += int((suit_hits & active_suits).sum().cpu())
                    suit_count += int(active_suits.sum().cpu())
                    suit_brier.extend(
                        ((suit_probs - suit_labels) ** 2)[active_suits]
                        .cpu()
                        .tolist()
                    )
                    suit_bce = F.binary_cross_entropy_with_logits(
                        output.suit_presence_logits.float(),
                        suit_labels,
                        reduction="none",
                    )
                    stage_ids = torch.tensor(
                        [_round_stage_index(sample.round_progress) for sample in mb],
                        dtype=torch.long,
                        device=self.device,
                    )
                    for stage in range(len(suit_stage_count)):
                        stage_mask = (
                            (stage_ids == stage).view(-1, 1, 1) & active_suits
                        )
                        suit_stage_count[stage] += int(stage_mask.sum().cpu())
                        suit_stage_bce[stage] += float(
                            suit_bce[stage_mask].sum().cpu()
                        )
                        suit_stage_correct[stage] += int(
                            (suit_hits & stage_mask).sum().cpu()
                        )

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
            oracle_value_explained_variance=(
                _explained_variance(oracle_values, targets)
                if oracle_values
                else 0.0
            ),
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
            trick_count_accuracy_bidtime=(
                trick_stage_correct[0] / max(trick_stage_count[0], 1)
            ),
            trick_count_accuracy_early=(
                trick_stage_correct[1] / max(trick_stage_count[1], 1)
            ),
            trick_count_accuracy_mid=(
                trick_stage_correct[2] / max(trick_stage_count[2], 1)
            ),
            trick_count_accuracy_late=(
                trick_stage_correct[3] / max(trick_stage_count[3], 1)
            ),
            trick_count_accuracy_by_players={
                players: trick_players_correct[players]
                / max(trick_players_count[players], 1)
                for players in sorted(trick_players_count)
            },
            trick_count_accuracy_by_hand_bucket={
                bucket: trick_bucket_correct[bucket]
                / max(trick_bucket_count[bucket], 1)
                for bucket in sorted(trick_bucket_count)
            },
            suit_presence_accuracy=suit_correct / max(suit_count, 1),
            suit_presence_brier=_mean(suit_brier),
            suit_presence_loss_early=(
                suit_stage_bce[0] / max(suit_stage_count[0], 1)
            ),
            suit_presence_loss_mid=(
                suit_stage_bce[1] / max(suit_stage_count[1], 1)
            ),
            suit_presence_loss_late=(
                suit_stage_bce[2] / max(suit_stage_count[2], 1)
            ),
            suit_presence_accuracy_early=(
                suit_stage_correct[0] / max(suit_stage_count[0], 1)
            ),
            suit_presence_accuracy_mid=(
                suit_stage_correct[1] / max(suit_stage_count[1], 1)
            ),
            suit_presence_accuracy_late=(
                suit_stage_correct[2] / max(suit_stage_count[2], 1)
            ),
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
            "owner_active_since": self.owner_active_since,
            "league": {
                # In-memory policies (empty path) cannot be persisted.
                "snapshot_paths": [
                    snapshot.path
                    for snapshot in self.historical_snapshots
                    if snapshot.path
                ],
                "reward_ema": dict(self.league_reward_ema),
                "payoffs": [
                    [focal_id, table_id, value]
                    for (focal_id, table_id), value in self.league_payoffs.items()
                ],
                "meta_mixture": dict(self.league_meta_mixture),
            },
            "extra": extra or {},
        }
        if self.magnet_model is not None:
            payload["magnet_state_dict"] = self.magnet_model.state_dict()
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
        self.sync_rollout_model()
        self.position_baseline.load_state_dict(payload.get("position_baseline", {}))
        stored_owner_active = payload.get("owner_active_since")
        if stored_owner_active is not None:
            self.owner_active_since = int(stored_owner_active)
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        league = payload.get("league", {})
        # Restore payoff cells before snapshots are re-added so admission-time
        # payoff fills skip every already-known pair instead of re-evaluating.
        self.league_payoffs.update(
            {
                (str(focal_id), str(table_id)): float(value)
                for focal_id, table_id, value in league.get("payoffs", [])
            }
        )
        missing_snapshots: list[str] = []
        for snapshot_path in league.get("snapshot_paths", []):
            stored = Path(snapshot_path)
            # Stored paths are absolute on the machine that saved them; when a
            # run migrates hosts the snapshots still sit beside the checkpoint.
            sibling = Path(path).parent / stored.name
            if stored.exists():
                self.add_historical_checkpoint(stored)
            elif sibling.exists():
                self.add_historical_checkpoint(sibling)
            else:
                missing_snapshots.append(snapshot_path)
        active_ids = {snapshot.snapshot_id for snapshot in self.historical_snapshots}
        self.league_reward_ema.update(
            {
                snapshot_id: float(value)
                for snapshot_id, value in league.get("reward_ema", {}).items()
                if snapshot_id in active_ids
            }
        )
        self._prune_league_state()
        self._refresh_league_mixture()
        if self.config.mmd_enabled:
            self._reset_magnet_model()
            if "magnet_state_dict" in payload:
                self.magnet_model.load_state_dict(payload["magnet_state_dict"])
        return {
            "path": str(path),
            "iteration": payload.get("iteration"),
            "optimizer_loaded": load_optimizer,
            "schema_version": payload["schema_version"],
            "league_snapshots_loaded": len(self.historical_snapshots),
            "league_snapshots_missing": missing_snapshots,
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

    def add_historical_checkpoint(
        self,
        path: str | Path,
        *,
        mirrors_current: bool = False,
    ) -> None:
        resolved = Path(path)
        self._append_snapshot(
            LeagueSnapshot(
                snapshot_id=resolved.stem,
                path=str(resolved),
                policy=ModelPolicy.from_checkpoint(
                    resolved,
                    device=self.device,
                    greedy=False,
                    event_length_buckets=self.config.event_length_buckets,
                    batch_packing=self.config.batch_packing,
                    lean_action_forward=self.config.lean_rollout_forward,
                ),
            ),
            mirrors_current=mirrors_current,
        )

    def replace_historical_snapshots(self, paths: Sequence[str | Path]) -> None:
        """Swap the whole league pool in one step (uniform-league mode).

        Members whose checkpoint is in the new draw keep their loaded policy;
        everyone else is dropped and the new members are loaded fresh.
        """

        resolved = [Path(path) for path in paths]
        loaded = {
            snapshot.snapshot_id: snapshot
            for snapshot in self.historical_snapshots
        }
        self.historical_snapshots = [
            loaded[path.stem] for path in resolved if path.stem in loaded
        ]
        self._prune_league_state()
        self._refresh_league_mixture()
        for path in resolved:
            if path.stem not in loaded:
                self.add_historical_checkpoint(path)

    def add_historical_policy(
        self,
        policy: ActionPolicy,
        *,
        snapshot_id: str | None = None,
    ) -> None:
        """Register an in-memory league opponent (tests, programmatic use)."""

        self._append_snapshot(
            LeagueSnapshot(
                snapshot_id=(
                    snapshot_id
                    or f"policy_{len(self.historical_snapshots)}"
                ),
                path="",
                policy=policy,
            )
        )

    def _append_snapshot(
        self,
        snapshot: LeagueSnapshot,
        *,
        mirrors_current: bool = False,
    ) -> None:
        self.historical_snapshots = [
            existing
            for existing in self.historical_snapshots
            if existing.snapshot_id != snapshot.snapshot_id
        ]
        self.historical_snapshots.append(snapshot)
        self._fill_league_payoffs(snapshot.snapshot_id)
        if mirrors_current:
            # A checkpoint admitted immediately after an update is an exact
            # frozen copy of current. Reuse its just-computed row/column to
            # complete the current payoff cells without another evaluation.
            for other_id in self._league_member_ids():
                if other_id == LEAGUE_CURRENT_MEMBER_ID:
                    continue
                row_source = (snapshot.snapshot_id, other_id)
                column_source = (other_id, snapshot.snapshot_id)
                if row_source in self.league_payoffs:
                    self.league_payoffs[
                        (LEAGUE_CURRENT_MEMBER_ID, other_id)
                    ] = self.league_payoffs[row_source]
                if column_source in self.league_payoffs:
                    self.league_payoffs[
                        (other_id, LEAGUE_CURRENT_MEMBER_ID)
                    ] = self.league_payoffs[column_source]
            self.league_payoffs[
                (LEAGUE_CURRENT_MEMBER_ID, LEAGUE_CURRENT_MEMBER_ID)
            ] = self.league_payoffs.get(
                (snapshot.snapshot_id, LEAGUE_CURRENT_MEMBER_ID),
                0.0,
            )
        self._refresh_league_mixture()
        while len(self.historical_snapshots) > self.config.historical_max_snapshots:
            evicted_id = self._select_league_eviction()
            self.historical_snapshots = [
                existing
                for existing in self.historical_snapshots
                if existing.snapshot_id != evicted_id
            ]
            self._prune_league_state()
            self._refresh_league_mixture()
        self._prune_league_state()

    def _select_league_eviction(self) -> str:
        """Pick the snapshot to drop when the pool is over capacity."""

        snapshots = self.historical_snapshots
        if (
            self.config.league_meta_solver == "regret_matching"
            and self.league_meta_mixture
        ):
            # Evict the member the meta-strategy cares least about; the
            # newest snapshot is protected so progress is always represented.
            candidates = snapshots[:-1]
            return min(
                candidates,
                key=lambda entry: self.league_meta_mixture.get(
                    entry.snapshot_id,
                    0.0,
                ),
            ).snapshot_id
        return snapshots[0].snapshot_id

    def _prune_league_state(self) -> None:
        active_ids = {entry.snapshot_id for entry in self.historical_snapshots}
        self.league_reward_ema = {
            snapshot_id: value
            for snapshot_id, value in self.league_reward_ema.items()
            if snapshot_id in active_ids
        }
        members = active_ids | {
            LEAGUE_HEURISTIC_MEMBER_ID,
            LEAGUE_CURRENT_MEMBER_ID,
        }
        self.league_payoffs = {
            key: value
            for key, value in self.league_payoffs.items()
            if key[0] in members and key[1] in members
        }

    def _league_member_ids(self) -> list[str]:
        return [
            LEAGUE_HEURISTIC_MEMBER_ID,
            *(snapshot.snapshot_id for snapshot in self.historical_snapshots),
            LEAGUE_CURRENT_MEMBER_ID,
        ]

    def _league_member_policy(self, member_id: str) -> ActionPolicy:
        if member_id == LEAGUE_HEURISTIC_MEMBER_ID:
            return self.heuristic_policy
        if member_id == LEAGUE_CURRENT_MEMBER_ID:
            return ModelPolicy(
                self.model,
                device=self.device,
                greedy=False,
                include_game_context=(
                    self.config.training_mode == "game"
                    or self.config.include_game_context
                ),
                precision=self.config.precision,
                event_length_buckets=self.config.event_length_buckets,
                batch_packing=self.config.batch_packing,
                lean_action_forward=self.config.lean_rollout_forward,
                name="league-current",
            )
        for snapshot in self.historical_snapshots:
            if snapshot.snapshot_id == member_id:
                return snapshot.policy
        raise KeyError(f"Unknown league member: {member_id}")

    def _league_payoff_deal_bank(self) -> DealBank:
        if self._league_deal_bank is None:
            hand_sizes = sorted(self.config.hand_sizes)
            self._league_deal_bank = DealBank.generate(
                player_counts=self.config.player_counts,
                hand_sizes=(hand_sizes[len(hand_sizes) // 2],),
                deals_per_configuration=(
                    self.config.league_eval_deals_per_configuration
                ),
                seed=self.config.seed + 7919,
            )
        return self._league_deal_bank

    def _fill_league_payoffs(self, member_id: str) -> None:
        """Compute missing payoff cells involving member_id, both directions.

        Members are frozen, so each cell is evaluated exactly once; resumes
        restore persisted cells before snapshots are re-added, which makes
        this a no-op for known pairs.
        """

        if self.config.league_meta_solver != "regret_matching":
            return
        bank = self._league_payoff_deal_bank()
        for other_id in self._league_member_ids():
            if other_id == member_id:
                continue
            for focal_id, table_id in (
                (member_id, other_id),
                (other_id, member_id),
            ):
                if (focal_id, table_id) in self.league_payoffs:
                    continue
                report = evaluate_policy(
                    self._league_member_policy(focal_id),
                    self._league_member_policy(table_id),
                    bank,
                    bootstrap_samples=8,
                    seed=self.config.seed,
                    batch_size=256,
                )
                self.league_payoffs[(focal_id, table_id)] = float(
                    report.macro_relative_reward
                )

    def refresh_league_payoffs(
        self,
        *,
        iteration: int,
        force: bool = False,
    ) -> bool:
        """Refresh every payoff cell involving the changing current policy.

        Frozen snapshot/snapshot and snapshot/heuristic cells are filled once
        on admission. Only the current row and column are overwritten on the
        configured cadence.
        """

        if self.config.league_meta_solver != "regret_matching":
            return False
        if not force and (
            self.config.league_eval_every <= 0
            or iteration % self.config.league_eval_every != 0
        ):
            return False
        bank = self._league_payoff_deal_bank()
        evaluated: set[tuple[str, str]] = set()
        for other_id in self._league_member_ids():
            for focal_id, table_id in (
                (LEAGUE_CURRENT_MEMBER_ID, other_id),
                (other_id, LEAGUE_CURRENT_MEMBER_ID),
            ):
                key = (focal_id, table_id)
                if key in evaluated:
                    continue
                evaluated.add(key)
                report = evaluate_policy(
                    self._league_member_policy(focal_id),
                    self._league_member_policy(table_id),
                    bank,
                    bootstrap_samples=8,
                    seed=self.config.seed + iteration,
                    batch_size=256,
                )
                self.league_payoffs[key] = float(
                    report.macro_relative_reward
                )
        self._refresh_league_mixture()
        return True

    def _refresh_league_mixture(self) -> None:
        if self.config.league_meta_solver != "regret_matching":
            self.league_meta_mixture = {}
            return
        self.league_meta_mixture = solve_meta_mixture(
            self._league_member_ids(),
            self.league_payoffs,
        )

    def _prepare_historical_iteration_sampling(self) -> None:
        """Choose the at-most-two historical models used this collection."""

        self._iteration_exploit_snapshot = None
        self._iteration_probe_snapshot = None
        snapshots = self.historical_snapshots
        if not self.config.batched_league_sampling or not snapshots:
            return
        if len(snapshots) == 1:
            self._iteration_exploit_snapshot = snapshots[0]
            self._iteration_probe_snapshot = snapshots[0]
            return
        if self.config.league_meta_solver == "uniform":
            self._iteration_exploit_snapshot = self.rng.choice(snapshots)
            self._iteration_probe_snapshot = self.rng.choice(snapshots)
            return
        if (
            self.config.league_meta_solver == "regret_matching"
            and self.league_meta_mixture
        ):
            missing_weight = max(self.league_meta_mixture.values())
            weights = [
                self.league_meta_mixture.get(snapshot.snapshot_id, missing_weight)
                for snapshot in snapshots
            ]
            self._iteration_exploit_snapshot = self._weighted_snapshot(
                snapshots,
                weights,
            )
            self._iteration_probe_snapshot = self.rng.choice(snapshots)
        else:
            # The difficulty fallback has no uniform smoothing today. Draw it
            # once per collection to increase batch size without changing its
            # marginal distribution across collections.
            self._iteration_exploit_snapshot = (
                self._draw_historical_snapshot_unbatched()
            )

    def _draw_historical_snapshot(self) -> LeagueSnapshot:
        """Sample from this iteration's bounded pool or the legacy mixture."""

        if self._iteration_exploit_snapshot is not None:
            if (
                self._iteration_probe_snapshot is not None
                and self.rng.random() < self.config.league_probe_fraction
            ):
                return self._iteration_probe_snapshot
            return self._iteration_exploit_snapshot
        return self._draw_historical_snapshot_unbatched()

    def _draw_historical_snapshot_unbatched(self) -> LeagueSnapshot:
        """Sample one snapshot using the original per-assignment policy."""

        snapshots = self.historical_snapshots
        if not snapshots:
            raise RuntimeError("No historical snapshots available to sample.")
        if self.config.league_meta_solver == "uniform":
            return self.rng.choice(snapshots)
        if (
            self.config.league_meta_solver == "regret_matching"
            and self.league_meta_mixture
        ):
            probe_weight = max(self.league_meta_mixture.values())
            raw_weights = [
                self.league_meta_mixture.get(snapshot.snapshot_id, probe_weight)
                for snapshot in snapshots
            ]
            total = sum(raw_weights)
            if total > 1e-12:
                # 10% uniform smoothing keeps low-mass members occasionally
                # in play so their difficulty estimates stay fresh.
                count = len(raw_weights)
                weights = [
                    0.9 * weight / total + 0.1 / count for weight in raw_weights
                ]
                return self._weighted_snapshot(snapshots, weights)
        emas = [
            self.league_reward_ema.get(snapshot.snapshot_id)
            for snapshot in snapshots
        ]
        known = [value for value in emas if value is not None]
        # Unseen snapshots are scored at least as hard as the hardest known
        # opponent (and never easier than neutral) so they get probed first.
        optimistic = min([0.0, *known])
        scores = [value if value is not None else optimistic for value in emas]
        temperature = max(self.config.league_temperature, 1e-6)
        logits = [-score / temperature for score in scores]
        max_logit = max(logits)
        weights = [math.exp(logit - max_logit) for logit in logits]
        return self._weighted_snapshot(snapshots, weights)

    def _weighted_snapshot(
        self,
        snapshots: list[LeagueSnapshot],
        weights: list[float],
    ) -> LeagueSnapshot:
        threshold = self.rng.random() * sum(weights)
        cumulative = 0.0
        for snapshot, weight in zip(snapshots, weights):
            cumulative += weight
            if threshold <= cumulative:
                return snapshot
        return snapshots[-1]

    def _update_league_ema(self, outcomes: Iterable[RoundOutcome]) -> None:
        grouped: dict[str, list[float]] = defaultdict(list)
        for outcome in outcomes:
            if (
                outcome.opponent_snapshot_id is not None
                and outcome.focal_reward is not None
                # Explore rounds have a noised focal seat; their returns
                # would corrupt snapshot difficulty estimates.
                and outcome.opponent_arm not in EXPLORE_ARMS
            ):
                grouped[outcome.opponent_snapshot_id].append(outcome.focal_reward)
        decay = self.config.league_reward_decay
        for snapshot_id, rewards in grouped.items():
            batch_mean = sum(rewards) / len(rewards)
            previous = self.league_reward_ema.get(snapshot_id, batch_mean)
            self.league_reward_ema[snapshot_id] = (
                decay * previous + (1.0 - decay) * batch_mean
            )

    def _effective_arm_fractions(self) -> dict[OpponentArm, float]:
        fractions: dict[OpponentArm, float] = {
            "self": self.config.self_play_fraction,
            "heuristic": self.config.heuristic_fraction,
            "mixed": self.config.mixed_fraction,
            "historical": self.config.historical_fraction,
            "explore_self": self.config.explore_self_fraction,
            "explore_historical": self.config.explore_historical_fraction,
        }
        if not self.historical_policies:
            fractions["self"] += fractions["historical"]
            fractions["historical"] = 0.0
            fractions["explore_self"] += fractions["explore_historical"]
            fractions["explore_historical"] = 0.0
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
        opponent_policies, snapshot_id = self._opponent_policies(
            spec.num_players,
            focal_player,
            arm,
        )
        return _ActiveEpisode(
            env=env,
            spec=spec,
            episode_id=episode_id,
            opponent_arm=arm,
            trainable_players=_trainable_players(
                focal_player,
                arm,
                opponent_policies,
            ),
            opponent_policies=opponent_policies,
            focal_player=focal_player,
            opponent_snapshot_id=snapshot_id,
            explore_tempered=self._draw_explore_tempered(arm),
            explore_uniform_index=self._draw_explore_uniform_index(
                arm,
                spec.hand_size,
            ),
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
        opponent_policies, snapshot_id = self._opponent_policies(
            num_players,
            focal_player,
            arm,
        )
        return _ActiveEpisode(
            env=env,
            spec=spec,
            episode_id=episode_id,
            opponent_arm=arm,
            trainable_players=_trainable_players(
                focal_player,
                arm,
                opponent_policies,
            ),
            opponent_policies=opponent_policies,
            focal_player=focal_player,
            game_schedule=game_schedule,
            opponent_snapshot_id=snapshot_id,
            explore_tempered=self._draw_explore_tempered(arm),
        )

    def _opponent_policies(
        self,
        num_players: int,
        focal_player: int,
        arm: OpponentArm,
    ) -> tuple[dict[int, ActionPolicy | None], str | None]:
        opponent_policies: dict[int, ActionPolicy | None] = {}
        # Historical episodes share one snapshot across all opponent seats so
        # the round outcome attributes cleanly to that snapshot's league EMA.
        episode_snapshot: LeagueSnapshot | None = (
            self._draw_historical_snapshot()
            if arm in ("historical", "explore_historical")
            else None
        )
        for player in range(num_players):
            if player == focal_player:
                continue
            if arm in ("self", "explore_self"):
                # None = current weights. On explore_self tables the caller
                # marks these seats non-trainable, so they act as frozen raw
                # copies of the current policy.
                policy = None
            elif arm == "heuristic":
                policy = self.heuristic_policy
            elif arm in ("historical", "explore_historical"):
                policy = episode_snapshot.policy
            else:
                # Mixed tables seat current policy and historical snapshots
                # 50/50; the heuristic plays only through its dedicated arm.
                policy = (
                    self._draw_historical_snapshot().policy
                    if self.historical_snapshots and self.rng.random() < 0.5
                    else None
                )
            opponent_policies[player] = policy
        return opponent_policies, (
            episode_snapshot.snapshot_id if episode_snapshot else None
        )

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
                if self._active_historical_policy_ids is not None:
                    self._active_historical_policy_ids.add(key)
                    if self._active_collection_stats is not None:
                        self._active_collection_stats.historical_policy_count = len(
                            self._active_historical_policy_ids
                        )
                started = time.perf_counter()
                selected = policy.act_many(
                    [episode.env for episode in rows],
                    rngs=[self.rng] * len(rows),
                )
                stats = self._active_collection_stats
                if stats is not None:
                    stats.historical_forward_sec += time.perf_counter() - started
                    stats.historical_forward_calls += 1
                    stats.historical_forward_rows += len(rows)
                    lengths = [
                        len(
                            [
                                event
                                for event in episode.env.state.event_log
                                if event.round_index
                                == episode.env.state.current_round.round_index
                            ]
                        )
                        for episode in rows
                    ]
                    stats.valid_event_tokens += sum(lengths)
                    stats.processed_event_tokens += (
                        len(rows) * self._event_batch_length(max(lengths, default=0))
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
        include_game_context = (
            self.config.training_mode == "game"
            or self.config.include_game_context
        )
        requests = []
        for episode in episodes:
            # Non-trainable current-weight seats (explore_self opponents) act
            # as frozen copies: raw sampling, and their rows are discarded.
            collect = episode.env.current_player() in episode.trainable_players
            # On explore tables only the focal collects, so its samples-so-far
            # count is exactly its decision index within the round.
            explore_uniform = (
                collect
                and episode.explore_uniform_index is not None
                and len(episode.samples) == episode.explore_uniform_index
            )
            requests.append(
                build_decision_request(
                    episode.env,
                    episode_id=episode.episode_id,
                    opponent_arm=episode.opponent_arm,
                    policy_ref="current",
                    model_config=self.config.model_config,
                    include_game_context=include_game_context,
                    explore_tempered=episode.explore_tempered and collect,
                    collect=collect,
                    explore_uniform=explore_uniform,
                )
            )
        rows = self._forward_decision_rows(requests)
        return {
            request.episode_id: row
            for request, row in zip(requests, rows)
        }

    def _draw_explore_tempered(self, arm: OpponentArm) -> bool:
        return (
            self.config.explore_temperature_fraction > 0.0
            and arm in self.config.explore_temperature_arms
            and self.rng.random() < self.config.explore_temperature_fraction
        )

    def _hand_noise_scale(self, hand_size: int) -> float:
        if not self.config.explore_noise_hand_normalized:
            return 1.0
        reference = min(self.config.hand_sizes)
        return (reference + 1) / (hand_size + 1)

    def _draw_explore_uniform_index(
        self,
        arm: OpponentArm,
        hand_size: int,
    ) -> int | None:
        # Arm check precedes any rng draw so non-explore rounds consume the
        # same number of draws on both collection paths.
        probability = self.config.explore_uniform_round_probability
        if probability <= 0.0 or arm not in EXPLORE_ARMS:
            return None
        if self.rng.random() >= probability * self._hand_noise_scale(hand_size):
            return None
        return self.rng.randrange(hand_size + 1)

    def _request_explore_eps(self, request: DecisionRequest) -> float:
        if not request.collect:
            return 0.0
        if request.explore_uniform:
            # The round's single injected action: pure uniform over legal.
            return 1.0
        bid_eps, play_eps = self.config.explore_eps_by_arm.get(
            request.opponent_arm,
            (self.config.explore_eps_bid, self.config.explore_eps_play),
        )
        return bid_eps if request.phase == "bid" else play_eps

    def _request_explore_temperature(self, request: DecisionRequest) -> float:
        if not request.collect or not request.explore_tempered:
            return 1.0
        base = (
            self.config.explore_temperature_bid
            if request.phase == "bid"
            else self.config.explore_temperature_play
        )
        return 1.0 + (base - 1.0) * self._hand_noise_scale(request.hand_size)

    def _forward_decision_rows(
        self,
        requests: list[DecisionRequest],
    ) -> list[tuple[RolloutSample, BidAction | PlayCardAction]]:
        started = time.perf_counter()
        encoded = [request.encoded for request in requests]
        owner_targets = [request.owner_targets for request in requests]
        batch = encoded_observations_to_batch(
            encoded,
            device=self.device,
            event_length_buckets=self.config.event_length_buckets,
            packing=self.config.batch_packing,
        )
        privileged_owner_targets = (
            torch.tensor(owner_targets, dtype=torch.long, device=self.device)
            if self.config.oracle_critic
            else None
        )
        with torch.inference_mode(), model_autocast(
            self.device,
            self.config.precision,
        ):
            collection_model = self._collection_model()
            output = (
                collection_model.forward_rollout(
                    batch,
                    privileged_owner_targets=privileged_owner_targets,
                )
                if self.config.lean_rollout_forward
                else collection_model(
                    batch,
                    need_owner=False,
                    privileged_owner_targets=privileged_owner_targets,
                )
            )
        implied_values_enabled = (
            self.config.trick_baseline
            and self.config.training_mode != "game"
        )
        implied_values_tensor = torch.zeros(
            len(requests),
            dtype=torch.float32,
            device=self.device,
        )
        if implied_values_enabled:
            # Trick-head-implied expected relative score acts as a per-state
            # potential; folding it into the intercept residualizes both the
            # value target and the advantage without biasing the optimum.
            implied_values_tensor = _trick_implied_relative_values(
                torch.softmax(output.masked_trick_count_logits.float(), dim=-1),
                batch.bid_values,
                batch.active_player_mask,
            )
        bid_mask = torch.tensor(
            [request.phase == "bid" for request in requests],
            dtype=torch.bool,
            device=self.device,
        )
        masked_logits = combined_action_logits(output, bid_mask)
        distribution = Categorical(logits=masked_logits)
        eps_values = [self._request_explore_eps(request) for request in requests]
        temperature_values = [
            self._request_explore_temperature(request) for request in requests
        ]
        tempered_any = any(value != 1.0 for value in temperature_values)
        if tempered_any or any(eps > 0.0 for eps in eps_values):
            # Behavior policy = (1-eps)*softmax(logits/T) + eps*uniform-over-
            # legal. The temperature flattens the current policy on designated
            # rounds so rollouts visit more diverse trajectories; eps keeps
            # rare strategies (e.g. extreme bids) at a guaranteed trial rate
            # even once the policy is sharp. Sampling AND old_logprob both use
            # this behavior distribution, which keeps the PPO ratio
            # importance-correct — the update itself stays standard. Both
            # knobs are per request (opponent arm x phase x round flag), so
            # rows from different exploration regimes share one forward batch.
            if tempered_any:
                temperature_column = torch.tensor(
                    temperature_values,
                    dtype=masked_logits.dtype,
                    device=masked_logits.device,
                ).unsqueeze(-1)
                base_probs = Categorical(
                    logits=masked_logits / temperature_column,
                ).probs
            else:
                base_probs = distribution.probs
            legal = masked_logits > torch.finfo(masked_logits.dtype).min / 2
            uniform = legal.float() / legal.sum(dim=-1, keepdim=True).clamp_min(1)
            eps_column = torch.tensor(
                eps_values,
                dtype=torch.float32,
                device=masked_logits.device,
            ).unsqueeze(-1)
            behavior = Categorical(
                probs=(1.0 - eps_column) * base_probs + eps_column * uniform,
            )
            action_indices = behavior.sample()
            sample_logprobs = behavior.log_prob(action_indices)
            policy_logprobs = distribution.log_prob(action_indices)
        else:
            action_indices = distribution.sample()
            sample_logprobs = distribution.log_prob(action_indices)
            policy_logprobs = sample_logprobs
        # The GAE baseline uses the oracle residual when the privileged
        # critic is enabled; the plain residual is kept for value training.
        baseline_residuals = (
            output.oracle_value.squeeze(-1).float()
            if output.oracle_value is not None
            else output.value.squeeze(-1).float()
        )
        rollout_rows = torch.stack(
            (
                action_indices.float(),
                output.value.squeeze(-1).float(),
                sample_logprobs,
                policy_logprobs,
                distribution.entropy(),
                baseline_residuals,
                implied_values_tensor,
            ),
            dim=-1,
        ).cpu().tolist()
        results: list[tuple[RolloutSample, BidAction | PlayCardAction]] = []
        for index, request in enumerate(requests):
            (
                action_index_value,
                residual,
                old_logprob,
                old_policy_logprob,
                old_entropy,
                baseline_residual,
                implied_value,
            ) = rollout_rows[index]
            action_index = int(action_index_value)
            position = request.encoded.bidding_position
            current_spec = RoundSpec(request.num_players, request.hand_size)
            key = (current_spec.num_players, current_spec.hand_size, position)
            intercept = (
                0.0
                if self.config.training_mode == "game"
                else self.position_baseline.get(key)
            )
            if implied_values_enabled:
                intercept += implied_value
            action: BidAction | PlayCardAction
            if request.phase == "bid":
                action = BidAction(request.player, action_index)
            else:
                action = PlayCardAction(request.player, card_from_id(action_index))
            sample = RolloutSample(
                encoded=request.encoded,
                phase=request.phase,
                action_index=action_index,
                old_logprob=old_logprob,
                old_policy_logprob=old_policy_logprob,
                old_value=intercept + baseline_residual,
                old_residual_value=residual,
                old_entropy=old_entropy,
                position_intercept=intercept,
                acting_player=request.player,
                episode_id=request.episode_id,
                round_id=request.round_index,
                spec=current_spec,
                bidding_position=position,
                opponent_arm=request.opponent_arm,
                owner_targets=request.owner_targets,
                suit_presence_targets=request.suit_presence_targets,
                observation=request.observation,
                trick_position=request.trick_position,
                round_progress=request.round_progress,
            )
            results.append((sample, action))
        stats = self._active_collection_stats
        if stats is not None:
            stats.current_forward_sec += time.perf_counter() - started
            stats.current_forward_calls += 1
            stats.current_forward_rows += len(requests)
            stats.valid_event_tokens += sum(
                sum(request.encoded.event_valid_mask)
                for request in requests
            )
            stats.processed_event_tokens += len(requests) * batch.event_length
        return results

    def _event_batch_length(self, max_valid_length: int) -> int:
        padded_length = self.config.model_config.max_seq_len
        for bucket in self.config.event_length_buckets:
            if bucket >= max_valid_length:
                return min(bucket, padded_length)
        return padded_length

    def _assign_terminal_targets(
        self,
        episode: _ActiveEpisode,
    ) -> list[tuple[PositionKey, float]]:
        return self._finalize_round_result(
            samples=episode.samples,
            trainable_players=episode.trainable_players,
            num_players=episode.env.config.num_players,
            result=RoundResult.from_round_state(episode.env.state.current_round),
        )

    def _finalize_round_result(
        self,
        *,
        samples: list[RolloutSample],
        trainable_players: frozenset[int],
        num_players: int,
        result: RoundResult,
    ) -> list[tuple[PositionKey, float]]:
        rewards = compute_relative_rewards(result.round_scores)
        bid_positions = {bid.player: bid.position for bid in result.bids}
        for player in trainable_players:
            self._assign_gae(
                [
                    sample
                    for sample in samples
                    if sample.acting_player == player
                ],
                terminal_reward=rewards[player],
                gae_lambda=self.config.round_gae_lambda,
            )
        for sample in samples:
            sample.final_trick_targets = _final_tricks_relative(
                result.tricks_won,
                sample.acting_player,
                num_players,
                self.config.model_config,
            )
            sample.final_bid_targets = _final_bids_relative(
                result.bids,
                sample.acting_player,
                num_players,
                self.config.model_config,
            )
        return [
            (
                (num_players, result.hand_size, bid_positions[player]),
                rewards[player],
            )
            for player in trainable_players
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
        return self._round_outcome_from_result(
            episode_id=episode.episode_id,
            num_players=episode.env.config.num_players,
            opponent_arm=episode.opponent_arm,
            focal_player=episode.focal_player,
            opponent_snapshot_id=episode.opponent_snapshot_id,
            result=RoundResult.from_round_state(round_state),
        )

    def _round_outcome_from_result(
        self,
        *,
        episode_id: int,
        num_players: int,
        opponent_arm: OpponentArm,
        focal_player: int,
        opponent_snapshot_id: str | None,
        result: RoundResult,
    ) -> RoundOutcome:
        bids = {bid.player: bid for bid in result.bids}
        rewards = compute_relative_rewards(result.round_scores)
        focal_bid = bids[focal_player]
        return RoundOutcome(
            episode_id=episode_id,
            spec=RoundSpec(num_players, result.hand_size),
            opponent_arm=opponent_arm,
            focal_reward=rewards[focal_player],
            bid_hit_count=sum(
                result.tricks_won[player] == bid.value
                for player, bid in bids.items()
            ),
            bid_player_count=len(bids),
            bid_abs_error_mean=_mean(
                [
                    float(abs(result.tricks_won[player] - bid.value))
                    for player, bid in bids.items()
                ]
            ),
            focal_bid_hit=int(
                result.tricks_won[focal_player] == focal_bid.value
            ),
            focal_bid_abs_error=float(
                abs(result.tricks_won[focal_player] - focal_bid.value)
            ),
            position_rewards={bid.position: rewards[player] for player, bid in bids.items()},
            opponent_snapshot_id=opponent_snapshot_id,
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
        # Gradient mass per cell follows the sampling weights: uniform cells
        # keep the classic 1/len(specs) share, weighted cells contribute in
        # proportion to their round quota (that's the point of the weights —
        # more players/cards should also carry more of the loss).
        quotas = self.config.spec_round_quotas()
        if quotas is None:
            cell_share = {spec: 1.0 / len(self.config.specs) for spec in self.config.specs}
        else:
            total_rounds = sum(quotas.values())
            cell_share = {spec: quota / total_rounds for spec, quota in quotas.items()}
        for sample in buffer.samples:
            sample.round_weight = fractions[sample.opponent_arm] * cell_share[sample.spec] / (
                rounds_per_cell[(sample.spec, sample.opponent_arm)]
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

    def _magnet_kl_terms(
        self,
        output,
        magnet_output,
        samples: list[RolloutSample],
    ) -> Tensor:
        kl = torch.zeros(len(samples), dtype=torch.float32, device=self.device)
        bid_indices = [index for index, sample in enumerate(samples) if sample.phase == "bid"]
        play_indices = [index for index, sample in enumerate(samples) if sample.phase == "play"]
        for indices, logits, magnet_logits in (
            (bid_indices, output.masked_bid_logits, magnet_output.masked_bid_logits),
            (play_indices, output.masked_card_logits, magnet_output.masked_card_logits),
        ):
            if not indices:
                continue
            kl[indices] = torch.distributions.kl_divergence(
                Categorical(logits=logits[indices].float()),
                Categorical(logits=magnet_logits[indices].float()),
            )
        # True KL is non-negative; matching distributions can round below 0.
        return kl.clamp_min(0.0)

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
        for weights, values, name in (
            (self.config.player_count_weights, self.config.player_counts, "player_count_weights"),
            (self.config.hand_size_weights, self.config.hand_sizes, "hand_size_weights"),
        ):
            if not weights:
                continue
            if len(weights) != len(values):
                raise ValueError(f"{name} must align with its value tuple.")
            if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
                raise ValueError(f"{name} must be nonnegative with a positive sum.")
        if (
            self.config.training_mode == "game"
            and (self.config.player_count_weights or self.config.hand_size_weights)
        ):
            raise ValueError("Sampling weights are only supported in round mode.")
        if not 0.0 <= self.config.explore_temperature_fraction <= 1.0:
            raise ValueError("explore_temperature_fraction must be in [0, 1].")
        if not 0.0 <= self.config.explore_uniform_round_probability <= 1.0:
            raise ValueError(
                "explore_uniform_round_probability must be in [0, 1]."
            )
        if (
            self.config.explore_temperature_bid < 1.0
            or self.config.explore_temperature_play < 1.0
        ):
            raise ValueError("explore temperatures must be at least 1.0.")
        unknown_temperature_arms = set(self.config.explore_temperature_arms) - {
            "self",
            "heuristic",
            "mixed",
            "historical",
            "explore_self",
            "explore_historical",
        }
        if unknown_temperature_arms:
            raise ValueError(
                f"explore_temperature_arms has unknown arms {sorted(unknown_temperature_arms)!r}."
            )
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
        if self.config.env_workers < 0:
            raise ValueError("env_workers must be zero or positive.")
        if (
            any(bucket <= 0 for bucket in self.config.event_length_buckets)
            or tuple(sorted(set(self.config.event_length_buckets)))
            != self.config.event_length_buckets
        ):
            raise ValueError(
                "event_length_buckets must be unique positive values in ascending order."
            )
        if self.config.batch_packing not in {"torch", "numpy"}:
            raise ValueError("batch_packing must be 'torch' or 'numpy'.")
        if not 0.0 <= self.config.league_probe_fraction <= 1.0:
            raise ValueError("league_probe_fraction must be in [0, 1].")
        eps_values = [self.config.explore_eps_bid, self.config.explore_eps_play]
        for arm, arm_eps in self.config.explore_eps_by_arm.items():
            if arm not in {
                "self",
                "heuristic",
                "mixed",
                "historical",
                "explore_self",
                "explore_historical",
            }:
                raise ValueError(f"explore_eps_by_arm has unknown arm {arm!r}.")
            if len(arm_eps) != 2:
                raise ValueError("explore_eps_by_arm values must be (bid, play).")
            eps_values.extend(arm_eps)
        for eps in eps_values:
            if not 0.0 <= eps <= 1.0:
                raise ValueError("explore_eps values must be in [0, 1].")
        if self.config.microbatch_size is not None and self.config.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive when set.")
        fractions = (
            self.config.self_play_fraction,
            self.config.heuristic_fraction,
            self.config.mixed_fraction,
            self.config.historical_fraction,
            self.config.explore_self_fraction,
            self.config.explore_historical_fraction,
        )
        if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
            raise ValueError("Opponent-arm fractions must be in [0, 1].")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError("Opponent-arm fractions must sum to 1.")
        if self.config.training_mode == "game" and (
            self.config.explore_self_fraction > 0.0
            or self.config.explore_historical_fraction > 0.0
        ):
            raise ValueError("Explore arms are only supported in round mode.")
        if self.config.owner_warmup_iterations < 0:
            raise ValueError("owner_warmup_iterations must be nonnegative.")
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
        if self.config.league_temperature <= 0.0:
            raise ValueError("league_temperature must be positive.")
        if self.config.league_meta_solver not in {
            "softmax_ema",
            "regret_matching",
            "uniform",
        }:
            raise ValueError(
                "league_meta_solver must be 'softmax_ema', 'regret_matching', "
                "or 'uniform'."
            )
        if self.config.league_eval_deals_per_configuration < 1:
            raise ValueError(
                "league_eval_deals_per_configuration must be positive."
            )
        if self.config.league_eval_every < 0:
            raise ValueError("league_eval_every must be nonnegative.")
        if self.config.target_kl is not None and self.config.target_kl <= 0.0:
            raise ValueError("target_kl must be positive when set.")
        if not 0.0 <= self.config.league_reward_decay < 1.0:
            raise ValueError("league_reward_decay must be in [0, 1).")
        if self.config.oracle_critic and not self.config.model_config.oracle_critic:
            raise ValueError(
                "oracle_critic training requires model_config.oracle_critic=True."
            )
        if self.config.oracle_value_coef < 0.0:
            raise ValueError("oracle_value_coef must be nonnegative.")
        if self.config.mmd_coef < 0.0:
            raise ValueError("mmd_coef must be nonnegative.")
        if not 0.0 <= self.config.mmd_magnet_decay < 1.0:
            raise ValueError("mmd_magnet_decay must be in [0, 1).")
        if self.config.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be one of: fp32, bf16, fp16.")
        if self.config.owner_capacity_coef < 0.0:
            raise ValueError("owner_capacity_coef must be nonnegative.")
        if self.config.suit_coef < 0.0:
            raise ValueError("suit_coef must be nonnegative.")
        if self.config.model_config.owner_sinkhorn_iterations < 1:
            raise ValueError(
                "owner_sinkhorn_iterations must be positive."
            )


def build_decision_request(
    env: PlumpEnv,
    *,
    episode_id: int,
    opponent_arm: OpponentArm,
    policy_ref: str,
    model_config: ModelConfig,
    include_game_context: bool,
    explore_tempered: bool = False,
    collect: bool = True,
    explore_uniform: bool = False,
) -> DecisionRequest:
    """Everything CPU-side a model decision needs; also runs in env workers."""

    player = env.current_player()
    observation = env.get_observation(player)
    encoded = encode_observation(
        observation,
        model_config,
        include_game_context=include_game_context,
    )
    current_round = env.state.current_round
    return DecisionRequest(
        episode_id=episode_id,
        player=player,
        phase="bid" if env.phase() == Phase.BIDDING else "play",
        policy_ref=policy_ref,
        encoded=encoded,
        owner_targets=_owner_targets_relative(
            env,
            player,
            encoded.owner_valid_mask,
            model_config,
        ),
        suit_presence_targets=_suit_presence_targets_relative(
            env,
            player,
            model_config,
        ),
        observation=_snapshot_observation(observation),
        round_index=current_round.round_index,
        num_players=env.config.num_players,
        hand_size=current_round.hand_size,
        opponent_arm=opponent_arm,
        explore_tempered=explore_tempered,
        collect=collect,
        explore_uniform=explore_uniform,
        trick_position=(
            len(observation.current_trick.plays)
            if observation.current_trick is not None
            else -1
        ),
        round_progress=(
            len(observation.completed_tricks)
            / max(current_round.hand_size, 1)
        ),
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


def _trainable_players(
    focal_player: int,
    arm: OpponentArm,
    opponent_policies: dict[int, ActionPolicy | None],
) -> frozenset[int]:
    """Seats that produce training samples for this arm.

    Explore arms train only the focal seat: current-weight opponents on
    explore_self tables are frozen copies that sample raw and yield no
    samples, so the noised learner never optimizes against noisy play.
    """

    if arm in EXPLORE_ARMS:
        return frozenset({focal_player})
    return _current_policy_players(focal_player, opponent_policies)


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


def _round_stage_index(round_progress: float) -> int:
    """Bucket a round's completed-trick fraction into early/mid/late (0/1/2)."""

    if round_progress < 1.0 / 3.0:
        return 0
    if round_progress < 2.0 / 3.0:
        return 1
    return 2


def _hand_bucket_label(hand_size: int) -> str:
    """Group hand sizes into the dashboard's 3-5 / 6-8 / 9-10 buckets."""

    if hand_size <= 5:
        return "3_5"
    if hand_size <= 8:
        return "6_8"
    return "9_10"


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
        "oracle": stats.oracle_value_loss,
        "magnet_kl": stats.magnet_kl,
        "entropy": stats.entropy,
        "aux": stats.auxiliary_loss,
        "trick": stats.trick_loss,
        "owner": stats.owner_loss,
        "owner_ce": stats.owner_ce_loss,
        "owner_cap": stats.owner_capacity_loss,
        "suit": stats.suit_presence_loss,
        "kl": stats.approx_kl,
        "clip": stats.clip_fraction,
        "epochs": stats.epochs_run,
        "skipped": stats.skipped_steps,
    }
    return " ".join(
        f"{key}={value:.4f}" if isinstance(value, float) and math.isfinite(value) else f"{key}={value}"
        for key, value in values.items()
    )
