"""Schema-v5 information-set expert-iteration training."""

from __future__ import annotations

import copy
import gzip
import math
import pickle
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from plump.env import PlumpEnv
from plump.information_search import (
    InformationSearchConfig,
    InformationSearchDecision,
    InformationSetSearch,
)
from plump.modeling import (
    EncodedObservation,
    ModelConfig,
    PlumpSearchModel,
    card_from_id,
    encode_observation,
)
from plump.modeling.encoding import NUM_CARDS
from plump.modeling.torch_model import (
    best_torch_device,
    encoded_observations_to_batch,
    index_model_batch,
    load_v4_weights,
    model_autocast,
)
from plump.policies import ActionPolicy, HeuristicPolicy, ModelPolicy
from plump.rounds import (
    RoundSpec,
    descending_ascending_schedule,
    round_game_config,
    rules_fingerprint,
)
from plump.state import BidAction, GameConfig, Phase, PlayCardAction

from .common import (
    OpponentArm,
    OpponentMix,
    PositionBaseline,
    allocate_opponent_arms,
    compute_relative_rewards,
    final_bids_relative,
    final_tricks_relative,
    owner_targets_relative,
    position_key,
)


CHECKPOINT_SCHEMA_VERSION = 5
OBSERVATION_SCHEMA_VERSION = 4
SamplePhase = Literal["bid", "play"]
TrainingMode = Literal["round", "game"]
ReplayKey = tuple[int, int, int, SamplePhase, int, OpponentArm]


@dataclass
class ExpertIterationConfig:
    player_counts: tuple[int, ...] = (3, 4, 5)
    hand_sizes: tuple[int, ...] = tuple(range(3, 11))
    rounds_per_configuration: int = 16
    games_per_player_seat: int = 4
    concurrent_episodes: int = 32
    play_search_fraction: float = 1.0
    training_mode: TrainingMode = "round"
    game_schedule: tuple[int, ...] = ()
    min_cards: int = 3
    max_cards: int = 10
    learning_rate: float = 2e-5
    minibatch_size: int = 1_440
    microbatch_size: int = 576
    replay_capacity: int = 50_000
    replay_max_age: int = 100
    updates_per_new_state: float = 4.0 / 1_440.0
    policy_coef: float = 1.0
    q_coef: float = 0.25
    value_coef: float = 0.5
    trick_coef: float = 0.1
    owner_coef: float = 0.05
    owner_capacity_coef: float = 0.1
    entropy_floor_coef: float = 0.002
    max_grad_norm: float = 1.0
    position_baseline_decay: float = 0.98
    opponent_mix: OpponentMix = field(default_factory=OpponentMix)
    historical_checkpoint_paths: tuple[str, ...] = ()
    historical_max_snapshots: int = 4
    precision: str = "bf16"
    seed: int = 1
    device: str | None = None
    model_config: ModelConfig = field(default_factory=ModelConfig)
    search_config: InformationSearchConfig = field(
        default_factory=InformationSearchConfig
    )

    @property
    def specs(self) -> tuple[RoundSpec, ...]:
        return tuple(
            RoundSpec(players, hand_size)
            for players in self.player_counts
            for hand_size in self.hand_sizes
        )

    @property
    def resolved_game_schedule(self) -> tuple[int, ...]:
        return self.game_schedule or tuple(
            descending_ascending_schedule(
                min_cards=self.min_cards,
                max_cards=self.max_cards,
            )
        )


@dataclass
class ExpertSample:
    encoded: EncodedObservation
    phase: SamplePhase
    action_index: int
    spec: RoundSpec
    bidding_position: int
    trick_position: int
    opponent_arm: OpponentArm
    cycle: int
    episode_id: int
    acting_player: int
    round_id: int
    position_intercept: float
    owner_targets: list[int]
    search_policy: list[float] | None = None
    q_targets: list[float] | None = None
    q_stderr: list[float] | None = None
    accepted: bool = False
    forced: bool = False
    target_value: float | None = None
    final_trick_targets: list[int] | None = None
    final_bid_targets: list[int] | None = None
    search_js: float = float("nan")
    search_agreement: bool = False
    search_nodes: int = 0
    search_depth: int = 0
    search_determinizations: int = 0
    search_leaf_rollouts: int = 0
    search_leaf_values: int = 0
    teacher_student_kl: float = float("nan")

    @property
    def key(self) -> ReplayKey:
        return (
            self.spec.num_players,
            self.spec.hand_size,
            self.bidding_position,
            self.phase,
            self.trick_position,
            self.opponent_arm,
        )


@dataclass
class ExpertRoundOutcome:
    spec: RoundSpec
    opponent_arm: OpponentArm
    focal_reward: float
    focal_bid_hit: int
    focal_bid_error: float


@dataclass
class ExpertCycle:
    samples: list[ExpertSample] = field(default_factory=list)
    outcomes: list[ExpertRoundOutcome] = field(default_factory=list)


@dataclass
class ExpertUpdateStats:
    updates: int
    samples: int
    total_loss: float
    policy_loss: float
    q_loss: float
    value_loss: float
    trick_loss: float
    owner_loss: float
    owner_ce_loss: float
    owner_capacity_loss: float
    entropy_floor_loss: float
    grad_norm: float


@dataclass
class ExpertDiagnostics:
    samples: int
    accepted_rate: float
    bid_accepted_rate: float
    play_accepted_rate: float
    split_half_agreement: float
    median_js: float
    mean_nodes: float
    mean_depth: float
    mean_determinizations: float
    leaf_rollout_fraction: float
    teacher_student_kl: float
    q_mae: float
    q_explained_variance: float
    q_rank_correlation: float
    value_mae: float
    value_explained_variance: float
    bid_value_explained_variance: float
    play_value_explained_variance: float
    owner_brier: float
    owner_uniform_brier: float
    owner_belief_weight: float
    bid_leaf_value_weight: float
    play_leaf_value_weight: float
    bid_search_temperature: float
    play_search_temperature: float
    sampler_infeasible_rejection_rate: float
    sampler_failed_draw_rate: float


class ExpertReplay:
    """Age-bounded replay sampled uniformly across deployment strata."""

    def __init__(self, capacity: int = 50_000, max_age: int = 100):
        self.capacity = capacity
        self.max_age = max_age
        self.rows: deque[ExpertSample] = deque()

    def add_many(self, rows: list[ExpertSample]) -> None:
        self.rows.extend(rows)
        while len(self.rows) > self.capacity:
            self.rows.popleft()

    def prune(self, cycle: int) -> None:
        minimum = cycle - self.max_age
        self.rows = deque(
            row for row in self.rows
            if row.cycle >= minimum
        )

    def balanced_sample(
        self,
        rng: random.Random,
        count: int,
        arm_fractions: dict[OpponentArm, float],
    ) -> list[ExpertSample]:
        if not self.rows or count <= 0:
            return []
        strata: dict[ReplayKey, list[ExpertSample]] = defaultdict(list)
        for row in self.rows:
            strata[row.key].append(row)
        configurations = sorted(
            {
                (key[0], key[1])
                for key in strata
            }
        )
        strata_by_cell_arm: dict[
            tuple[int, int, OpponentArm],
            list[ReplayKey],
        ] = defaultdict(list)
        for key in strata:
            strata_by_cell_arm[(key[0], key[1], key[5])].append(key)
        for keys in strata_by_cell_arm.values():
            rng.shuffle(keys)
        cursors: dict[tuple[int, int, OpponentArm], int] = defaultdict(int)
        configuration_sequence = [
            configurations[index % len(configurations)]
            for index in range(count)
        ]
        configuration_counts: dict[tuple[int, int], int] = defaultdict(int)
        for configuration in configuration_sequence:
            configuration_counts[configuration] += 1
        arm_schedules: dict[
            tuple[int, int],
            list[OpponentArm],
        ] = {}
        for players, hand_size in configurations:
            available = {
                arm
                for arm in ("self", "heuristic", "mixed", "historical")
                if (players, hand_size, arm) in strata_by_cell_arm
            }
            cell_fractions = dict(arm_fractions)
            if "historical" not in available and "self" in available:
                cell_fractions["self"] += cell_fractions["historical"]
                cell_fractions["historical"] = 0.0
            for arm in cell_fractions:
                if arm not in available:
                    cell_fractions[arm] = 0.0
            total = sum(cell_fractions.values())
            if total <= 0.0:
                cell_fractions = {
                    arm: float(arm in available) / len(available)
                    for arm in cell_fractions
                }
            else:
                cell_fractions = {
                    arm: value / total
                    for arm, value in cell_fractions.items()
                }
            arm_schedules[(players, hand_size)] = allocate_opponent_arms(
                configuration_counts[(players, hand_size)],
                cell_fractions,
                rng,
            )
        arm_cursors: dict[tuple[int, int], int] = defaultdict(int)
        selected = []
        for players, hand_size in configuration_sequence:
            schedule = arm_schedules[(players, hand_size)]
            arm = schedule[arm_cursors[(players, hand_size)]]
            arm_cursors[(players, hand_size)] += 1
            cell_arm = (players, hand_size, arm)
            keys = strata_by_cell_arm[cell_arm]
            key = keys[cursors[cell_arm] % len(keys)]
            cursors[cell_arm] += 1
            selected.append(rng.choice(strata[key]))
        return selected

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            "capacity": self.capacity,
            "max_age": self.max_age,
            "rows": list(self.rows),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wb", compresslevel=3) as file:
            pickle.dump(
                self.state_dict(),
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ExpertReplay":
        with gzip.open(Path(path), "rb") as file:
            state = pickle.load(file)
        return cls.from_state_dict(state)

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ExpertReplay":
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Expert replay is not schema v5.")
        if state.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("Expert replay rules fingerprint does not match.")
        replay = cls(int(state["capacity"]), int(state["max_age"]))
        replay.rows.extend(state["rows"])
        return replay

    def __len__(self) -> int:
        return len(self.rows)


@dataclass
class SearchScheduleState:
    owner_ready_history: deque[bool] = field(
        default_factory=lambda: deque(maxlen=3)
    )
    owner_ramp_cycles: int = 0
    phase_ev_history: dict[SamplePhase, deque[float]] = field(
        default_factory=lambda: {
            "bid": deque(maxlen=3),
            "play": deque(maxlen=3),
        }
    )
    phase_stable_history: dict[SamplePhase, deque[bool]] = field(
        default_factory=lambda: {
            "bid": deque(maxlen=3),
            "play": deque(maxlen=3),
        }
    )
    phase_leaf_ramp_cycles: dict[SamplePhase, int] = field(
        default_factory=lambda: {"bid": 0, "play": 0}
    )
    phase_temperature_ramp_cycles: dict[SamplePhase, int] = field(
        default_factory=lambda: {"bid": 0, "play": 0}
    )

    def owner_weight(self) -> float:
        return min(1.0, self.owner_ramp_cycles / 200.0)

    def leaf_weight(self, phase: SamplePhase) -> float:
        return min(1.0, self.phase_leaf_ramp_cycles[phase] / 200.0)

    def temperature(self, phase: SamplePhase) -> float:
        fraction = min(
            1.0,
            self.phase_temperature_ramp_cycles[phase] / 500.0,
        )
        return 2.0 + (0.75 - 2.0) * fraction

    def update(self, diagnostics: ExpertDiagnostics) -> None:
        owner_ready = (
            diagnostics.owner_uniform_brier > 0.0
            and diagnostics.owner_brier
            <= 0.95 * diagnostics.owner_uniform_brier
        )
        self.owner_ready_history.append(owner_ready)
        if (
            len(self.owner_ready_history) == 3
            and all(self.owner_ready_history)
        ):
            self.owner_ramp_cycles += 1
        for phase, ev, accepted in (
            (
                "bid",
                diagnostics.bid_value_explained_variance,
                diagnostics.bid_accepted_rate,
            ),
            (
                "play",
                diagnostics.play_value_explained_variance,
                diagnostics.play_accepted_rate,
            ),
        ):
            self.phase_ev_history[phase].append(ev)
            ev_ready = (
                len(self.phase_ev_history[phase]) == 3
                and all(value >= 0.30 for value in self.phase_ev_history[phase])
            )
            if ev_ready:
                self.phase_leaf_ramp_cycles[phase] += 1
            self.phase_stable_history[phase].append(
                ev_ready and accepted >= 0.80
            )
            if (
                len(self.phase_stable_history[phase]) == 3
                and all(self.phase_stable_history[phase])
            ):
                self.phase_temperature_ramp_cycles[phase] += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "owner_ready_history": list(self.owner_ready_history),
            "owner_ramp_cycles": self.owner_ramp_cycles,
            "phase_ev_history": {
                phase: list(values)
                for phase, values in self.phase_ev_history.items()
            },
            "phase_stable_history": {
                phase: list(values)
                for phase, values in self.phase_stable_history.items()
            },
            "phase_leaf_ramp_cycles": self.phase_leaf_ramp_cycles,
            "phase_temperature_ramp_cycles": (
                self.phase_temperature_ramp_cycles
            ),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.owner_ready_history.extend(state["owner_ready_history"])
        self.owner_ramp_cycles = int(state["owner_ramp_cycles"])
        for phase in ("bid", "play"):
            self.phase_ev_history[phase].extend(
                state["phase_ev_history"][phase]
            )
            self.phase_stable_history[phase].extend(
                state["phase_stable_history"][phase]
            )
        self.phase_leaf_ramp_cycles = {
            phase: int(value)
            for phase, value in state[
                "phase_leaf_ramp_cycles"
            ].items()
        }
        self.phase_temperature_ramp_cycles = {
            phase: int(value)
            for phase, value in state[
                "phase_temperature_ramp_cycles"
            ].items()
        }


@dataclass
class _ActiveExpertEpisode:
    """One in-flight episode advanced in lockstep with its peers."""

    env: PlumpEnv
    spec: RoundSpec
    episode_id: int
    arm: OpponentArm
    focal_player: int
    opponents: dict[int, ActionPolicy | None]
    rows: list[ExpertSample] = field(default_factory=list)


class ExpertIterationTrainer:
    """Generate search-improved play and distill it without policy gradients."""

    def __init__(
        self,
        model: PlumpSearchModel | None = None,
        config: ExpertIterationConfig | None = None,
    ) -> None:
        self.config = config or ExpertIterationConfig()
        self._validate_config()
        self.device = (
            torch.device(self.config.device)
            if self.config.device
            else best_torch_device()
        )
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.rng = random.Random(self.config.seed)
        self.model = model or PlumpSearchModel(self.config.model_config)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.position_baseline = PositionBaseline(
            self.config.position_baseline_decay
        )
        self.replay = ExpertReplay(
            self.config.replay_capacity,
            self.config.replay_max_age,
        )
        self.search_schedule = SearchScheduleState()
        self.heuristic_policy = HeuristicPolicy()
        self.historical_paths = list(
            self.config.historical_checkpoint_paths
        )
        self.historical_policies = [
            self._load_v5_policy(path)
            for path in self.historical_paths
        ]
        self.sampler_counters = [0, 0, 0, 0, 0]

    def collect_cycle(
        self,
        *,
        cycle: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ExpertCycle:
        frozen_model = copy.deepcopy(self.model).to(self.device).eval()
        frozen_policy = ModelPolicy(
            frozen_model,
            device=self.device,
            greedy=False,
            include_game_context=(
                self.config.training_mode == "game"
            ),
            precision=self.config.precision,
            name=f"v5-cycle-{cycle}",
        )
        with torch.no_grad(), model_autocast(
            self.device,
            self.config.precision,
        ):
            if self.config.training_mode == "game":
                return self._collect_games(
                    cycle,
                    frozen_policy,
                    progress_callback,
                )
            return self._collect_round_cycle(
                cycle,
                frozen_policy,
                progress_callback,
            )

    def _collect_round_cycle(
        self,
        cycle: int,
        frozen_policy: ModelPolicy,
        progress_callback: Callable[[int, int], None] | None,
    ) -> ExpertCycle:
        result = ExpertCycle()
        baseline_updates: list[tuple[tuple[int, int, int], float]] = []
        schedule: list[tuple[RoundSpec, OpponentArm, int, int]] = []
        for spec in self.config.specs:
            fractions = self.config.opponent_mix.effective(
                has_history=bool(self.historical_policies)
            )
            arms = allocate_opponent_arms(
                self.config.rounds_per_configuration,
                fractions,
                self.rng,
            )
            for repetition, arm in enumerate(arms):
                focal_player = (
                    cycle - 1 + repetition
                ) % spec.num_players
                bidding_position = repetition % spec.num_players
                bidding_start = (
                    focal_player - bidding_position
                ) % spec.num_players
                schedule.append((spec, arm, focal_player, bidding_start))

        total_episodes = len(schedule)
        completed = 0
        next_index = 0
        active: list[_ActiveExpertEpisode] = []

        def _start(index: int) -> _ActiveExpertEpisode:
            spec, arm, focal_player, bidding_start = schedule[index]
            env = PlumpEnv(
                round_game_config(
                    spec,
                    bidding_start_player=bidding_start,
                ),
                seed=self.rng.randrange(2**31),
            )
            env.reset()
            return _ActiveExpertEpisode(
                env=env,
                spec=spec,
                episode_id=index,
                arm=arm,
                focal_player=focal_player,
                opponents=self._opponent_policies(
                    spec.num_players,
                    focal_player,
                    arm,
                ),
            )

        while (
            next_index < total_episodes
            and len(active) < self.config.concurrent_episodes
        ):
            active.append(_start(next_index))
            next_index += 1

        while active:
            self._advance_episodes_lockstep(
                active,
                cycle=cycle,
                frozen_policy=frozen_policy,
                search_enabled=True,
            )
            remaining: list[_ActiveExpertEpisode] = []
            for episode in active:
                if not episode.env.is_done():
                    remaining.append(episode)
                    continue
                round_state = episode.env.state.current_round
                rewards = compute_relative_rewards(
                    round_state.round_scores
                )
                focal_reward = rewards[episode.focal_player]
                self._finish_episode_rows(
                    episode.rows,
                    episode.env,
                    focal_reward,
                )
                result.samples.extend(episode.rows)
                focal_bid = next(
                    bid
                    for bid in round_state.bids
                    if bid.player == episode.focal_player
                )
                result.outcomes.append(
                    ExpertRoundOutcome(
                        spec=episode.spec,
                        opponent_arm=episode.arm,
                        focal_reward=focal_reward,
                        focal_bid_hit=int(
                            round_state.tricks_won[episode.focal_player]
                            == focal_bid.value
                        ),
                        focal_bid_error=float(
                            abs(
                                round_state.tricks_won[episode.focal_player]
                                - focal_bid.value
                            )
                        ),
                    )
                )
                baseline_updates.append(
                    (
                        position_key(episode.spec, focal_bid.position),
                        focal_reward,
                    )
                )
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_episodes)
                if next_index < total_episodes:
                    remaining.append(_start(next_index))
                    next_index += 1
            active = remaining
        self.position_baseline.update_many(baseline_updates)
        return result

    def _collect_games(
        self,
        cycle: int,
        frozen_policy: ModelPolicy,
        progress_callback: Callable[[int, int], None] | None,
    ) -> ExpertCycle:
        result = ExpertCycle()
        game_schedule = self.config.resolved_game_schedule
        fractions = self.config.opponent_mix.effective(
            has_history=bool(self.historical_policies)
        )
        schedule: list[tuple[int, int, OpponentArm]] = []
        for num_players in self.config.player_counts:
            for focal_player in range(num_players):
                for arm in allocate_opponent_arms(
                    self.config.games_per_player_seat,
                    fractions,
                    self.rng,
                ):
                    schedule.append((num_players, focal_player, arm))

        total_episodes = len(schedule)
        completed = 0
        next_index = 0
        active: list[_ActiveExpertEpisode] = []

        def _start(index: int) -> _ActiveExpertEpisode:
            num_players, focal_player, arm = schedule[index]
            env = PlumpEnv(
                GameConfig(
                    num_players=num_players,
                    hand_sizes=list(game_schedule),
                ),
                seed=self.rng.randrange(2**31),
            )
            env.reset()
            return _ActiveExpertEpisode(
                env=env,
                spec=RoundSpec(num_players, game_schedule[0]),
                episode_id=index,
                arm=arm,
                focal_player=focal_player,
                opponents=self._opponent_policies(
                    num_players,
                    focal_player,
                    arm,
                ),
            )

        while (
            next_index < total_episodes
            and len(active) < self.config.concurrent_episodes
        ):
            active.append(_start(next_index))
            next_index += 1

        while active:
            self._advance_episodes_lockstep(
                active,
                cycle=cycle,
                frozen_policy=frozen_policy,
                search_enabled=False,
            )
            remaining: list[_ActiveExpertEpisode] = []
            for episode in active:
                if not episode.env.is_done():
                    remaining.append(episode)
                    continue
                focal_player = episode.focal_player
                reward = compute_relative_rewards(
                    episode.env.state.cumulative_scores
                )[focal_player]
                self._finish_episode_rows(
                    episode.rows,
                    episode.env,
                    reward,
                )
                result.samples.extend(episode.rows)
                for round_state in episode.env.state.rounds:
                    round_rewards = compute_relative_rewards(
                        round_state.round_scores
                    )
                    focal_bid = next(
                        bid for bid in round_state.bids
                        if bid.player == focal_player
                    )
                    result.outcomes.append(
                        ExpertRoundOutcome(
                            spec=RoundSpec(
                                episode.env.config.num_players,
                                round_state.hand_size,
                            ),
                            opponent_arm=episode.arm,
                            focal_reward=round_rewards[focal_player],
                            focal_bid_hit=int(
                                round_state.tricks_won[focal_player]
                                == focal_bid.value
                            ),
                            focal_bid_error=float(
                                abs(
                                    round_state.tricks_won[focal_player]
                                    - focal_bid.value
                                )
                            ),
                        )
                    )
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_episodes)
                if next_index < total_episodes:
                    remaining.append(_start(next_index))
                    next_index += 1
            active = remaining
        return result

    def _advance_episodes_lockstep(
        self,
        episodes: list[_ActiveExpertEpisode],
        *,
        cycle: int,
        frozen_policy: ModelPolicy,
        search_enabled: bool,
    ) -> None:
        """Advance every active episode by exactly one action.

        Frozen-model turns are grouped and inferred as one batch per policy;
        searches stay per-decision but batch internally across worlds.
        """

        actions: dict[int, BidAction | PlayCardAction] = {}
        model_groups: dict[
            int,
            tuple[ModelPolicy, list[_ActiveExpertEpisode]],
        ] = {}
        focal_turns: list[_ActiveExpertEpisode] = []
        for episode in episodes:
            player = episode.env.current_player()
            if player == episode.focal_player:
                focal_turns.append(episode)
                continue
            policy = episode.opponents.get(player) or frozen_policy
            if isinstance(policy, ModelPolicy):
                group = model_groups.setdefault(id(policy), (policy, []))
                group[1].append(episode)
            else:
                actions[episode.episode_id] = policy.act(
                    episode.env,
                    rng=self.rng,
                )

        for policy, group in model_groups.values():
            selected = policy.act_many(
                [episode.env for episode in group],
                rngs=[self.rng] * len(group),
            )
            for episode, action in zip(group, selected):
                actions[episode.episode_id] = action

        deferred: list[tuple[_ActiveExpertEpisode, ExpertSample]] = []
        for episode in focal_turns:
            row, action = self._focal_decision(
                episode,
                cycle=cycle,
                frozen_policy=frozen_policy,
                search_enabled=search_enabled,
            )
            if action is None:
                deferred.append((episode, row))
                continue
            row.action_index = _action_index(action)
            episode.rows.append(row)
            actions[episode.episode_id] = action

        if deferred:
            selected = frozen_policy.act_many(
                [episode.env for episode, _ in deferred],
                rngs=[self.rng] * len(deferred),
            )
            for (episode, row), action in zip(deferred, selected):
                row.action_index = _action_index(action)
                episode.rows.append(row)
                actions[episode.episode_id] = action

        for episode in episodes:
            episode.env.step(actions[episode.episode_id])

    def _focal_decision(
        self,
        episode: _ActiveExpertEpisode,
        *,
        cycle: int,
        frozen_policy: ModelPolicy,
        search_enabled: bool,
    ) -> tuple[ExpertSample, BidAction | PlayCardAction | None]:
        """Build the focal sample; a None action defers to the batched policy."""

        env = episode.env
        player = episode.focal_player
        observation = env.get_observation(player)
        encoded = encode_observation(
            observation,
            self.config.model_config,
            include_game_context=(
                self.config.training_mode == "game"
            ),
        )
        legal = env.legal_actions()
        phase: SamplePhase = (
            "bid"
            if env.phase() == Phase.BIDDING
            else "play"
        )
        current_spec = RoundSpec(
            env.config.num_players,
            env.state.current_round.hand_size,
        )
        intercept = (
            0.0
            if self.config.training_mode == "game"
            else self.position_baseline.get(
                position_key(
                    current_spec,
                    encoded.bidding_position,
                )
            )
        )
        row = ExpertSample(
            encoded=encoded,
            phase=phase,
            action_index=-1,
            spec=current_spec,
            bidding_position=encoded.bidding_position,
            trick_position=(
                len(observation.current_trick.plays)
                if observation.current_trick is not None
                else -1
            ),
            opponent_arm=episode.arm,
            cycle=cycle,
            episode_id=episode.episode_id,
            acting_player=player,
            round_id=env.state.current_round.round_index,
            position_intercept=intercept,
            owner_targets=owner_targets_relative(
                env,
                player,
                encoded.owner_valid_mask,
                self.config.model_config,
            ),
            forced=len(legal) == 1,
        )
        if len(legal) == 1:
            return row, legal[0]
        search_this_decision = search_enabled and (
            phase == "bid"
            or self.config.play_search_fraction >= 1.0
            or self.rng.random() < self.config.play_search_fraction
        )
        if not search_this_decision:
            return row, None
        opponents = episode.opponents
        phase_temperature = self.search_schedule.temperature(phase)
        search = InformationSetSearch(
            frozen_policy,
            opponent_policy_for_player=lambda opponent: (
                opponents.get(opponent) or frozen_policy
            ),
            value_intercept=lambda obs: self.position_baseline.get(
                (
                    len(obs.scores),
                    obs.hand_size,
                    _bidding_position(obs),
                )
            ),
            config=replace(
                self.config.search_config,
                internal_temperature=phase_temperature,
                target_temperature=phase_temperature,
                seed=self.rng.randrange(2**31),
            ),
            belief_weight=self.search_schedule.owner_weight(),
            leaf_value_weight=self.search_schedule.leaf_weight(phase),
        )
        before = search.sampler.counters()
        decision = search.search(
            observation,
            legal_actions=legal,
            rng=random.Random(self.rng.randrange(2**31)),
        )
        after = search.sampler.counters()
        self.sampler_counters = [
            total + end - start
            for total, start, end in zip(
                self.sampler_counters,
                before,
                after,
            )
        ]
        self._attach_search(row, decision)
        if decision.accepted:
            return row, self._sample_search_behavior(
                legal,
                decision,
                cycle,
            )
        return row, None

    def _attach_search(
        self,
        row: ExpertSample,
        decision: InformationSearchDecision,
    ) -> None:
        size = (
            self.config.model_config.bid_count
            if row.phase == "bid"
            else NUM_CARDS
        )
        policy = [0.0] * size
        q_targets = [0.0] * size
        q_stderr = [0.0] * size
        for key, probability in decision.action_probabilities.items():
            policy[int(key.split(":", 1)[1])] = probability
        for key, value in decision.action_values.items():
            q_targets[int(key.split(":", 1)[1])] = (
                value - row.position_intercept
            )
        for key, value in decision.action_stderr.items():
            q_stderr[int(key.split(":", 1)[1])] = value
        row.search_policy = policy
        row.q_targets = q_targets
        row.q_stderr = q_stderr
        row.accepted = decision.accepted
        row.search_js = decision.target_js_divergence
        row.search_agreement = (
            decision.split_half_argmax_agreement
        )
        row.search_nodes = decision.expanded_nodes
        row.search_depth = decision.maximum_depth
        row.search_determinizations = decision.determinizations
        row.search_leaf_rollouts = decision.leaf_rollouts
        row.search_leaf_values = decision.leaf_value_predictions
        row.teacher_student_kl = _distribution_kl(
            decision.action_probabilities,
            decision.prior_probabilities,
        )

    def _sample_search_behavior(
        self,
        legal: list[BidAction | PlayCardAction],
        decision: InformationSearchDecision,
        cycle: int,
    ) -> BidAction | PlayCardAction:
        epsilon = 0.05 + (0.01 - 0.05) * min(cycle / 1000.0, 1.0)
        uniform = 1.0 / len(legal)
        weights = [
            (
                (1.0 - epsilon)
                * decision.action_probabilities[_action_key(action)]
                + epsilon * uniform
            )
            for action in legal
        ]
        return self.rng.choices(legal, weights=weights, k=1)[0]

    def _finish_episode_rows(
        self,
        rows: list[ExpertSample],
        env: PlumpEnv,
        terminal_reward: float,
    ) -> None:
        rounds = {
            round_state.round_index: round_state
            for round_state in env.state.rounds
        }
        for row in rows:
            round_state = rounds[row.round_id]
            row.target_value = (
                terminal_reward - row.position_intercept
            )
            row.final_trick_targets = final_tricks_relative(
                round_state.tricks_won,
                row.acting_player,
                env.config.num_players,
                self.config.model_config,
            )
            row.final_bid_targets = final_bids_relative(
                round_state.bids,
                row.acting_player,
                env.config.num_players,
                self.config.model_config,
            )

    def add_cycle(self, cycle: ExpertCycle, *, cycle_index: int) -> None:
        self.replay.prune(cycle_index)
        self.replay.add_many(cycle.samples)

    def update(
        self,
        *,
        new_state_count: int,
    ) -> ExpertUpdateStats:
        update_count = max(
            1,
            math.ceil(
                self.config.updates_per_new_state
                * new_state_count
            ),
        )
        totals: dict[str, list[float]] = defaultdict(list)
        self.model.train()
        for _ in range(update_count):
            rows = self.replay.balanced_sample(
                self.rng,
                self.config.minibatch_size,
                self.config.opponent_mix.effective(
                    has_history=bool(self.historical_policies)
                ),
            )
            if not rows:
                raise RuntimeError("Cannot update from empty expert replay.")
            staged = encoded_observations_to_batch(
                [row.encoded for row in rows],
                device=self.device,
            )
            logical_count = len(rows)
            accepted_count = max(sum(row.accepted for row in rows), 1)
            q_label_count = max(
                sum(
                    probability > 0.0
                    for row in rows
                    if row.accepted and row.search_policy is not None
                    for probability in row.search_policy
                ),
                1,
            )
            trick_label_count = max(
                sum(
                    target != -100
                    for row in rows
                    for target in row.final_trick_targets
                ),
                1,
            )
            owner_label_count = max(
                sum(
                    target != -100
                    for row in rows
                    for target in row.owner_targets
                ),
                1,
            )
            capacity_count = max(
                sum(
                    value > 0
                    for row in rows
                    for value in row.encoded.owner_capacities
                ),
                1,
            )
            self.optimizer.zero_grad(set_to_none=True)
            step_totals: dict[str, float] = defaultdict(float)
            for start in range(
                0,
                logical_count,
                self.config.microbatch_size,
            ):
                indices = list(
                    range(
                        start,
                        min(
                            start + self.config.microbatch_size,
                            logical_count,
                        ),
                    )
                )
                selected = torch.tensor(
                    indices,
                    dtype=torch.long,
                    device=self.device,
                )
                mb_rows = [rows[index] for index in indices]
                batch = index_model_batch(staged, selected)
                with model_autocast(
                    self.device,
                    self.config.precision,
                ):
                    output = self.model(batch)
                losses = self._microbatch_losses(
                    output,
                    batch,
                    mb_rows,
                    logical_count=logical_count,
                    accepted_count=accepted_count,
                    q_label_count=q_label_count,
                    trick_label_count=trick_label_count,
                    owner_label_count=owner_label_count,
                    capacity_count=capacity_count,
                )
                losses["total"].backward()
                for name, value in losses.items():
                    step_totals[name] += float(value.detach().cpu())
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                ).cpu()
            )
            self.optimizer.step()
            step_totals["grad_norm"] = grad_norm
            for name, value in step_totals.items():
                totals[name].append(value)
        return ExpertUpdateStats(
            updates=update_count,
            samples=update_count * self.config.minibatch_size,
            total_loss=_mean(totals["total"]),
            policy_loss=_mean(totals["policy"]),
            q_loss=_mean(totals["q"]),
            value_loss=_mean(totals["value"]),
            trick_loss=_mean(totals["trick"]),
            owner_loss=_mean(totals["owner"]),
            owner_ce_loss=_mean(totals["owner_ce"]),
            owner_capacity_loss=_mean(totals["owner_capacity"]),
            entropy_floor_loss=_mean(totals["entropy_floor"]),
            grad_norm=_mean(totals["grad_norm"]),
        )

    def _microbatch_losses(
        self,
        output,
        batch,
        rows: list[ExpertSample],
        *,
        logical_count: int,
        accepted_count: int,
        q_label_count: int,
        trick_label_count: int,
        owner_label_count: int,
        capacity_count: int,
    ) -> dict[str, Tensor]:
        device = self.device
        accepted = torch.tensor(
            [row.accepted for row in rows],
            dtype=torch.bool,
            device=device,
        )
        bid_rows = torch.tensor(
            [row.phase == "bid" for row in rows],
            dtype=torch.bool,
            device=device,
        )
        policy_loss = torch.zeros((), device=device)
        q_loss = torch.zeros((), device=device)
        entropy_floor = torch.zeros((), device=device)
        for phase_mask, logits, q_values, size in (
            (
                bid_rows,
                output.masked_bid_logits.float(),
                output.masked_bid_q_values.float(),
                self.config.model_config.bid_count,
            ),
            (
                ~bid_rows,
                output.masked_card_logits.float(),
                output.masked_card_q_values.float(),
                NUM_CARDS,
            ),
        ):
            active = phase_mask & accepted
            if not active.any():
                continue
            targets = torch.tensor(
                [
                    (
                        row.search_policy
                        if (
                            row.search_policy is not None
                            and len(row.search_policy) == size
                        )
                        else [0.0] * size
                    )
                    for row in rows
                ],
                dtype=torch.float32,
                device=device,
            )
            target_q = torch.tensor(
                [
                    (
                        row.q_targets
                        if (
                            row.q_targets is not None
                            and len(row.q_targets) == size
                        )
                        else [0.0] * size
                    )
                    for row in rows
                ],
                dtype=torch.float32,
                device=device,
            )
            stderr = torch.tensor(
                [
                    (
                        row.q_stderr
                        if (
                            row.q_stderr is not None
                            and len(row.q_stderr) == size
                        )
                        else [0.0] * size
                    )
                    for row in rows
                ],
                dtype=torch.float32,
                device=device,
            )
            legal = targets > 0.0
            log_probs = torch.log_softmax(logits, dim=-1)
            policy_loss = policy_loss + (
                -(targets * log_probs).sum(dim=-1)[active].sum()
                / accepted_count
            )
            confidence = 1.0 / (1.0 + stderr.pow(2))
            q_terms = F.smooth_l1_loss(
                q_values,
                target_q,
                reduction="none",
            )
            active_legal = active.unsqueeze(-1) & legal
            q_loss = q_loss + (
                (q_terms * confidence)[active_legal].sum()
                / q_label_count
            )
            probabilities = torch.softmax(logits, dim=-1)
            entropy = -(
                probabilities
                * torch.log(probabilities.clamp_min(1e-12))
            ).sum(dim=-1)
            target_entropy = -(
                targets
                * torch.log(targets.clamp_min(1e-12))
            ).sum(dim=-1)
            entropy_floor = entropy_floor + (
                torch.relu(target_entropy - entropy)[active].sum()
                / accepted_count
            )

        value_targets = torch.tensor(
            [row.target_value for row in rows],
            dtype=torch.float32,
            device=device,
        )
        value_loss = F.smooth_l1_loss(
            output.value.squeeze(-1).float(),
            value_targets,
            reduction="sum",
        ) / logical_count
        trick_targets = torch.tensor(
            [row.final_trick_targets for row in rows],
            dtype=torch.long,
            device=device,
        )
        trick_loss = F.cross_entropy(
            output.masked_trick_count_logits.float().reshape(
                -1,
                self.config.model_config.bid_count,
            ),
            trick_targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / trick_label_count
        owner_targets = torch.tensor(
            [row.owner_targets for row in rows],
            dtype=torch.long,
            device=device,
        )
        active_owner = owner_targets != -100
        safe_owner = owner_targets.clamp_min(0)
        true_probability = output.owner_probs.gather(
            -1,
            safe_owner.unsqueeze(-1),
        ).squeeze(-1)
        owner_ce = -torch.log(
            true_probability.clamp_min(1e-12)
        )[active_owner].sum() / owner_label_count
        capacities = batch.owner_capacities.float()
        active_capacity = capacities > 0.0
        hidden_count = capacities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        owner_capacity = (
            (
                (
                    output.owner_pre_sinkhorn_probs.sum(dim=1)
                    - capacities
                )
                / hidden_count
            ).pow(2)[active_capacity].sum()
            / capacity_count
        )
        owner_loss = (
            owner_ce
            + self.config.owner_capacity_coef * owner_capacity
        )
        total = (
            self.config.policy_coef * policy_loss
            + self.config.q_coef * q_loss
            + self.config.value_coef * value_loss
            + self.config.trick_coef * trick_loss
            + self.config.owner_coef * owner_loss
            + self.config.entropy_floor_coef * entropy_floor
        )
        return {
            "total": total,
            "policy": policy_loss,
            "q": q_loss,
            "value": value_loss,
            "trick": trick_loss,
            "owner": owner_loss,
            "owner_ce": owner_ce,
            "owner_capacity": owner_capacity,
            "entropy_floor": entropy_floor,
        }

    def diagnostics(
        self,
        cycle: ExpertCycle,
        *,
        max_samples: int = 2_048,
        batch_size: int = 256,
    ) -> ExpertDiagnostics:
        rows = _evenly_spaced(cycle.samples, max_samples)
        values: list[float] = []
        value_targets: list[float] = []
        bid_values: list[float] = []
        bid_targets: list[float] = []
        play_values: list[float] = []
        play_targets: list[float] = []
        q_predictions: list[float] = []
        q_targets: list[float] = []
        q_rank_rows: list[float] = []
        owner_briers: list[float] = []
        uniform_briers: list[float] = []
        teacher_kls = [
            row.teacher_student_kl
            for row in rows
            if math.isfinite(row.teacher_student_kl)
        ]
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                mb = rows[start : start + batch_size]
                batch = encoded_observations_to_batch(
                    [row.encoded for row in mb],
                    device=self.device,
                )
                with model_autocast(
                    self.device,
                    self.config.precision,
                ):
                    output = self.model(batch)
                full_values = (
                    output.value.squeeze(-1).float()
                    + torch.tensor(
                        [row.position_intercept for row in mb],
                        device=self.device,
                    )
                ).cpu().tolist()
                for row, value in zip(mb, full_values):
                    target = (
                        float(row.target_value)
                        + row.position_intercept
                    )
                    values.append(value)
                    value_targets.append(target)
                    if row.phase == "bid":
                        bid_values.append(value)
                        bid_targets.append(target)
                    else:
                        play_values.append(value)
                        play_targets.append(target)
                for index, row in enumerate(mb):
                    if row.accepted:
                        predicted = (
                            output.masked_bid_q_values[index].float()
                            if row.phase == "bid"
                            else output.masked_card_q_values[index].float()
                        )
                        target = torch.tensor(
                            row.q_targets,
                            device=self.device,
                        )
                        legal = torch.tensor(
                            row.search_policy,
                            device=self.device,
                        ) > 0.0
                        q_predictions.extend(
                            predicted[legal].cpu().tolist()
                        )
                        q_targets.extend(target[legal].cpu().tolist())
                        q_rank_rows.append(
                            _spearman(
                                predicted[legal].cpu().tolist(),
                                target[legal].cpu().tolist(),
                            )
                        )
                owner_targets = torch.tensor(
                    [row.owner_targets for row in mb],
                    dtype=torch.long,
                    device=self.device,
                )
                active = owner_targets >= 0
                safe = owner_targets.clamp_min(0)
                one_hot = F.one_hot(
                    safe,
                    num_classes=(
                        self.config.model_config.owner_class_count
                    ),
                ).float()
                projected = output.owner_probs.float()
                owner_briers.extend(
                    ((projected - one_hot) ** 2)
                    .sum(dim=-1)[active]
                    .cpu()
                    .tolist()
                )
                uniform = _capacity_aware_uniform(
                    batch.owner_valid_mask,
                    batch.owner_capacities,
                )
                uniform_briers.extend(
                    ((uniform - one_hot) ** 2)
                    .sum(dim=-1)[active]
                    .cpu()
                    .tolist()
                )
        searched = [row for row in rows if not row.forced]
        accepted = [row for row in searched if row.accepted]
        bid_searched = [row for row in searched if row.phase == "bid"]
        play_searched = [row for row in searched if row.phase == "play"]
        leaf_total = sum(
            row.search_leaf_rollouts + row.search_leaf_values
            for row in searched
        )
        draws, _, failed, attempts, rejected = self.sampler_counters
        return ExpertDiagnostics(
            samples=len(rows),
            accepted_rate=len(accepted) / max(len(searched), 1),
            bid_accepted_rate=sum(
                row.accepted for row in bid_searched
            ) / max(len(bid_searched), 1),
            play_accepted_rate=sum(
                row.accepted for row in play_searched
            ) / max(len(play_searched), 1),
            split_half_agreement=_mean(
                [float(row.search_agreement) for row in searched]
            ),
            median_js=_median(
                [
                    row.search_js for row in searched
                    if math.isfinite(row.search_js)
                ]
            ),
            mean_nodes=_mean(
                [float(row.search_nodes) for row in searched]
            ),
            mean_depth=_mean(
                [float(row.search_depth) for row in searched]
            ),
            mean_determinizations=_mean(
                [
                    float(row.search_determinizations)
                    for row in searched
                ]
            ),
            leaf_rollout_fraction=(
                sum(row.search_leaf_rollouts for row in searched)
                / max(leaf_total, 1)
            ),
            teacher_student_kl=_mean(teacher_kls),
            q_mae=_mae(q_predictions, q_targets),
            q_explained_variance=_explained_variance(
                q_predictions,
                q_targets,
            ),
            q_rank_correlation=_mean(q_rank_rows),
            value_mae=_mae(values, value_targets),
            value_explained_variance=_explained_variance(
                values,
                value_targets,
            ),
            bid_value_explained_variance=_explained_variance(
                bid_values,
                bid_targets,
            ),
            play_value_explained_variance=_explained_variance(
                play_values,
                play_targets,
            ),
            owner_brier=_mean(owner_briers),
            owner_uniform_brier=_mean(uniform_briers),
            owner_belief_weight=self.search_schedule.owner_weight(),
            bid_leaf_value_weight=self.search_schedule.leaf_weight(
                "bid"
            ),
            play_leaf_value_weight=self.search_schedule.leaf_weight(
                "play"
            ),
            bid_search_temperature=self.search_schedule.temperature(
                "bid"
            ),
            play_search_temperature=self.search_schedule.temperature(
                "play"
            ),
            sampler_infeasible_rejection_rate=(
                rejected / attempts if attempts else 0.0
            ),
            sampler_failed_draw_rate=(
                failed / draws if draws else 0.0
            ),
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        cycle: int,
        extra: dict | None = None,
    ) -> None:
        path = Path(path)
        replay_path = path.with_name(
            f"{path.stem}_replay.pkl.gz"
        )
        self.replay.save(replay_path)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "training_algorithm": "information_set_expert_iteration",
            "cycle": cycle,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": asdict(self.config.model_config),
            "training_config": asdict(self.config),
            "position_baseline": self.position_baseline.state_dict(),
            "replay_path": replay_path.name,
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "historical_checkpoint_paths": self.historical_paths,
            "search_schedule": self.search_schedule.state_dict(),
            "sampler_counters": self.sampler_counters,
            "include_game_context": (
                self.config.training_mode == "game"
            ),
            "precision": self.config.precision,
            "rules_fingerprint": rules_fingerprint(),
            "extra": extra or {},
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
        load_replay: bool = True,
    ) -> dict[str, object]:
        payload = _torch_load(path, self.device)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Only schema-v5 expert-iteration checkpoints can resume "
                "schema-v5 training."
            )
        if payload.get("training_algorithm") != (
            "information_set_expert_iteration"
        ):
            raise ValueError("Checkpoint training algorithm does not match.")
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("Checkpoint rules fingerprint does not match.")
        self.model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )
        if load_optimizer:
            self.optimizer.load_state_dict(
                payload["optimizer_state_dict"]
            )
        if load_replay:
            replay_reference = payload.get("replay_path")
            if replay_reference is None:
                self.replay = ExpertReplay.from_state_dict(
                    payload["replay"]
                )
            else:
                replay_path = Path(path).parent / replay_reference
                self.replay = ExpertReplay.load(replay_path)
        self.position_baseline.load_state_dict(
            payload["position_baseline"]
        )
        self.rng.setstate(payload["rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        self.search_schedule.load_state_dict(
            payload["search_schedule"]
        )
        self.sampler_counters = list(
            payload.get("sampler_counters", [0, 0, 0, 0, 0])
        )
        self.historical_paths = list(
            payload.get("historical_checkpoint_paths", [])
        )
        self.historical_policies = [
            self._load_v5_policy(item)
            for item in self.historical_paths
            if Path(item).exists()
        ]
        return {
            "path": str(path),
            "cycle": int(payload["cycle"]),
            "optimizer_loaded": load_optimizer,
            "replay_loaded": load_replay,
        }

    def initialize_from_v4_checkpoint(
        self,
        path: str | Path,
    ) -> dict[str, object]:
        """Warm-start the v5 trunk/heads from a schema-v4 PPO checkpoint."""

        payload = _torch_load(path, self.device)
        if payload.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                "V4 initialization requires a schema-v4 PPO checkpoint."
            )
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError(
                "V4 checkpoint rules fingerprint does not match."
            )
        migration = load_v4_weights(
            self.model,
            payload["model_state_dict"],
        )
        self.position_baseline.load_state_dict(
            payload.get("position_baseline", {})
        )
        return {
            "path": str(path),
            "source_iteration": int(payload.get("iteration", 0)),
            "loaded_tensors": len(migration["loaded"]),
            "fresh_tensors": migration["fresh"],
            "optimizer_loaded": False,
        }

    def initialize_game_from_round_checkpoint(
        self,
        path: str | Path,
    ) -> dict[str, object]:
        if self.config.training_mode != "game":
            raise ValueError(
                "Round-checkpoint initialization is only for game mode."
            )
        payload = _torch_load(path, self.device)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Game initialization requires schema v5.")
        self.model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )
        self.position_baseline.load_state_dict(
            payload.get("position_baseline", {})
        )
        return {
            "path": str(path),
            "source_cycle": int(payload.get("cycle", 0)),
            "optimizer_loaded": False,
        }

    def add_historical_checkpoint(self, path: str | Path) -> None:
        policy = self._load_v5_policy(path)
        self._append_historical(str(path), policy)

    def add_current_historical_snapshot(
        self,
        path: str | Path,
    ) -> None:
        snapshot = copy.deepcopy(self.model).to(self.device).eval()
        policy = ModelPolicy(
            snapshot,
            device=self.device,
            greedy=False,
            include_game_context=(
                self.config.training_mode == "game"
            ),
            precision=self.config.precision,
            name=Path(path).stem,
        )
        self._append_historical(str(path), policy)

    def _append_historical(
        self,
        path: str,
        policy: ModelPolicy,
    ) -> None:
        self.historical_paths.append(path)
        self.historical_policies.append(policy)
        if (
            len(self.historical_policies)
            > self.config.historical_max_snapshots
        ):
            self.historical_policies = self.historical_policies[
                -self.config.historical_max_snapshots :
            ]
            self.historical_paths = self.historical_paths[
                -self.config.historical_max_snapshots :
            ]

    def _load_v5_policy(self, path: str | Path) -> ModelPolicy:
        payload = _torch_load(path, "cpu")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "The schema-v5 historical league accepts only v5 "
                "expert-iteration checkpoints."
            )
        return ModelPolicy.from_checkpoint(
            path,
            device=self.device,
            greedy=False,
        )

    def _opponent_policies(
        self,
        num_players: int,
        focal_player: int,
        arm: OpponentArm,
    ) -> dict[int, ActionPolicy | None]:
        result: dict[int, ActionPolicy | None] = {}
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
                category = self.rng.choice(
                    ("self", "heuristic", "historical")
                )
                if category == "heuristic":
                    policy = self.heuristic_policy
                elif (
                    category == "historical"
                    and self.historical_policies
                ):
                    policy = self.rng.choice(
                        self.historical_policies
                    )
                else:
                    policy = None
            result[player] = policy
        return result

    def _validate_config(self) -> None:
        if (
            self.config.model_config.schema_version
            != OBSERVATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Schema-v5 training requires unchanged schema-v4 "
                "observations."
            )
        for spec in self.config.specs:
            spec.validate()
        self.config.opponent_mix.validate()
        active_arms = sum(
            value > 0.0
            for value in asdict(self.config.opponent_mix).values()
        )
        if self.config.rounds_per_configuration < active_arms:
            raise ValueError(
                "Each configuration must cover every active opponent arm."
            )
        if self.config.training_mode not in {"round", "game"}:
            raise ValueError("training_mode must be round or game.")
        if self.config.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("Unsupported precision.")
        if self.config.minibatch_size < 1:
            raise ValueError("minibatch_size must be positive.")
        if self.config.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive.")
        if self.config.search_config.node_budget < 1:
            raise ValueError("search node budget must be positive.")
        if self.config.concurrent_episodes < 1:
            raise ValueError("concurrent_episodes must be positive.")
        if not 0.0 <= self.config.play_search_fraction <= 1.0:
            raise ValueError("play_search_fraction must be in [0, 1].")


def _capacity_aware_uniform(
    valid_mask: Tensor,
    capacities: Tensor,
) -> Tensor:
    weights = (
        valid_mask.float()
        * capacities.unsqueeze(1).float()
    )
    return weights / weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)


def _bidding_position(observation) -> int:
    for position, player in enumerate(observation.bidding_order):
        if player == observation.player_id:
            return position
    return 0


def _action_index(action: BidAction | PlayCardAction) -> int:
    if isinstance(action, BidAction):
        return action.bid
    from plump.modeling import card_id
    return card_id(action.card)


def _action_key(action: BidAction | PlayCardAction) -> str:
    prefix = "bid" if isinstance(action, BidAction) else "card"
    return f"{prefix}:{_action_index(action)}"


def _distribution_kl(
    target: dict[str, float],
    prior: dict[str, float],
) -> float:
    return sum(
        probability
        * math.log(
            probability / max(prior[key], 1e-12)
        )
        for key, probability in target.items()
        if probability > 0.0
    )


def _evenly_spaced(
    rows: list[ExpertSample],
    maximum: int,
) -> list[ExpertSample]:
    if len(rows) <= maximum:
        return list(rows)
    stride = len(rows) / maximum
    return [
        rows[min(int(index * stride), len(rows) - 1)]
        for index in range(maximum)
    ]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _mae(values: list[float], targets: list[float]) -> float:
    return _mean(
        abs(value - target)
        for value, target in zip(values, targets)
    )


def _explained_variance(
    values: list[float],
    targets: list[float],
) -> float:
    if not targets:
        return 0.0
    target_mean = _mean(targets)
    target_variance = _mean(
        (target - target_mean) ** 2
        for target in targets
    )
    if target_variance <= 1e-12:
        return 0.0
    errors = [
        target - value
        for value, target in zip(values, targets)
    ]
    error_mean = _mean(errors)
    error_variance = _mean(
        (error - error_mean) ** 2
        for error in errors
    )
    return 1.0 - error_variance / target_variance


def _spearman(first: list[float], second: list[float]) -> float:
    if len(first) < 2:
        return 0.0
    first_ranks = _ranks(first)
    second_ranks = _ranks(second)
    first_mean = _mean(first_ranks)
    second_mean = _mean(second_ranks)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first_ranks, second_ranks)
    )
    denominator = math.sqrt(
        sum((value - first_mean) ** 2 for value in first_ranks)
        * sum((value - second_mean) ** 2 for value in second_ranks)
    )
    return numerator / denominator if denominator > 1e-12 else 0.0


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(ordered):
        ranks[index] = float(rank)
    return ranks


def _torch_load(path: str | Path, device) -> dict:
    try:
        return torch.load(
            Path(path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:  # pragma: no cover
        return torch.load(Path(path), map_location=device)
