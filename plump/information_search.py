"""Information-set-consistent tree search for schema-v5 expert iteration."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Callable

from plump.env import PlumpEnv
from plump.modeling import ModelConfig, card_id
from plump.policies import ActionPolicy, ModelPolicy
from plump.search import (
    ConstrainedDeterminizationSampler,
    Determinization,
    reconstruct_public_state,
)
from plump.state import (
    BidAction,
    Observation,
    Phase,
    PlayCardAction,
    ScoringConfig,
    Trick,
)


Action = BidAction | PlayCardAction


@dataclass(frozen=True)
class InformationSearchConfig:
    min_determinizations: int = 4
    max_determinizations: int = 32
    batch_determinizations: int = 4
    maximum_target_js: float = 0.05
    close_action_stderr: float = 1.5
    node_budget: int = 65_536
    internal_temperature: float = 2.0
    target_temperature: float = 2.0
    prior_mix: float = 0.05
    seed: int = 1

    def depth_for_hand(self, hand_size: int) -> int | None:
        if hand_size <= 6:
            return None
        if hand_size <= 8:
            return 3
        return 2


@dataclass
class InformationSearchDecision:
    action: Action
    action_values: dict[str, float]
    action_probabilities: dict[str, float]
    action_regrets: dict[str, float]
    action_stderr: dict[str, float]
    prior_probabilities: dict[str, float]
    split_half_argmax_agreement: bool
    target_js_divergence: float
    accepted: bool
    determinizations: int
    expanded_nodes: int
    maximum_depth: int
    leaf_rollouts: int
    leaf_value_predictions: int
    node_budget_fallback: bool


@dataclass
class _World:
    env: PlumpEnv
    index: int
    seed: int
    path: tuple[str, ...] = ()


@dataclass
class _Budget:
    remaining: int
    expanded: int = 0
    maximum_depth: int = 0
    leaf_rollouts: int = 0
    leaf_value_predictions: int = 0

    def spend(self, count: int = 1, *, depth: int = 0) -> None:
        if count > self.remaining:
            raise _NodeBudgetExceeded
        self.remaining -= count
        self.expanded += count
        self.maximum_depth = max(self.maximum_depth, depth)


class _NodeBudgetExceeded(RuntimeError):
    pass


class InformationSetSearch:
    """One-sided expectimax with focal choices shared by information set."""

    def __init__(
        self,
        focal_policy: ModelPolicy,
        *,
        opponent_policy_for_player: Callable[[int], ActionPolicy],
        value_intercept: Callable[[Observation], float] | None = None,
        config: InformationSearchConfig | None = None,
        scoring: ScoringConfig | None = None,
        belief_weight: float = 0.0,
        leaf_value_weight: float = 0.0,
    ) -> None:
        self.focal_policy = focal_policy
        self.opponent_policy_for_player = opponent_policy_for_player
        self.value_intercept = value_intercept or (lambda _: 0.0)
        self.config = config or InformationSearchConfig()
        self.scoring = scoring or ScoringConfig()
        self.belief_weight = min(max(belief_weight, 0.0), 1.0)
        self.leaf_value_weight = min(max(leaf_value_weight, 0.0), 1.0)
        self.sampler = ConstrainedDeterminizationSampler()
        self._policy_cache: dict[
            tuple,
            tuple[
                dict[Action, float],
                dict[Action, float],
                list[list[float]],
                ModelConfig,
            ],
        ] = {}

    def search(
        self,
        observation: Observation,
        *,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> InformationSearchDecision:
        if len(legal_actions) < 2:
            raise ValueError("Information-set search requires a choice.")
        focal_player = observation.player_id
        prior, _, owner_probs, model_config = self._model_information(
            observation,
            legal_actions,
        )
        proposal = _mix_owner_proposal(owner_probs, self.belief_weight)
        worlds: list[_World] = []
        breadths = _breadths(self.config)
        final_values: dict[Action, list[float]] = {}
        final_half_values: tuple[dict[Action, float], dict[Action, float]] | None = None
        fallback = False
        budget = _Budget(self.config.node_budget)

        for target in breadths:
            while len(worlds) < target:
                determinization = self.sampler.sample(
                    observation,
                    owner_probs=proposal,
                    model_config=model_config,
                    rng=rng,
                )
                worlds.append(
                    _World(
                        reconstruct_public_state(
                            observation,
                            determinization,
                            scoring=self.scoring,
                        ),
                        len(worlds),
                        rng.randrange(2**31),
                    )
                )
            depth_limit = self.config.depth_for_hand(
                observation.hand_size
            )
            if (
                _projected_tree_nodes(
                    observation,
                    world_count=len(worlds),
                    root_action_count=len(legal_actions),
                    depth_limit=depth_limit,
                )
                > budget.remaining
            ):
                fallback = True
                final_values = self._root_rollout_values(
                    worlds,
                    legal_actions,
                    focal_player=focal_player,
                    budget=budget,
                )
                midpoint = len(worlds) // 2
                if midpoint >= 2:
                    final_half_values = (
                        _means(
                            {
                                action: rows[:midpoint]
                                for action, rows in final_values.items()
                            }
                        ),
                        _means(
                            {
                                action: rows[midpoint:]
                                for action, rows in final_values.items()
                            }
                        ),
                    )
                break
            try:
                final_values = self._root_action_values(
                    worlds,
                    legal_actions,
                    focal_player=focal_player,
                    depth_limit=depth_limit,
                    budget=budget,
                )
                if len(worlds) >= 4:
                    midpoint = len(worlds) // 2
                    first = self._root_action_values(
                        worlds[:midpoint],
                        legal_actions,
                        focal_player=focal_player,
                        depth_limit=depth_limit,
                        budget=budget,
                    )
                    second = self._root_action_values(
                        worlds[midpoint:],
                        legal_actions,
                        focal_player=focal_player,
                        depth_limit=depth_limit,
                        budget=budget,
                    )
                    final_half_values = (
                        _means(first),
                        _means(second),
                    )
            except _NodeBudgetExceeded:
                fallback = True
                final_values = self._root_rollout_values(
                    worlds,
                    legal_actions,
                    focal_player=focal_player,
                    budget=budget,
                )
                midpoint = len(worlds) // 2
                if midpoint >= 2:
                    final_half_values = (
                        _means(
                            {
                                action: rows[:midpoint]
                                for action, rows in final_values.items()
                            }
                        ),
                        _means(
                            {
                                action: rows[midpoint:]
                                for action, rows in final_values.items()
                            }
                        ),
                    )
                break
            if target >= self.config.max_determinizations:
                break
            if not _actions_are_close(
                final_values,
                self.config.close_action_stderr,
            ):
                break

        means = _means(final_values)
        q_policy = _softmax_values(
            means,
            temperature=self.config.target_temperature,
        )
        target = {
            action: (
                (1.0 - self.config.prior_mix) * q_policy[action]
                + self.config.prior_mix * prior[action]
            )
            for action in legal_actions
        }
        baseline = sum(prior[action] * means[action] for action in legal_actions)
        regrets = {
            action: means[action] - baseline
            for action in legal_actions
        }
        agreement = False
        target_js = float("inf")
        if final_half_values is not None:
            half_targets = [
                _blend_target(
                    _softmax_values(
                        values,
                        temperature=self.config.target_temperature,
                    ),
                    prior,
                    self.config.prior_mix,
                )
                for values in final_half_values
            ]
            agreement = (
                max(final_half_values[0], key=final_half_values[0].get)
                == max(final_half_values[1], key=final_half_values[1].get)
            )
            target_js = _jensen_shannon(half_targets[0], half_targets[1])
        selected = rng.choices(
            legal_actions,
            weights=[target[action] for action in legal_actions],
            k=1,
        )[0]
        return InformationSearchDecision(
            action=selected,
            action_values={
                _action_key(action): means[action]
                for action in legal_actions
            },
            action_probabilities={
                _action_key(action): target[action]
                for action in legal_actions
            },
            action_regrets={
                _action_key(action): regrets[action]
                for action in legal_actions
            },
            action_stderr={
                _action_key(action): _stderr(final_values[action])
                for action in legal_actions
            },
            prior_probabilities={
                _action_key(action): prior[action]
                for action in legal_actions
            },
            split_half_argmax_agreement=agreement,
            target_js_divergence=target_js,
            accepted=(
                agreement
                and target_js <= self.config.maximum_target_js
            ),
            determinizations=len(worlds),
            expanded_nodes=budget.expanded,
            maximum_depth=budget.maximum_depth,
            leaf_rollouts=budget.leaf_rollouts,
            leaf_value_predictions=budget.leaf_value_predictions,
            node_budget_fallback=fallback,
        )

    def _root_action_values(
        self,
        worlds: list[_World],
        legal_actions: list[Action],
        *,
        focal_player: int,
        depth_limit: int | None,
        budget: _Budget,
    ) -> dict[Action, list[float]]:
        values: dict[Action, list[float]] = {}
        _, q_order, _, _ = self._model_information(
            worlds[0].env.get_observation(focal_player),
            legal_actions,
        )
        ordered = sorted(
            legal_actions,
            key=lambda item: q_order[item],
            reverse=True,
        )
        return self._all_action_world_values(
            worlds,
            ordered,
            focal_player=focal_player,
            depth=0,
            depth_limit=depth_limit,
            path=(),
            budget=budget,
        )

    def _state_value(
        self,
        worlds: list[_World],
        *,
        focal_player: int,
        depth: int,
        depth_limit: int | None,
        path: tuple[str, ...],
        budget: _Budget,
    ) -> float:
        budget.maximum_depth = max(budget.maximum_depth, depth)
        if all(world.env.is_done() for world in worlds):
            return sum(
                _terminal_relative_reward(world.env, focal_player)
                for world in worlds
            ) / len(worlds)
        if depth_limit is not None and depth >= depth_limit:
            return sum(
                self._leaf_values(
                    worlds,
                    focal_player=focal_player,
                    path=path,
                    budget=budget,
                )
            ) / len(worlds)

        observation = worlds[0].env.get_observation(focal_player)
        legal_actions = worlds[0].env.legal_actions()
        key = information_set_key(observation)
        for world in worlds[1:]:
            other = world.env.get_observation(focal_player)
            if information_set_key(other) != key:
                raise AssertionError(
                    "A focal search node mixed distinct information sets."
                )
            if world.env.legal_actions() != legal_actions:
                raise AssertionError(
                    "Information-set worlds disagree on legal actions."
                )
        ordered = sorted(legal_actions, key=_action_index)
        rows_by_action = self._all_action_world_values(
            worlds,
            ordered,
            focal_player=focal_player,
            depth=depth,
            depth_limit=depth_limit,
            path=path,
            budget=budget,
        )
        action_values = {
            action: sum(rows) / len(rows)
            for action, rows in rows_by_action.items()
        }
        improved = _softmax_values(
            action_values,
            temperature=self.config.internal_temperature,
        )
        return sum(
            improved[action] * action_values[action]
            for action in legal_actions
        )

    def _all_action_world_values(
        self,
        worlds: list[_World],
        actions: list[Action],
        *,
        focal_player: int,
        depth: int,
        depth_limit: int | None,
        path: tuple[str, ...],
        budget: _Budget,
    ) -> list[float]:
        children_by_action: dict[Action, list[_World]] = {
            action: [] for action in actions
        }
        all_children = []
        for action in actions:
            action_path = (*path, _action_key(action))
            for world in worlds:
                budget.spend(depth=depth)
                child = _World(
                    world.env.clone(),
                    world.index,
                    world.seed,
                    action_path,
                )
                child.env.step(action)
                children_by_action[action].append(child)
                all_children.append(child)
        self._advance_opponents_many(
            all_children,
            focal_player=focal_player,
            budget=budget,
            depth=depth,
        )
        expansions: dict[
            Action,
            tuple[dict[int, float], dict[tuple, list[_World]]],
        ] = {}
        for action, children in children_by_action.items():
            terminal: dict[int, float] = {}
            partitions: dict[tuple, list[_World]] = {}
            for child in children:
                if child.env.is_done():
                    terminal[child.index] = _terminal_relative_reward(
                        child.env,
                        focal_player,
                    )
                else:
                    observation = child.env.get_observation(focal_player)
                    partitions.setdefault(
                        information_set_key(observation),
                        [],
                    ).append(child)
            expansions[action] = (terminal, partitions)
        result: dict[Action, list[float]] = {}
        for action in actions:
            terminal, partitions = expansions[action]
            values = dict(terminal)
            for key, partition in partitions.items():
                partition_value = self._state_value(
                    partition,
                    focal_player=focal_player,
                    depth=depth + 1,
                    depth_limit=depth_limit,
                    path=(
                        *partition[0].path,
                        _key_digest(key),
                    ),
                    budget=budget,
                )
                for world in partition:
                    values[world.index] = partition_value
            result[action] = [
                values[world.index]
                for world in worlds
            ]
        return result

    def _advance_opponents_many(
        self,
        worlds: list[_World],
        *,
        focal_player: int,
        budget: _Budget,
        depth: int,
    ) -> None:
        steps = {id(world): 0 for world in worlds}
        while True:
            active = [
                world
                for world in worlds
                if (
                    not world.env.is_done()
                    and world.env.current_player() != focal_player
                )
            ]
            if not active:
                return
            grouped: dict[int, list[_World]] = {}
            policies: dict[int, ActionPolicy] = {}
            rngs: dict[int, random.Random] = {}
            for world in active:
                budget.spend(depth=depth)
                player = world.env.current_player()
                policy = self.opponent_policy_for_player(player)
                key = id(policy)
                grouped.setdefault(key, []).append(world)
                policies[key] = policy
                world_key = id(world)
                rngs[world_key] = random.Random(
                    _path_seed(
                        world.seed,
                        world.index,
                        ("opponent", str(depth)),
                        steps[world_key],
                        player,
                    )
                )
            for key, rows in grouped.items():
                policy = policies[key]
                if isinstance(policy, ModelPolicy):
                    actions = policy.act_many(
                        [world.env for world in rows],
                        rngs=[rngs[id(world)] for world in rows],
                    )
                else:
                    actions = [
                        policy.act(
                            world.env,
                            rng=rngs[id(world)],
                        )
                        for world in rows
                    ]
                for world, selected in zip(rows, actions):
                    world.env.step(selected)
                    steps[id(world)] += 1

    def _leaf_values(
        self,
        worlds: list[_World],
        *,
        focal_player: int,
        path: tuple[str, ...],
        budget: _Budget,
    ) -> list[float]:
        rollout_values = self._rollout_values_many(
            worlds,
            focal_player=focal_player,
            path=path,
            budget=budget,
        )
        if self.leaf_value_weight <= 0.0:
            return rollout_values
        observations = [
            world.env.get_observation(focal_player)
            for world in worlds
        ]
        _, output = self.focal_policy.predict_observations(observations)
        direct = output.value.squeeze(-1).float().cpu().tolist()
        budget.leaf_value_predictions += len(worlds)
        return [
            (
                (1.0 - self.leaf_value_weight) * rollout
                + self.leaf_value_weight
                * (prediction + self.value_intercept(observation))
            )
            for rollout, prediction, observation in zip(
                rollout_values,
                direct,
                observations,
            )
        ]

    def _rollout_value(
        self,
        world: _World,
        *,
        focal_player: int,
        path: tuple[str, ...],
        budget: _Budget,
    ) -> float:
        return self._rollout_values_many(
            [world],
            focal_player=focal_player,
            path=path,
            budget=budget,
        )[0]

    def _rollout_values_many(
        self,
        worlds: list[_World],
        *,
        focal_player: int,
        path: tuple[str, ...],
        budget: _Budget,
        allow_budget_overflow: bool = False,
    ) -> list[float]:
        rollout_worlds = [
            _World(
                world.env.clone(),
                world.index,
                world.seed,
                world.path,
            )
            for world in worlds
        ]
        steps = {id(world): 0 for world in rollout_worlds}
        while True:
            active = [
                world
                for world in rollout_worlds
                if not world.env.is_done()
            ]
            if not active:
                break
            grouped: dict[int, list[_World]] = {}
            policies: dict[int, ActionPolicy] = {}
            rngs: dict[int, random.Random] = {}
            for world in active:
                if allow_budget_overflow and budget.remaining == 0:
                    pass
                else:
                    budget.spend()
                player = world.env.current_player()
                policy = (
                    self.focal_policy
                    if player == focal_player
                    else self.opponent_policy_for_player(player)
                )
                key = id(policy)
                grouped.setdefault(key, []).append(world)
                policies[key] = policy
                world_key = id(world)
                rngs[world_key] = random.Random(
                    _path_seed(
                        world.seed,
                        world.index,
                        ("rollout", str(len(path))),
                        steps[world_key],
                        player,
                    )
                )
            for key, rows in grouped.items():
                policy = policies[key]
                if isinstance(policy, ModelPolicy):
                    actions = policy.act_many(
                        [world.env for world in rows],
                        rngs=[rngs[id(world)] for world in rows],
                    )
                else:
                    actions = [
                        policy.act(
                            world.env,
                            rng=rngs[id(world)],
                        )
                        for world in rows
                    ]
                for world, selected in zip(rows, actions):
                    world.env.step(selected)
                    steps[id(world)] += 1
        budget.leaf_rollouts += len(rollout_worlds)
        return [
            _terminal_relative_reward(world.env, focal_player)
            for world in rollout_worlds
        ]

    def _root_rollout_values(
        self,
        worlds: list[_World],
        legal_actions: list[Action],
        *,
        focal_player: int,
        budget: _Budget,
    ) -> dict[Action, list[float]]:
        children: list[_World] = []
        child_actions: list[Action] = []
        for action in legal_actions:
            for world in worlds:
                child = _World(
                    world.env.clone(),
                    world.index,
                    world.seed,
                    ("fallback", _action_key(action)),
                )
                child.env.step(action)
                children.append(child)
                child_actions.append(action)
        rollout_values = self._rollout_values_many(
            children,
            focal_player=focal_player,
            path=(),
            budget=budget,
            allow_budget_overflow=True,
        )
        values = {action: [] for action in legal_actions}
        for action, value in zip(child_actions, rollout_values):
            values[action].append(value)
        return values

    def _model_information(
        self,
        observation: Observation,
        legal_actions: list[Action],
    ) -> tuple[
        dict[Action, float],
        dict[Action, float],
        list[list[float]],
        ModelConfig,
    ]:
        key = information_set_key(observation)
        cached = self._policy_cache.get(key)
        if cached is not None:
            cached_prior, cached_q, owner_probs, model_config = cached
            return (
                {action: cached_prior[action] for action in legal_actions},
                {action: cached_q[action] for action in legal_actions},
                owner_probs,
                model_config,
            )
        encoded, output = self.focal_policy.predict_observation(observation)
        logits = (
            output.masked_bid_logits[0].float()
            if observation.phase == Phase.BIDDING
            else output.masked_card_logits[0].float()
        )
        probabilities = logits.softmax(dim=-1).cpu().tolist()
        q_tensor = (
            output.masked_bid_q_values[0]
            if observation.phase == Phase.BIDDING
            else output.masked_card_q_values[0]
        )
        if q_tensor is None:
            q_values = [0.0] * len(probabilities)
        else:
            q_values = q_tensor.float().cpu().tolist()
        prior = {
            action: probabilities[_action_index(action)]
            for action in legal_actions
        }
        q_order = {
            action: q_values[_action_index(action)]
            for action in legal_actions
        }
        self._policy_cache[key] = (
            prior,
            q_order,
            output.owner_probs[0].float().cpu().tolist(),
            self.focal_policy.model_config,
        )
        if encoded.observer_player != observation.player_id:
            raise AssertionError("Search model used the wrong observer.")
        return self._policy_cache[key]

class InformationSearchPolicy:
    """Deploy information-set search around an observation-only model policy."""

    def __init__(
        self,
        base_policy: ModelPolicy,
        opponent_policy: ActionPolicy,
        *,
        config: InformationSearchConfig | None = None,
        belief_weight: float = 1.0,
        leaf_value_weight: float = 1.0,
        name: str = "information-search",
    ) -> None:
        self.base_policy = base_policy
        self.opponent_policy = opponent_policy
        self.config = config or InformationSearchConfig()
        self.belief_weight = belief_weight
        self.leaf_value_weight = leaf_value_weight
        self.name = name
        self.forward_passes = 0
        self.last_decision: InformationSearchDecision | None = None

    def act(
        self,
        env: PlumpEnv,
        *,
        rng: random.Random | None = None,
    ) -> Action:
        legal = env.legal_actions()
        if len(legal) == 1:
            return legal[0]
        before = self.base_policy.forward_passes
        search = InformationSetSearch(
            self.base_policy,
            opponent_policy_for_player=lambda _: self.opponent_policy,
            config=self.config,
            belief_weight=self.belief_weight,
            leaf_value_weight=self.leaf_value_weight,
        )
        decision = search.search(
            env.get_observation(env.current_player()),
            legal_actions=legal,
            rng=rng or random.Random(self.config.seed),
        )
        self.last_decision = decision
        self.forward_passes += (
            self.base_policy.forward_passes - before
        )
        best_key = max(
            decision.action_values,
            key=decision.action_values.get,
        )
        index = int(best_key.split(":", 1)[1])
        if env.phase() == Phase.BIDDING:
            return BidAction(env.current_player(), index)
        return PlayCardAction(
            env.current_player(),
            next(
                action.card
                for action in legal
                if isinstance(action, PlayCardAction)
                and card_id(action.card) == index
            ),
        )

    def reset_counters(self) -> None:
        self.forward_passes = 0
        self.base_policy.reset_counters()
        self.opponent_policy.reset_counters()


def information_set_key(observation: Observation) -> tuple:
    """Complete immutable focal information set, independent of hidden deal."""

    return (
        observation.player_id,
        observation.phase.value,
        observation.round_index,
        observation.total_rounds,
        observation.rounds_remaining,
        observation.hand_size,
        observation.trump_suit.value if observation.trump_suit else None,
        observation.current_player,
        observation.bidding_start_player,
        tuple(observation.bidding_order),
        observation.play_start_player,
        tuple(sorted(card_id(card) for card in observation.my_hand)),
        tuple(
            (bid.player, bid.value, bid.position)
            for bid in observation.bids
        ),
        tuple(sorted(observation.tricks_won.items())),
        _trick_key(observation.current_trick),
        tuple(_trick_key(trick) for trick in observation.completed_tricks),
        tuple(
            (
                player,
                tuple(card_id(card) for card in cards),
            )
            for player, cards in sorted(
                observation.played_cards_by_player.items()
            )
        ),
        tuple(
            (
                player,
                tuple(
                    (suit.value, is_void)
                    for suit, is_void in sorted(
                        suits.items(),
                        key=lambda item: item[0].value,
                    )
                ),
            )
            for player, suits in sorted(observation.voids.items())
        ),
        tuple(observation.legal_bids),
        tuple(sorted(card_id(card) for card in observation.legal_cards)),
        tuple(sorted(observation.scores.items())),
        tuple(
            (
                event.type.value,
                event.round_index,
                event.player,
                card_id(event.card) if event.card is not None else None,
                event.bid,
                event.trick_index,
                event.position_in_trick,
            )
            for event in observation.event_log
        ),
        tuple(observation.hand_size_schedule),
    )


def _trick_key(trick: Trick | None) -> tuple | None:
    if trick is None:
        return None
    return (
        trick.trick_index,
        trick.leader,
        trick.led_suit.value if trick.led_suit else None,
        tuple(
            (play.player, card_id(play.card), play.position)
            for play in trick.plays
        ),
        trick.winner,
    )


def _mix_owner_proposal(
    owner_probs: list[list[float]],
    weight: float,
) -> list[list[float]] | None:
    if weight <= 0.0:
        return None
    mixed = []
    for row in owner_probs:
        active = [value > 0.0 for value in row]
        count = sum(active)
        if count == 0:
            mixed.append(row)
            continue
        uniform = 1.0 / count
        mixed.append(
            [
                (
                    weight * value
                    + (1.0 - weight) * uniform
                    if valid
                    else 0.0
                )
                for value, valid in zip(row, active)
            ]
        )
    return mixed


def _breadths(config: InformationSearchConfig) -> list[int]:
    values = []
    breadth = config.min_determinizations
    while breadth < config.max_determinizations:
        values.append(breadth)
        breadth = min(
            config.max_determinizations,
            max(
                breadth * 2,
                breadth + config.batch_determinizations,
            ),
        )
    values.append(config.max_determinizations)
    return values


def _projected_tree_nodes(
    observation: Observation,
    *,
    world_count: int,
    root_action_count: int,
    depth_limit: int | None,
) -> int:
    """Conservative work estimate used only to avoid doomed tree expansion."""

    remaining_cards = len(observation.my_hand)
    focal_plies = (
        remaining_cards
        if depth_limit is None
        else min(depth_limit, remaining_cards)
    )
    paths = world_count * root_action_count
    for offset in range(1, focal_plies):
        paths *= max(1, remaining_cards - offset + 1)
    public_plays = sum(
        len(trick.plays)
        for trick in observation.completed_tricks
    )
    if observation.current_trick is not None:
        public_plays += len(observation.current_trick.plays)
    card_actions_left = (
        observation.hand_size * len(observation.tricks_won)
        - public_plays
    )
    bid_actions_left = max(
        0,
        len(observation.tricks_won) - len(observation.bids),
    )
    rollout_actions = max(
        1,
        card_actions_left + bid_actions_left - focal_plies,
    )
    # The full solve plus two split-half solves is approximately twice the
    # full work. The additional margin covers intervening opponent actions.
    return 3 * paths * rollout_actions


def _action_index(action: Action) -> int:
    return action.bid if isinstance(action, BidAction) else card_id(action.card)


def _legal_actions_from_observation(
    observation: Observation,
) -> list[Action]:
    if observation.phase == Phase.BIDDING:
        return [
            BidAction(observation.player_id, bid)
            for bid in observation.legal_bids
        ]
    return [
        PlayCardAction(observation.player_id, card)
        for card in observation.legal_cards
    ]


def _action_key(action: Action) -> str:
    prefix = "bid" if isinstance(action, BidAction) else "card"
    return f"{prefix}:{_action_index(action)}"


def _terminal_relative_reward(env: PlumpEnv, focal_player: int) -> float:
    if len(env.state.rounds) == 1:
        scores = env.state.current_round.round_scores
    else:
        scores = env.state.cumulative_scores
    total = sum(scores.values())
    opponents = len(scores) - 1
    return float(
        scores[focal_player]
        - (total - scores[focal_player]) / opponents
    )


def _softmax_values(
    values: dict[Action, float],
    *,
    temperature: float,
) -> dict[Action, float]:
    if temperature <= 0.0:
        raise ValueError("Search temperature must be positive.")
    peak = max(values.values())
    weights = {
        action: math.exp((value - peak) / temperature)
        for action, value in values.items()
    }
    total = sum(weights.values())
    return {
        action: weight / total
        for action, weight in weights.items()
    }


def _blend_target(
    improved: dict[Action, float],
    prior: dict[Action, float],
    prior_mix: float,
) -> dict[Action, float]:
    return {
        action: (
            (1.0 - prior_mix) * improved[action]
            + prior_mix * prior[action]
        )
        for action in improved
    }


def _means(values: dict[Action, list[float]]) -> dict[Action, float]:
    return {
        action: sum(rows) / len(rows)
        for action, rows in values.items()
    }


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    mean = sum(values) / len(values)
    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _actions_are_close(
    values: dict[Action, list[float]],
    multiplier: float,
) -> bool:
    ranked = sorted(
        (
            (sum(rows) / len(rows), _stderr(rows))
            for rows in values.values()
        ),
        reverse=True,
    )
    if len(ranked) < 2:
        return False
    return (
        ranked[0][0] - ranked[1][0]
        <= multiplier * (ranked[0][1] + ranked[1][1])
    )


def _jensen_shannon(
    first: dict[Action, float],
    second: dict[Action, float],
) -> float:
    midpoint = {
        action: 0.5 * (first[action] + second[action])
        for action in first
    }

    def divergence(source: dict[Action, float]) -> float:
        return sum(
            probability
            * math.log(
                probability
                / max(midpoint[action], 1e-12)
            )
            for action, probability in source.items()
            if probability > 0.0
        )

    return 0.5 * (divergence(first) + divergence(second))


def _key_digest(key: tuple) -> str:
    return hashlib.blake2b(
        repr(key).encode("utf-8"),
        digest_size=8,
    ).hexdigest()


def _path_seed(
    seed: int,
    world_index: int,
    path: tuple[str, ...],
    step: int,
    player: int,
) -> int:
    payload = "|".join(
        [
            str(seed),
            str(world_index),
            *path,
            str(step),
            str(player),
        ]
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "big",
    )
