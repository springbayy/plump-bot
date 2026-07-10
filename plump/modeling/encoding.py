"""Schema-v4 observation encoding for position-robust Plump agents."""

from __future__ import annotations

from dataclasses import dataclass

from plump.cards import Card, Rank, Suit
from plump.state import EventType, GameEvent, Observation, Phase


SCHEMA_VERSION = 4
EVENT_TOKEN_WIDTH = 8
NUM_CARDS = 52
NUM_SCHEDULE_STATUSES = 4

SUITS: tuple[Suit, ...] = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
RANKS: tuple[Rank, ...] = tuple(Rank)
PHASES: tuple[Phase, ...] = tuple(Phase)

EVENT_TYPE_IDS: dict[EventType, int] = {
    EventType.ROUND_START: 1,
    EventType.BID: 2,
    EventType.PLAY: 3,
    EventType.TRICK_WIN: 4,
    EventType.ROUND_END: 5,
}
PAD_EVENT_TYPE_ID = 0
NUM_EVENT_TYPES = 6


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and schema-v4 encoding limits."""

    schema_version: int = SCHEMA_VERSION
    max_players: int = 5
    max_hand_size: int = 10
    max_rounds: int = 64
    max_seq_len: int = 128
    d_model: int = 128
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 512
    context_hidden_dim: int = 256
    game_hidden_dim: int = 128
    schedule_layers: int = 1
    schedule_heads: int = 4
    owner_sinkhorn_iterations: int = 16
    dropout: float = 0.0

    @property
    def bid_count(self) -> int:
        return self.max_hand_size + 1

    @property
    def bid_feature_count(self) -> int:
        return self.max_hand_size + 2

    @property
    def owner_class_count(self) -> int:
        # Relative opponents 1..max_players-1, followed by the undealt pool.
        return self.max_players

    @property
    def undealt_owner_class(self) -> int:
        return self.max_players - 1

    @property
    def player_na_id(self) -> int:
        return self.max_players

    @property
    def rank_na_id(self) -> int:
        return len(RANKS)

    @property
    def suit_none_id(self) -> int:
        return len(SUITS)

    @property
    def card_na_id(self) -> int:
        return NUM_CARDS

    @property
    def bid_na_id(self) -> int:
        return self.max_hand_size + 1

    @property
    def trick_na_id(self) -> int:
        return self.max_hand_size + 1

    @property
    def pos_na_id(self) -> int:
        return self.max_players

    @property
    def game_context_dim(self) -> int:
        p = self.max_players
        return p + p + 3  # standings, score differences, schedule/remaining/total

    @property
    def game_feature_dim(self) -> int:
        p = self.max_players
        return 2 * p + 7

    @property
    def context_dim(self) -> int:
        p = self.max_players
        return (
            3 * NUM_CARDS
            + 2 * p * NUM_CARDS
            + p * self.bid_feature_count
            + 3 * p
            + p * len(SUITS)
            + len(SUITS)
            + 1
            + self.bid_count
            + len(PHASES)
            + (p + 1)
            + (p + 1)
            + p
            + (p + 1)
            + (p + 1)
            + len(SUITS)
            + 1
            + (p + 1)
            + p
            + 1
            + self.game_context_dim
            + 1
        )

    @property
    def player_feature_dim(self) -> int:
        return (
            self.bid_feature_count
            + 5
            + len(SUITS)
            + NUM_CARDS
            + 3
            + self.max_players
        )


@dataclass
class EncodedObservation:
    """Fixed-shape schema-v4 model input."""

    event_tokens: list[list[int]]
    event_valid_mask: list[bool]
    context_features: list[float]
    player_features: list[list[float]]
    active_player_mask: list[bool]
    legal_bid_mask: list[bool]
    legal_card_mask: list[bool]
    final_trick_count_mask: list[list[bool]]
    owner_valid_mask: list[list[bool]]
    owner_capacities: list[int]
    bid_values: list[int]
    game_context_features: list[float]
    schedule_hand_sizes: list[int]
    schedule_statuses: list[int]
    schedule_valid_mask: list[bool]
    game_context_enabled: bool
    num_players: int
    observer_player: int
    current_player_relative: int
    bidding_position: int


def card_id(card: Card) -> int:
    return SUITS.index(card.suit) * len(RANKS) + RANKS.index(card.rank)


def card_from_id(index: int) -> Card:
    if index < 0 or index >= NUM_CARDS:
        raise ValueError(f"Card id must be in 0..{NUM_CARDS - 1}.")
    return Card(SUITS[index // len(RANKS)], RANKS[index % len(RANKS)])


def encode_observation(
    observation: Observation,
    config: ModelConfig | None = None,
    *,
    include_game_context: bool = False,
) -> EncodedObservation:
    """Encode one player-visible observation without hidden-state leakage."""

    config = config or ModelConfig()
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Expected schema version {SCHEMA_VERSION}, got {config.schema_version}.")
    num_players = len(observation.scores)
    _validate_observation_limits(observation, config, num_players)

    round_events = [event for event in observation.event_log if event.round_index == observation.round_index]
    event_tokens = _encode_events(round_events, observation.player_id, num_players, config)
    visible_event_count = min(len(round_events), config.max_seq_len)
    event_valid_mask = [index < visible_event_count for index in range(config.max_seq_len)]

    legal_bid_mask = [False] * config.bid_count
    for bid in observation.legal_bids:
        if 0 <= bid < config.bid_count:
            legal_bid_mask[bid] = True

    legal_card_mask = [False] * NUM_CARDS
    for card in observation.legal_cards:
        legal_card_mask[card_id(card)] = True

    bidding_position = _bidding_position(observation, observation.player_id, num_players, config.pos_na_id)
    context_features = _encode_context(
        observation,
        config,
        num_players,
        legal_bid_mask,
        legal_card_mask,
        include_game_context,
    )
    player_features, bid_values = _encode_player_features(observation, config, num_players)
    active_player_mask = [index < num_players for index in range(config.max_players)]

    owner_valid_mask = _owner_valid_mask(
        observation,
        config,
        num_players,
    )
    owner_capacities = _owner_capacities(
        observation,
        config,
        num_players,
    )
    hidden_card_count = sum(any(row) for row in owner_valid_mask)
    if sum(owner_capacities) != hidden_card_count:
        raise AssertionError(
            "Owner capacities do not match the number of hidden cards."
        )

    return EncodedObservation(
        event_tokens=event_tokens,
        event_valid_mask=event_valid_mask,
        context_features=context_features,
        player_features=player_features,
        active_player_mask=active_player_mask,
        legal_bid_mask=legal_bid_mask,
        legal_card_mask=legal_card_mask,
        final_trick_count_mask=_final_trick_count_mask_by_relative(observation, config, num_players),
        owner_valid_mask=owner_valid_mask,
        owner_capacities=owner_capacities,
        bid_values=bid_values,
        game_context_features=_encode_game_context(
            observation,
            config,
            num_players,
            include_game_context,
        ),
        schedule_hand_sizes=_encode_schedule_hand_sizes(
            observation,
            config,
            include_game_context,
        ),
        schedule_statuses=_encode_schedule_statuses(
            observation,
            config,
            include_game_context,
        ),
        schedule_valid_mask=_encode_schedule_valid_mask(
            observation,
            config,
            include_game_context,
        ),
        game_context_enabled=include_game_context,
        num_players=num_players,
        observer_player=observation.player_id,
        current_player_relative=_relative_player_id(
            observation.current_player,
            observation.player_id,
            num_players,
            config.player_na_id,
        ),
        bidding_position=bidding_position,
    )


def _validate_observation_limits(observation: Observation, config: ModelConfig, num_players: int) -> None:
    if num_players > config.max_players:
        raise ValueError(f"Observation has {num_players} players but model supports {config.max_players}.")
    if observation.hand_size > config.max_hand_size:
        raise ValueError(
            f"Observation hand size {observation.hand_size} exceeds model max {config.max_hand_size}."
        )
    if observation.round_index >= config.max_rounds:
        raise ValueError(f"Round index {observation.round_index} exceeds model max {config.max_rounds}.")
    if len(observation.hand_size_schedule) > config.max_rounds:
        raise ValueError(
            f"Schedule has {len(observation.hand_size_schedule)} rounds but model supports "
            f"{config.max_rounds}."
        )
    if any(hand_size > config.max_hand_size for hand_size in observation.hand_size_schedule):
        raise ValueError("Schedule hand size exceeds model capacity.")


def _encode_events(
    events: list[GameEvent],
    observer: int,
    num_players: int,
    config: ModelConfig,
) -> list[list[int]]:
    tokens: list[list[int]] = []
    for event in events[-config.max_seq_len :]:
        token = [
            EVENT_TYPE_IDS[event.type],
            _relative_player_id(event.player, observer, num_players, config.player_na_id),
            RANKS.index(event.card.rank) if event.card is not None else config.rank_na_id,
            SUITS.index(event.card.suit) if event.card is not None else config.suit_none_id,
            card_id(event.card) if event.card is not None else config.card_na_id,
            event.bid if event.bid is not None else config.bid_na_id,
            event.trick_index if event.trick_index is not None else config.trick_na_id,
            event.position_in_trick if event.position_in_trick is not None else config.pos_na_id,
        ]
        tokens.append(token)

    pad = [
        PAD_EVENT_TYPE_ID,
        config.player_na_id,
        config.rank_na_id,
        config.suit_none_id,
        config.card_na_id,
        config.bid_na_id,
        config.trick_na_id,
        config.pos_na_id,
    ]
    while len(tokens) < config.max_seq_len:
        tokens.append(list(pad))
    return tokens


def _encode_context(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
    legal_bid_mask: list[bool],
    legal_card_mask: list[bool],
    include_game_context: bool,
) -> list[float]:
    played_by_relative = _cards_by_relative_player(observation.played_cards_by_player, observation, config, num_players)
    current_trick_by_relative = [[0.0] * NUM_CARDS for _ in range(config.max_players)]
    current_trick_leader = None
    current_trick_led_suit = None
    trick_position = config.pos_na_id
    if observation.current_trick is not None:
        current_trick_leader = observation.current_trick.leader
        current_trick_led_suit = observation.current_trick.led_suit
        trick_position = min(len(observation.current_trick.plays), config.max_players)
        for play in observation.current_trick.plays:
            rel = _relative_player_id(play.player, observation.player_id, num_players, config.player_na_id)
            if rel < config.max_players:
                current_trick_by_relative[rel][card_id(play.card)] = 1.0

    bids = _bid_status_by_relative(observation, config, num_players)
    tricks_won = _scalar_by_relative(
        observation.tricks_won, observation.player_id, num_players, config.max_players, config.max_hand_size
    )
    tricks_needed = _tricks_needed_by_relative(observation, config, num_players)
    cards_remaining = _cards_remaining_by_relative(observation, config, num_players)
    voids = _voids_by_relative(observation, config, num_players)

    features: list[float] = []
    features.extend(_multi_hot_cards(observation.my_hand))
    features.extend(float(value) for value in legal_card_mask)
    features.extend(_multi_hot_cards(observation.played_cards_total))
    for row in played_by_relative:
        features.extend(row)
    for row in current_trick_by_relative:
        features.extend(row)
    for row in bids:
        features.extend(row)
    features.extend(tricks_won)
    features.extend(tricks_needed)
    features.extend(cards_remaining)
    for row in voids:
        features.extend(row)
    features.extend(_suit_one_hot(observation.trump_suit, config))
    features.extend(_one_hot(observation.hand_size, config.bid_count))
    features.extend(_one_hot(PHASES.index(observation.phase), len(PHASES)))
    features.extend(_relative_player_one_hot(observation.current_player, observation, config, num_players))
    features.extend(_one_hot(_bidding_position(observation, observation.player_id, num_players, config.pos_na_id), config.max_players + 1))
    features.extend(_relative_player_one_hot(observation.bidding_start_player, observation, config, num_players)[:-1])
    features.extend(_relative_player_one_hot(observation.play_start_player, observation, config, num_players))
    features.extend(_relative_player_one_hot(current_trick_leader, observation, config, num_players))
    features.extend(_suit_one_hot(current_trick_led_suit, config))
    features.extend(_one_hot(trick_position, config.max_players + 1))
    features.extend(1.0 if rel < num_players else 0.0 for rel in range(config.max_players))
    features.append(_normalize(num_players, config.max_players))

    if include_game_context:
        maximum_score = _maximum_attainable_score(
            observation.hand_size_schedule,
        )
        scores = _scalar_by_relative(
            observation.scores,
            observation.player_id,
            num_players,
            config.max_players,
            maximum_score,
        )
        own_score = observation.scores.get(observation.player_id, 0)
        score_diffs = []
        for rel in range(config.max_players):
            if rel >= num_players:
                score_diffs.append(0.0)
            else:
                player = (observation.player_id + rel) % num_players
                score_diffs.append(
                    _normalize_signed(
                        own_score - observation.scores.get(player, 0),
                        maximum_score,
                    )
                )
        features.extend(scores)
        features.extend(score_diffs)
        features.append(_normalize(observation.round_index, max(observation.total_rounds - 1, 1)))
        features.append(_normalize(observation.rounds_remaining, max(observation.total_rounds, 1)))
        features.append(_normalize(observation.total_rounds, config.max_rounds))
    else:
        features.extend([0.0] * config.game_context_dim)
    features.append(1.0 if include_game_context else 0.0)

    if len(features) != config.context_dim:
        raise AssertionError(f"Context feature size {len(features)} != expected {config.context_dim}.")
    return features


def _encode_game_context(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
    include_game_context: bool,
) -> list[float]:
    if not include_game_context:
        return [0.0] * config.game_feature_dim

    schedule = observation.hand_size_schedule
    maximum_score = _maximum_attainable_score(schedule)
    scores = _scalar_by_relative(
        observation.scores,
        observation.player_id,
        num_players,
        config.max_players,
        maximum_score,
    )
    own_score = observation.scores.get(observation.player_id, 0)
    score_differences = []
    for rel in range(config.max_players):
        if rel >= num_players:
            score_differences.append(0.0)
            continue
        player = (observation.player_id + rel) % num_players
        score_differences.append(
            _normalize_signed(
                own_score - observation.scores.get(player, 0),
                maximum_score,
            )
        )

    total_rounds = max(len(schedule), 1)
    current_index = max(min(observation.round_index, total_rounds - 1), 0)
    current_hand = schedule[current_index] if schedule else observation.hand_size
    next_hand = schedule[current_index + 1] if current_index + 1 < len(schedule) else current_hand
    previous_hand = schedule[current_index - 1] if current_index > 0 and schedule else current_hand
    descending = next_hand < current_hand or (
        next_hand == current_hand and current_hand < previous_hand
    )
    ascending = next_hand > current_hand or (
        next_hand == current_hand and current_hand > previous_hand
    )
    features = [
        *scores,
        *score_differences,
        _normalize(current_index, max(total_rounds - 1, 1)),
        _normalize(observation.rounds_remaining, total_rounds),
        _normalize(total_rounds, config.max_rounds),
        _normalize(min(schedule) if schedule else observation.hand_size, config.max_hand_size),
        _normalize(max(schedule) if schedule else observation.hand_size, config.max_hand_size),
        float(descending),
        float(ascending),
    ]
    if len(features) != config.game_feature_dim:
        raise AssertionError(
            f"Game context size {len(features)} != expected {config.game_feature_dim}."
        )
    return features


def _encode_schedule_hand_sizes(
    observation: Observation,
    config: ModelConfig,
    include_game_context: bool,
) -> list[int]:
    if not include_game_context:
        return [0] * config.max_rounds
    values = [hand_size + 1 for hand_size in observation.hand_size_schedule]
    return values + [0] * (config.max_rounds - len(values))


def _encode_schedule_statuses(
    observation: Observation,
    config: ModelConfig,
    include_game_context: bool,
) -> list[int]:
    if not include_game_context:
        return [0] * config.max_rounds
    statuses = []
    for index in range(len(observation.hand_size_schedule)):
        if index < observation.round_index:
            statuses.append(1)
        elif index == observation.round_index:
            statuses.append(2)
        else:
            statuses.append(3)
    return statuses + [0] * (config.max_rounds - len(statuses))


def _encode_schedule_valid_mask(
    observation: Observation,
    config: ModelConfig,
    include_game_context: bool,
) -> list[bool]:
    count = len(observation.hand_size_schedule) if include_game_context else 0
    return [index < count for index in range(config.max_rounds)]


def _maximum_attainable_score(schedule: list[int]) -> int:
    return max(
        sum(max(5, 10 + hand_size) for hand_size in schedule),
        1,
    )


def _encode_player_features(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> tuple[list[list[float]], list[int]]:
    played_by_relative = _cards_by_relative_player(observation.played_cards_by_player, observation, config, num_players)
    bids_by_abs = {bid.player: bid.value for bid in observation.bids}
    features: list[list[float]] = []
    bid_values: list[int] = []

    for rel in range(config.max_players):
        active = rel < num_players
        abs_player = (observation.player_id + rel) % num_players if active else None
        bid_value = bids_by_abs.get(abs_player, -1) if abs_player is not None else -1
        bid_values.append(bid_value)
        tricks_won = observation.tricks_won.get(abs_player, 0) if abs_player is not None else 0
        played_count = len(observation.played_cards_by_player.get(abs_player, [])) if abs_player is not None else 0
        cards_remaining = max(observation.hand_size - played_count, 0) if active else 0

        row: list[float] = []
        row.extend(_bid_status_one_hot(bid_value, config))
        row.append(1.0 if bid_value >= 0 else 0.0)
        row.append(_normalize(max(bid_value, 0), config.max_hand_size))
        row.append(_normalize(tricks_won, config.max_hand_size))
        row.append(_normalize_signed(bid_value - tricks_won, config.max_hand_size) if bid_value >= 0 else 0.0)
        row.append(_normalize(cards_remaining, config.max_hand_size))
        row.extend(_void_list(observation, abs_player))
        row.extend(played_by_relative[rel])
        row.append(1.0 if abs_player == observation.current_player else 0.0)
        row.append(1.0 if rel == 0 and active else 0.0)
        row.append(1.0 if active else 0.0)
        position = _bidding_position(observation, abs_player, num_players, config.pos_na_id)
        row.extend(_one_hot(position, config.max_players) if position < config.max_players else [0.0] * config.max_players)
        if len(row) != config.player_feature_dim:
            raise AssertionError(f"Player feature size {len(row)} != expected {config.player_feature_dim}.")
        features.append(row)
    return features, bid_values


def _final_trick_count_mask_by_relative(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[bool]]:
    rows: list[list[bool]] = []
    completed_tricks = len(observation.completed_tricks)
    unresolved_tricks = max(observation.hand_size - completed_tricks, 0)
    for rel in range(config.max_players):
        row = [False] * config.bid_count
        if rel < num_players:
            player = (observation.player_id + rel) % num_players
            won = observation.tricks_won.get(player, 0)
            for count in range(won, min(won + unresolved_tricks, config.max_hand_size) + 1):
                row[count] = True
        rows.append(row)
    return rows


def _owner_valid_mask(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[bool]]:
    own_cards = {card_id(card) for card in observation.my_hand}
    played_cards = {card_id(card) for card in observation.played_cards_total}
    kitty_size = NUM_CARDS - observation.hand_size * num_players
    rows: list[list[bool]] = []
    for index in range(NUM_CARDS):
        row = [False] * config.owner_class_count
        if index not in own_cards and index not in played_cards:
            card = card_from_id(index)
            for rel in range(1, num_players):
                player = (observation.player_id + rel) % num_players
                if not observation.voids.get(player, {}).get(card.suit, False):
                    row[rel - 1] = True
            if kitty_size > 0:
                row[config.undealt_owner_class] = True
        rows.append(row)
    return rows


def _owner_capacities(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[int]:
    capacities = [0] * config.owner_class_count
    for rel in range(1, num_players):
        player = (observation.player_id + rel) % num_players
        played = len(observation.played_cards_by_player.get(player, []))
        capacities[rel - 1] = observation.hand_size - played
    capacities[config.undealt_owner_class] = (
        NUM_CARDS - observation.hand_size * num_players
    )
    return capacities


def _cards_by_relative_player(
    cards_by_abs_player: dict[int, list[Card]],
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[float]]:
    rows = [[0.0] * NUM_CARDS for _ in range(config.max_players)]
    for player, cards in cards_by_abs_player.items():
        rel = _relative_player_id(player, observation.player_id, num_players, config.player_na_id)
        if rel < config.max_players:
            rows[rel] = _multi_hot_cards(cards)
    return rows


def _bid_status_by_relative(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[float]]:
    bids = {bid.player: bid.value for bid in observation.bids}
    rows = []
    for rel in range(config.max_players):
        player = (observation.player_id + rel) % num_players if rel < num_players else None
        rows.append(_bid_status_one_hot(bids.get(player, -1) if player is not None else -1, config))
    return rows


def _tricks_needed_by_relative(observation: Observation, config: ModelConfig, num_players: int) -> list[float]:
    bids = {bid.player: bid.value for bid in observation.bids}
    values = []
    for rel in range(config.max_players):
        if rel >= num_players:
            values.append(0.0)
            continue
        player = (observation.player_id + rel) % num_players
        bid = bids.get(player)
        values.append(
            _normalize_signed(bid - observation.tricks_won.get(player, 0), config.max_hand_size)
            if bid is not None
            else 0.0
        )
    return values


def _cards_remaining_by_relative(observation: Observation, config: ModelConfig, num_players: int) -> list[float]:
    values = []
    for rel in range(config.max_players):
        if rel >= num_players:
            values.append(0.0)
            continue
        player = (observation.player_id + rel) % num_players
        remaining = observation.hand_size - len(observation.played_cards_by_player.get(player, []))
        values.append(_normalize(max(remaining, 0), config.max_hand_size))
    return values


def _voids_by_relative(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[float]]:
    rows = []
    for rel in range(config.max_players):
        player = (observation.player_id + rel) % num_players if rel < num_players else None
        rows.append(_void_list(observation, player))
    return rows


def _void_list(observation: Observation, player: int | None) -> list[float]:
    if player is None:
        return [0.0] * len(SUITS)
    return [1.0 if observation.voids.get(player, {}).get(suit, False) else 0.0 for suit in SUITS]


def _scalar_by_relative(
    values: dict[int, int],
    observer: int,
    num_players: int,
    max_players: int,
    denominator: float,
) -> list[float]:
    result = []
    for rel in range(max_players):
        player = (observer + rel) % num_players if rel < num_players else None
        result.append(_normalize(values.get(player, 0), denominator) if player is not None else 0.0)
    return result


def _bidding_position(
    observation: Observation,
    player: int | None,
    num_players: int,
    na_id: int,
) -> int:
    if player is None or player not in observation.bidding_order:
        return na_id
    position = observation.bidding_order.index(player)
    return position if position < num_players else na_id


def _relative_player_id(player: int | None, observer: int, num_players: int, na_id: int) -> int:
    return na_id if player is None else (player - observer) % num_players


def _relative_player_one_hot(
    player: int | None,
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[float]:
    return _one_hot(
        _relative_player_id(player, observation.player_id, num_players, config.player_na_id),
        config.max_players + 1,
    )


def _bid_status_one_hot(bid: int, config: ModelConfig) -> list[float]:
    index = 0 if bid < 0 else bid + 1
    return _one_hot(index, config.bid_feature_count)


def _suit_one_hot(suit: Suit | None, config: ModelConfig) -> list[float]:
    index = SUITS.index(suit) if suit is not None else config.suit_none_id
    return _one_hot(index, len(SUITS) + 1)


def _multi_hot_cards(cards: list[Card]) -> list[float]:
    values = [0.0] * NUM_CARDS
    for card in cards:
        values[card_id(card)] = 1.0
    return values


def _one_hot(index: int, size: int) -> list[float]:
    if index < 0 or index >= size:
        raise ValueError(f"One-hot index {index} outside size {size}.")
    values = [0.0] * size
    values[index] = 1.0
    return values


def _normalize(value: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(value) / float(denominator)


def _normalize_signed(value: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, float(value) / float(denominator)))
