"""Read-only schema-v1 observation encoding for legacy Plump checkpoints.

This module intentionally depends only on the standard library and the engine
types. PyTorch-specific batching and modules live in ``torch_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from plump.cards import Card, Rank, Suit
from plump.state import EventType, GameEvent, Observation, Phase


EVENT_TOKEN_WIDTH = 9
NUM_CARDS = 52

SUITS: tuple[Suit, ...] = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
RANKS: tuple[Rank, ...] = (
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
)
PHASES: tuple[Phase, ...] = (
    Phase.NOT_STARTED,
    Phase.BIDDING,
    Phase.PLAYING,
    Phase.ROUND_OVER,
    Phase.GAME_OVER,
)

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
    """Architecture and encoding limits for Plump neural agents."""

    max_players: int = 5
    max_hand_size: int = 13
    max_rounds: int = 64
    max_seq_len: int = 192
    d_model: int = 128
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 512
    context_hidden_dim: int = 256
    dropout: float = 0.1

    @property
    def bid_count(self) -> int:
        return self.max_hand_size + 1

    @property
    def bid_feature_count(self) -> int:
        # one slot for "has not bid yet", then bid values 0..max_hand_size
        return self.max_hand_size + 2

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
    def round_na_id(self) -> int:
        return self.max_rounds

    @property
    def trick_na_id(self) -> int:
        return self.max_hand_size + 1

    @property
    def pos_na_id(self) -> int:
        return self.max_players

    @property
    def context_dim(self) -> int:
        p = self.max_players
        return (
            NUM_CARDS  # my hand
            + NUM_CARDS  # legal cards
            + NUM_CARDS  # all played cards
            + p * NUM_CARDS  # played by each relative player
            + p * NUM_CARDS  # current trick cards by relative player
            + p * self.bid_feature_count  # bid status/value by player
            + p  # tricks won
            + p  # tricks needed
            + p  # cards remaining
            + p * len(SUITS)  # voids
            + p  # cumulative scores
            + len(SUITS)
            + 1  # trump, including no trump
            + self.bid_count  # hand size one-hot
            + len(PHASES)  # phase
            + p
            + 1  # current player, including N/A
            + p
            + 1  # current trick leader, including N/A
            + len(SUITS)
            + 1  # current trick led suit, including none
            + 1  # normalized round index
            + 1  # is last bidder
            + p  # active relative players
            + 1  # normalized number of players
        )

    @property
    def player_feature_dim(self) -> int:
        return (
            self.bid_feature_count
            + 1  # has bid
            + 1  # bid value normalized
            + 1  # tricks won normalized
            + 1  # tricks needed normalized
            + 1  # cards remaining normalized
            + len(SUITS)  # voids
            + NUM_CARDS  # cards played by this player
            + 1  # is current player
            + 1  # is observation player
            + 1  # active player slot
        )


@dataclass
class EncodedObservation:
    """Fixed-shape model input derived from one player-visible observation."""

    event_tokens: list[list[int]]
    event_valid_mask: list[bool]
    context_features: list[float]
    player_features: list[list[float]]
    active_player_mask: list[bool]
    legal_bid_mask: list[bool]
    legal_card_mask: list[bool]
    final_trick_count_mask: list[list[bool]]
    hand_belief_mask: list[list[bool]]
    bid_values: list[int]
    num_players: int
    observer_player: int
    current_player_relative: int


def card_id(card: Card) -> int:
    """Map a card to a stable 0..51 id."""

    return SUITS.index(card.suit) * len(RANKS) + RANKS.index(card.rank)


def card_from_id(index: int) -> Card:
    """Map a stable 0..51 id back to a card."""

    if index < 0 or index >= NUM_CARDS:
        raise ValueError(f"Card id must be in 0..{NUM_CARDS - 1}.")
    suit = SUITS[index // len(RANKS)]
    rank = RANKS[index % len(RANKS)]
    return Card(suit, rank)


def encode_observation(observation: Observation, config: ModelConfig | None = None) -> EncodedObservation:
    """Encode one player-visible observation for the transformer model.

    Player ids are converted to relative ids from ``observation.player_id``:
    relative player 0 is the observer, 1 is next clockwise, and so on. During
    policy use, pass ``env.get_observation(env.current_player())`` so relative
    player 0 is also the acting player.
    """

    config = config or ModelConfig()
    num_players = len(observation.scores)
    _validate_observation_limits(observation, config, num_players)

    event_tokens = _encode_events(observation.event_log, observation.player_id, num_players, config)
    event_valid_mask = [index < min(len(observation.event_log), config.max_seq_len) for index in range(config.max_seq_len)]

    legal_bid_mask = [False] * config.bid_count
    for bid in observation.legal_bids:
        if 0 <= bid < config.bid_count:
            legal_bid_mask[bid] = True

    legal_card_mask = [False] * NUM_CARDS
    for card in observation.legal_cards:
        legal_card_mask[card_id(card)] = True

    context_features = _encode_context(observation, config, num_players, legal_bid_mask, legal_card_mask)
    player_features, bid_values = _encode_player_features(observation, config, num_players)
    active_player_mask = [index < num_players for index in range(config.max_players)]
    final_trick_count_mask = _final_trick_count_mask_by_relative(observation, config, num_players)
    hand_belief_mask = _hand_belief_mask_by_relative(observation, config, num_players)
    current_player_relative = _relative_player_id(
        observation.current_player, observation.player_id, num_players, config.player_na_id
    )

    return EncodedObservation(
        event_tokens=event_tokens,
        event_valid_mask=event_valid_mask,
        context_features=context_features,
        player_features=player_features,
        active_player_mask=active_player_mask,
        legal_bid_mask=legal_bid_mask,
        legal_card_mask=legal_card_mask,
        final_trick_count_mask=final_trick_count_mask,
        hand_belief_mask=hand_belief_mask,
        bid_values=bid_values,
        num_players=num_players,
        observer_player=observation.player_id,
        current_player_relative=current_player_relative,
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


def _encode_events(
    event_log: list[GameEvent],
    observer: int,
    num_players: int,
    config: ModelConfig,
) -> list[list[int]]:
    visible_events = event_log[-config.max_seq_len :]
    tokens: list[list[int]] = []
    for event in visible_events:
        token = [
            EVENT_TYPE_IDS[event.type],
            _relative_player_id(event.player, observer, num_players, config.player_na_id),
            _rank_id(event.card, config),
            _suit_id(event.card.suit if event.card is not None else None, config),
            card_id(event.card) if event.card is not None else config.card_na_id,
            event.bid if event.bid is not None else config.bid_na_id,
            event.round_index,
            event.trick_index if event.trick_index is not None else config.trick_na_id,
            event.position_in_trick if event.position_in_trick is not None else config.pos_na_id,
        ]
        _validate_event_token(token, config)
        tokens.append(token)

    pad_token = [
        PAD_EVENT_TYPE_ID,
        config.player_na_id,
        config.rank_na_id,
        config.suit_none_id,
        config.card_na_id,
        config.bid_na_id,
        config.round_na_id,
        config.trick_na_id,
        config.pos_na_id,
    ]
    while len(tokens) < config.max_seq_len:
        tokens.append(list(pad_token))
    return tokens


def _validate_event_token(token: list[int], config: ModelConfig) -> None:
    limits = [
        NUM_EVENT_TYPES,
        config.max_players + 1,
        len(RANKS) + 1,
        len(SUITS) + 1,
        NUM_CARDS + 1,
        config.max_hand_size + 2,
        config.max_rounds + 1,
        config.max_hand_size + 2,
        config.max_players + 1,
    ]
    for value, limit in zip(token, limits):
        if value < 0 or value >= limit:
            raise ValueError(f"Event token value {value} outside embedding limit {limit}.")


def _encode_context(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
    legal_bid_mask: list[bool],
    legal_card_mask: list[bool],
) -> list[float]:
    played_by_relative = _cards_by_relative_player(observation.played_cards_by_player, observation, config, num_players)
    current_trick_by_relative = [[0.0] * NUM_CARDS for _ in range(config.max_players)]
    current_trick_leader = None
    current_trick_led_suit = None
    if observation.current_trick is not None:
        current_trick_leader = observation.current_trick.leader
        current_trick_led_suit = observation.current_trick.led_suit
        for play in observation.current_trick.plays:
            rel = _relative_player_id(play.player, observation.player_id, num_players, config.player_na_id)
            if rel < config.max_players:
                current_trick_by_relative[rel][card_id(play.card)] = 1.0

    bids = _bid_status_by_relative(observation, config, num_players)
    tricks_won = _scalar_by_relative(observation.tricks_won, observation, config, num_players, config.max_hand_size)
    tricks_needed = _tricks_needed_by_relative(observation, config, num_players)
    cards_remaining = _cards_remaining_by_relative(observation, config, num_players)
    voids = _voids_by_relative(observation, config, num_players)
    scores = _scalar_by_relative(observation.scores, observation, config, num_players, 100.0)

    features: list[float] = []
    features.extend(_multi_hot_cards(observation.my_hand))
    features.extend(float(value) for value in legal_card_mask)
    features.extend(_multi_hot_cards(observation.played_cards_total))
    for rel in range(config.max_players):
        features.extend(played_by_relative[rel])
    for rel in range(config.max_players):
        features.extend(current_trick_by_relative[rel])
    for rel in range(config.max_players):
        features.extend(bids[rel])
    features.extend(tricks_won)
    features.extend(tricks_needed)
    features.extend(cards_remaining)
    for rel in range(config.max_players):
        features.extend(voids[rel])
    features.extend(scores)
    features.extend(_suit_one_hot(observation.trump_suit, config))
    features.extend(_one_hot(observation.hand_size, config.bid_count))
    features.extend(_phase_one_hot(observation.phase))
    features.extend(
        _relative_player_one_hot(observation.current_player, observation.player_id, num_players, config)
    )
    features.extend(_relative_player_one_hot(current_trick_leader, observation.player_id, num_players, config))
    features.extend(_suit_one_hot(current_trick_led_suit, config))
    features.append(_normalize(observation.round_index, max(config.max_rounds - 1, 1)))
    features.append(1.0 if observation.phase == Phase.BIDDING and len(observation.bids) == num_players - 1 else 0.0)
    features.extend(1.0 if rel < num_players else 0.0 for rel in range(config.max_players))
    features.append(_normalize(num_players, config.max_players))

    if len(features) != config.context_dim:
        raise AssertionError(f"Context feature size {len(features)} != expected {config.context_dim}.")
    return features


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

        has_bid = bid_value >= 0
        tricks_won = observation.tricks_won.get(abs_player, 0) if abs_player is not None else 0
        tricks_needed = bid_value - tricks_won if has_bid else 0
        played_count = len(observation.played_cards_by_player.get(abs_player, [])) if abs_player is not None else 0
        cards_remaining = max(observation.hand_size - played_count, 0) if active else 0
        voids = _void_list(observation, abs_player) if abs_player is not None else [0.0] * len(SUITS)

        row: list[float] = []
        row.extend(_bid_status_one_hot(bid_value, config))
        row.append(1.0 if has_bid else 0.0)
        row.append(_normalize(bid_value if has_bid else 0, config.max_hand_size))
        row.append(_normalize(tricks_won, config.max_hand_size))
        row.append(_normalize_signed(tricks_needed, config.max_hand_size))
        row.append(_normalize(cards_remaining, config.max_hand_size))
        row.extend(voids)
        row.extend(played_by_relative[rel])
        row.append(
            1.0
            if observation.current_player is not None and abs_player is not None and observation.current_player == abs_player
            else 0.0
        )
        row.append(1.0 if rel == 0 and active else 0.0)
        row.append(1.0 if active else 0.0)

        if len(row) != config.player_feature_dim:
            raise AssertionError(f"Player feature size {len(row)} != expected {config.player_feature_dim}.")
        features.append(row)

    return features, bid_values


def _cards_by_relative_player(
    cards_by_abs_player: dict[int, list[Card]],
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[float]]:
    cards_by_relative = [[0.0] * NUM_CARDS for _ in range(config.max_players)]
    for abs_player, cards in cards_by_abs_player.items():
        rel = _relative_player_id(abs_player, observation.player_id, num_players, config.player_na_id)
        if rel < config.max_players:
            cards_by_relative[rel] = _multi_hot_cards(cards)
    return cards_by_relative


def _bid_status_by_relative(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[float]]:
    bids_by_abs = {bid.player: bid.value for bid in observation.bids}
    rows: list[list[float]] = []
    for rel in range(config.max_players):
        if rel >= num_players:
            rows.append(_bid_status_one_hot(-1, config))
            continue
        abs_player = (observation.player_id + rel) % num_players
        rows.append(_bid_status_one_hot(bids_by_abs.get(abs_player, -1), config))
    return rows


def _scalar_by_relative(
    values_by_abs_player: dict[int, int],
    observation: Observation,
    config: ModelConfig,
    num_players: int,
    denominator: float,
) -> list[float]:
    values: list[float] = []
    for rel in range(config.max_players):
        if rel >= num_players:
            values.append(0.0)
            continue
        abs_player = (observation.player_id + rel) % num_players
        values.append(_normalize(values_by_abs_player.get(abs_player, 0), denominator))
    return values


def _tricks_needed_by_relative(observation: Observation, config: ModelConfig, num_players: int) -> list[float]:
    bids_by_abs = {bid.player: bid.value for bid in observation.bids}
    values: list[float] = []
    for rel in range(config.max_players):
        if rel >= num_players:
            values.append(0.0)
            continue
        abs_player = (observation.player_id + rel) % num_players
        bid_value = bids_by_abs.get(abs_player)
        if bid_value is None:
            values.append(0.0)
        else:
            values.append(_normalize_signed(bid_value - observation.tricks_won.get(abs_player, 0), config.max_hand_size))
    return values


def _cards_remaining_by_relative(observation: Observation, config: ModelConfig, num_players: int) -> list[float]:
    values: list[float] = []
    for rel in range(config.max_players):
        if rel >= num_players:
            values.append(0.0)
            continue
        abs_player = (observation.player_id + rel) % num_players
        played_count = len(observation.played_cards_by_player.get(abs_player, []))
        values.append(_normalize(max(observation.hand_size - played_count, 0), config.max_hand_size))
    return values


def _voids_by_relative(observation: Observation, config: ModelConfig, num_players: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for rel in range(config.max_players):
        if rel >= num_players:
            rows.append([0.0] * len(SUITS))
            continue
        abs_player = (observation.player_id + rel) % num_players
        rows.append(_void_list(observation, abs_player))
    return rows


def _void_list(observation: Observation, abs_player: int | None) -> list[float]:
    if abs_player is None:
        return [0.0] * len(SUITS)
    return [1.0 if observation.voids.get(abs_player, {}).get(suit, False) else 0.0 for suit in SUITS]


def _final_trick_count_mask_by_relative(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[bool]]:
    rows: list[list[bool]] = []
    for rel in range(config.max_players):
        row = [False] * config.bid_count
        if rel >= num_players:
            rows.append(row)
            continue
        high = min(observation.hand_size, config.bid_count - 1)
        for count in range(high + 1):
            row[count] = True
        rows.append(row)
    return rows


def _hand_belief_mask_by_relative(
    observation: Observation,
    config: ModelConfig,
    num_players: int,
) -> list[list[bool]]:
    own_cards = {card_id(card) for card in observation.my_hand}
    played_cards = {card_id(card) for card in observation.played_cards_total}
    publicly_impossible = own_cards | played_cards
    rows: list[list[bool]] = []

    for rel in range(config.max_players):
        row = [False] * NUM_CARDS
        if rel == 0 or rel >= num_players:
            rows.append(row)
            continue

        abs_player = (observation.player_id + rel) % num_players
        void_suits = {
            suit
            for suit, is_void in observation.voids.get(abs_player, {}).items()
            if is_void
        }
        for card_index in range(NUM_CARDS):
            card = card_from_id(card_index)
            if card_index in publicly_impossible:
                continue
            if card.suit in void_suits:
                continue
            row[card_index] = True
        rows.append(row)

    return rows


def _relative_player_id(abs_player: int | None, observer: int, num_players: int, na_id: int) -> int:
    if abs_player is None:
        return na_id
    return (abs_player - observer) % num_players


def _relative_player_one_hot(
    abs_player: int | None,
    observer: int,
    num_players: int,
    config: ModelConfig,
) -> list[float]:
    return _one_hot(_relative_player_id(abs_player, observer, num_players, config.player_na_id), config.max_players + 1)


def _rank_id(card: Card | None, config: ModelConfig) -> int:
    return RANKS.index(card.rank) if card is not None else config.rank_na_id


def _suit_id(suit: Suit | None, config: ModelConfig) -> int:
    return SUITS.index(suit) if suit is not None else config.suit_none_id


def _multi_hot_cards(cards: list[Card]) -> list[float]:
    values = [0.0] * NUM_CARDS
    for card in cards:
        values[card_id(card)] = 1.0
    return values


def _bid_status_one_hot(bid_value: int, config: ModelConfig) -> list[float]:
    if bid_value < 0:
        index = 0
    else:
        if bid_value > config.max_hand_size:
            raise ValueError(f"Bid {bid_value} exceeds model max hand size {config.max_hand_size}.")
        index = bid_value + 1
    return _one_hot(index, config.bid_feature_count)


def _suit_one_hot(suit: Suit | None, config: ModelConfig) -> list[float]:
    return _one_hot(_suit_id(suit, config), len(SUITS) + 1)


def _phase_one_hot(phase: Phase) -> list[float]:
    return _one_hot(PHASES.index(phase), len(PHASES))


def _one_hot(index: int, size: int) -> list[float]:
    if index < 0 or index >= size:
        raise ValueError(f"One-hot index {index} outside size {size}.")
    values = [0.0] * size
    values[index] = 1.0
    return values


def _normalize(value: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(value) / float(denominator)


def _normalize_signed(value: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, float(value) / float(denominator)))
