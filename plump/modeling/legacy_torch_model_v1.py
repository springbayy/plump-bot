"""Read-only schema-v1 PyTorch model for legacy Plump checkpoints.

This is intentionally architecture-only: no replay buffers, trainers, losses,
or optimization loops. It consumes encoded player observations and returns
masked action logits plus auxiliary per-player predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only without torch.
    raise ImportError("plump.modeling.torch_model requires PyTorch. Run `uv sync`.") from exc

from .legacy_encoding_v1 import (
    EVENT_TOKEN_WIDTH,
    NUM_CARDS,
    NUM_EVENT_TYPES,
    RANKS,
    SUITS,
    EncodedObservation,
    ModelConfig,
)


@dataclass
class ModelBatch:
    """Batched tensor inputs for ``PlumpTransformerModel``."""

    event_tokens: Tensor
    event_valid_mask: Tensor
    context_features: Tensor
    player_features: Tensor
    active_player_mask: Tensor
    legal_bid_mask: Tensor
    legal_card_mask: Tensor
    final_trick_count_mask: Tensor
    hand_belief_mask: Tensor
    bid_values: Tensor


@dataclass
class PlumpModelOutput:
    """Forward-pass outputs.

    ``masked_bid_logits`` and ``masked_card_logits`` are the policy logits to
    sample from. Unmasked logits are retained for diagnostics and losses.
    """

    state: Tensor
    player_state: Tensor
    bid_logits: Tensor
    card_logits: Tensor
    masked_bid_logits: Tensor
    masked_card_logits: Tensor
    value: Tensor
    trick_count_logits: Tensor
    masked_trick_count_logits: Tensor
    point_logits: Tensor
    point_probs: Tensor
    hand_logits: Tensor
    masked_hand_logits: Tensor
    hand_probs: Tensor
    hit_bid_probs: Tensor


def best_torch_device() -> torch.device:
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encoded_observations_to_batch(
    observations: Sequence[EncodedObservation],
    *,
    device: torch.device | str | None = None,
) -> ModelBatch:
    """Convert encoded observations into tensors for the model."""

    if not observations:
        raise ValueError("Cannot build a batch from zero observations.")

    device = torch.device(device) if device is not None else best_torch_device()
    return ModelBatch(
        event_tokens=torch.tensor([obs.event_tokens for obs in observations], dtype=torch.long, device=device),
        event_valid_mask=torch.tensor([obs.event_valid_mask for obs in observations], dtype=torch.bool, device=device),
        context_features=torch.tensor([obs.context_features for obs in observations], dtype=torch.float32, device=device),
        player_features=torch.tensor([obs.player_features for obs in observations], dtype=torch.float32, device=device),
        active_player_mask=torch.tensor([obs.active_player_mask for obs in observations], dtype=torch.bool, device=device),
        legal_bid_mask=torch.tensor([obs.legal_bid_mask for obs in observations], dtype=torch.bool, device=device),
        legal_card_mask=torch.tensor([obs.legal_card_mask for obs in observations], dtype=torch.bool, device=device),
        final_trick_count_mask=torch.tensor(
            [obs.final_trick_count_mask for obs in observations], dtype=torch.bool, device=device
        ),
        hand_belief_mask=torch.tensor(
            [obs.hand_belief_mask for obs in observations], dtype=torch.bool, device=device
        ),
        bid_values=torch.tensor([obs.bid_values for obs in observations], dtype=torch.long, device=device),
    )


class PlumpTransformerModel(nn.Module):
    """Transformer policy/value architecture for Plump.

    Input:
        encoded public event history, current context features, per-player
        features, and legal action masks.

    Output:
        masked bid/card logits for the acting player, a scalar value estimate,
        and shared per-player auxiliary predictions.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.type_emb = nn.Embedding(NUM_EVENT_TYPES, cfg.d_model)
        self.player_emb = nn.Embedding(cfg.max_players + 1, cfg.d_model)
        self.rank_emb = nn.Embedding(len(RANKS) + 1, cfg.d_model)
        self.suit_emb = nn.Embedding(len(SUITS) + 1, cfg.d_model)
        self.card_emb = nn.Embedding(NUM_CARDS + 1, cfg.d_model)
        self.bid_emb = nn.Embedding(cfg.max_hand_size + 2, cfg.d_model)
        self.round_emb = nn.Embedding(cfg.max_rounds + 1, cfg.d_model)
        self.trick_emb = nn.Embedding(cfg.max_hand_size + 2, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_players + 1, cfg.d_model)
        self.abs_pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.context_mlp = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.context_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.context_hidden_dim, cfg.d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(cfg.d_model)

        self.bid_head = nn.Linear(cfg.d_model, cfg.bid_count)
        self.card_head = nn.Linear(cfg.d_model, NUM_CARDS)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

        self.player_query_emb = nn.Embedding(cfg.max_players, cfg.d_model)
        self.player_mlp = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_model + cfg.player_feature_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.trick_count_head = nn.Linear(cfg.d_model, cfg.bid_count)
        self.point_head = nn.Linear(cfg.d_model, 1)
        self.hand_head = nn.Linear(cfg.d_model, NUM_CARDS)

    def forward(self, batch: ModelBatch) -> PlumpModelOutput:
        """Run a forward pass from encoded observations to policy/value heads."""

        self._validate_batch(batch)

        event_emb = self._embed_events(batch.event_tokens)
        context_emb = self.context_mlp(batch.context_features).unsqueeze(1)
        x = torch.cat([context_emb, event_emb], dim=1)

        context_valid = torch.ones(
            batch.event_valid_mask.shape[0],
            1,
            dtype=torch.bool,
            device=batch.event_valid_mask.device,
        )
        valid_mask = torch.cat([context_valid, batch.event_valid_mask], dim=1)
        padding_mask = ~valid_mask

        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        state = self.final_norm(hidden[:, 0, :])

        bid_logits = self.bid_head(state)
        card_logits = self.card_head(state)
        masked_bid_logits = _masked_logits(bid_logits, batch.legal_bid_mask)
        masked_card_logits = _masked_logits(card_logits, batch.legal_card_mask)
        value = self.value_head(state)

        player_state = self._player_states(state, batch.player_features)
        trick_count_logits = self.trick_count_head(player_state)
        masked_trick_count_logits = _masked_logits(trick_count_logits, batch.final_trick_count_mask)
        point_logits = self.point_head(player_state).squeeze(-1)
        point_probs = torch.sigmoid(point_logits).masked_fill(~batch.active_player_mask, 0.0)
        hand_logits = self.hand_head(player_state)
        masked_hand_logits = _masked_logits(hand_logits, batch.hand_belief_mask)
        hand_probs = torch.sigmoid(masked_hand_logits).masked_fill(~batch.hand_belief_mask, 0.0)
        hit_bid_probs = self._hit_bid_probs(
            masked_trick_count_logits,
            batch.bid_values,
            batch.active_player_mask,
        )

        return PlumpModelOutput(
            state=state,
            player_state=player_state,
            bid_logits=bid_logits,
            card_logits=card_logits,
            masked_bid_logits=masked_bid_logits,
            masked_card_logits=masked_card_logits,
            value=value,
            trick_count_logits=trick_count_logits,
            masked_trick_count_logits=masked_trick_count_logits,
            point_logits=point_logits,
            point_probs=point_probs,
            hand_logits=hand_logits,
            masked_hand_logits=masked_hand_logits,
            hand_probs=hand_probs,
            hit_bid_probs=hit_bid_probs,
        )

    def _embed_events(self, event_tokens: Tensor) -> Tensor:
        t = event_tokens
        event_emb = (
            self.type_emb(t[..., 0])
            + self.player_emb(t[..., 1])
            + self.rank_emb(t[..., 2])
            + self.suit_emb(t[..., 3])
            + self.card_emb(t[..., 4])
            + self.bid_emb(t[..., 5])
            + self.round_emb(t[..., 6])
            + self.trick_emb(t[..., 7])
            + self.pos_emb(t[..., 8])
        )
        pos = torch.arange(t.shape[1], device=t.device).unsqueeze(0).expand(t.shape[0], t.shape[1])
        return event_emb + self.abs_pos_emb(pos)

    def _player_states(self, state: Tensor, player_features: Tensor) -> Tensor:
        batch_size = state.shape[0]
        player_ids = torch.arange(self.config.max_players, device=state.device)
        player_emb = self.player_query_emb(player_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_expanded = state.unsqueeze(1).expand(-1, self.config.max_players, -1)
        player_input = torch.cat([state_expanded, player_emb, player_features], dim=-1)
        return self.player_mlp(player_input)

    def _hit_bid_probs(self, trick_count_logits: Tensor, bid_values: Tensor, active_mask: Tensor) -> Tensor:
        safe_logits = trick_count_logits.masked_fill(~active_mask[:, :, None], 0.0)
        probs = torch.softmax(safe_logits, dim=-1)
        has_bid = (bid_values >= 0) & active_mask
        safe_bids = bid_values.clamp(min=0, max=self.config.bid_count - 1)
        hit_probs = probs.gather(dim=-1, index=safe_bids.unsqueeze(-1)).squeeze(-1)
        return hit_probs.masked_fill(~has_bid, 0.0)

    def _validate_batch(self, batch: ModelBatch) -> None:
        if batch.event_tokens.ndim != 3 or batch.event_tokens.shape[-1] != EVENT_TOKEN_WIDTH:
            raise ValueError(f"event_tokens must have shape [B, L, {EVENT_TOKEN_WIDTH}].")
        if batch.event_tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("event sequence is longer than model max_seq_len.")
        if batch.context_features.shape[-1] != self.config.context_dim:
            raise ValueError("context_features has the wrong final dimension.")
        if batch.player_features.shape[1:] != (self.config.max_players, self.config.player_feature_dim):
            raise ValueError("player_features has the wrong shape.")
        if batch.final_trick_count_mask.shape != (
            batch.event_tokens.shape[0],
            self.config.max_players,
            self.config.bid_count,
        ):
            raise ValueError("final_trick_count_mask has the wrong shape.")
        if batch.hand_belief_mask.shape != (
            batch.event_tokens.shape[0],
            self.config.max_players,
            NUM_CARDS,
        ):
            raise ValueError("hand_belief_mask has the wrong shape.")


def _masked_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    neg = torch.finfo(logits.dtype).min
    return logits.masked_fill(~legal_mask, neg)
