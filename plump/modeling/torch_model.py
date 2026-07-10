"""Schema-v4 PyTorch architecture for round-local and full-game Plump agents."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from typing import Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("plump.modeling.torch_model requires PyTorch. Run `uv sync`.") from exc

from .encoding import (
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
    event_tokens: Tensor
    event_valid_mask: Tensor
    context_features: Tensor
    game_context_mask: Tensor
    player_features: Tensor
    active_player_mask: Tensor
    legal_bid_mask: Tensor
    legal_card_mask: Tensor
    final_trick_count_mask: Tensor
    owner_valid_mask: Tensor
    owner_capacities: Tensor
    bid_values: Tensor
    game_context_features: Tensor
    schedule_hand_sizes: Tensor
    schedule_statuses: Tensor
    schedule_valid_mask: Tensor


@dataclass
class PlumpModelOutput:
    state: Tensor
    player_state: Tensor
    bid_logits: Tensor
    card_logits: Tensor
    masked_bid_logits: Tensor
    masked_card_logits: Tensor
    value: Tensor
    round_value: Tensor
    game_value: Tensor
    trick_count_logits: Tensor
    masked_trick_count_logits: Tensor
    # Owner outputs are None when the forward pass was asked to skip the
    # owner head (action sampling needs only logits and value).
    owner_logits: Tensor | None
    masked_owner_logits: Tensor | None
    owner_pre_sinkhorn_probs: Tensor | None
    owner_probs: Tensor | None
    hit_bid_probs: Tensor
    score_probs: Tensor
    bid_q_values: Tensor | None = None
    card_q_values: Tensor | None = None
    masked_bid_q_values: Tensor | None = None
    masked_card_q_values: Tensor | None = None


def best_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_autocast(device: torch.device, precision: str):
    """Return the requested accelerator autocast context."""

    if precision == "fp32":
        return nullcontext()
    dtype_by_precision = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if precision not in dtype_by_precision:
        raise ValueError("precision must be one of: fp32, bf16, fp16.")
    return torch.autocast(device_type=device.type, dtype=dtype_by_precision[precision])


def encoded_observations_to_batch(
    observations: Sequence[EncodedObservation],
    *,
    device: torch.device | str | None = None,
) -> ModelBatch:
    if not observations:
        raise ValueError("Cannot build a batch from zero observations.")
    device = torch.device(device) if device is not None else best_torch_device()
    return ModelBatch(
        event_tokens=torch.tensor([obs.event_tokens for obs in observations], dtype=torch.long, device=device),
        event_valid_mask=torch.tensor([obs.event_valid_mask for obs in observations], dtype=torch.bool, device=device),
        context_features=torch.tensor([obs.context_features for obs in observations], dtype=torch.float32, device=device),
        game_context_mask=torch.tensor(
            [obs.game_context_enabled for obs in observations], dtype=torch.bool, device=device
        ),
        player_features=torch.tensor([obs.player_features for obs in observations], dtype=torch.float32, device=device),
        active_player_mask=torch.tensor([obs.active_player_mask for obs in observations], dtype=torch.bool, device=device),
        legal_bid_mask=torch.tensor([obs.legal_bid_mask for obs in observations], dtype=torch.bool, device=device),
        legal_card_mask=torch.tensor([obs.legal_card_mask for obs in observations], dtype=torch.bool, device=device),
        final_trick_count_mask=torch.tensor(
            [obs.final_trick_count_mask for obs in observations], dtype=torch.bool, device=device
        ),
        owner_valid_mask=torch.tensor(
            [obs.owner_valid_mask for obs in observations], dtype=torch.bool, device=device
        ),
        owner_capacities=torch.tensor(
            [obs.owner_capacities for obs in observations],
            dtype=torch.float32,
            device=device,
        ),
        bid_values=torch.tensor([obs.bid_values for obs in observations], dtype=torch.long, device=device),
        game_context_features=torch.tensor(
            [obs.game_context_features for obs in observations],
            dtype=torch.float32,
            device=device,
        ),
        schedule_hand_sizes=torch.tensor(
            [obs.schedule_hand_sizes for obs in observations],
            dtype=torch.long,
            device=device,
        ),
        schedule_statuses=torch.tensor(
            [obs.schedule_statuses for obs in observations],
            dtype=torch.long,
            device=device,
        ),
        schedule_valid_mask=torch.tensor(
            [obs.schedule_valid_mask for obs in observations],
            dtype=torch.bool,
            device=device,
        ),
    )


def index_model_batch(batch: ModelBatch, indices: Tensor) -> ModelBatch:
    """Select a device minibatch from a staged rollout batch."""

    return ModelBatch(
        event_tokens=batch.event_tokens.index_select(0, indices),
        event_valid_mask=batch.event_valid_mask.index_select(0, indices),
        context_features=batch.context_features.index_select(0, indices),
        game_context_mask=batch.game_context_mask.index_select(0, indices),
        player_features=batch.player_features.index_select(0, indices),
        active_player_mask=batch.active_player_mask.index_select(0, indices),
        legal_bid_mask=batch.legal_bid_mask.index_select(0, indices),
        legal_card_mask=batch.legal_card_mask.index_select(0, indices),
        final_trick_count_mask=batch.final_trick_count_mask.index_select(0, indices),
        owner_valid_mask=batch.owner_valid_mask.index_select(0, indices),
        owner_capacities=batch.owner_capacities.index_select(0, indices),
        bid_values=batch.bid_values.index_select(0, indices),
        game_context_features=batch.game_context_features.index_select(0, indices),
        schedule_hand_sizes=batch.schedule_hand_sizes.index_select(0, indices),
        schedule_statuses=batch.schedule_statuses.index_select(0, indices),
        schedule_valid_mask=batch.schedule_valid_mask.index_select(0, indices),
    )


def combined_action_logits(output: PlumpModelOutput, bid_mask: Tensor) -> Tensor:
    """Return one categorical action space for mixed bid and play batches."""

    bid_logits = F.pad(
        output.masked_bid_logits.float(),
        (0, NUM_CARDS - output.masked_bid_logits.shape[-1]),
        value=float("-inf"),
    )
    return torch.where(
        bid_mask.unsqueeze(-1),
        bid_logits,
        output.masked_card_logits.float(),
    )


def slice_model_output(
    output: PlumpModelOutput,
    size: int,
) -> PlumpModelOutput:
    """Remove inference padding from every batched model output."""

    replacements = {}
    for field in fields(output):
        value = getattr(output, field.name)
        if isinstance(value, Tensor) and value.ndim > 0:
            replacements[field.name] = value[:size]
    return replace(output, **replacements)


class PlumpTransformerModel(nn.Module):
    """Shared round policy with residual value and supervised belief heads."""

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
        self.trick_emb = nn.Embedding(cfg.max_hand_size + 2, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_players + 1, cfg.d_model)
        self.abs_pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.game_context_emb = nn.Embedding(2, cfg.d_model)
        self.game_context_mlp = nn.Sequential(
            nn.Linear(cfg.game_feature_dim, cfg.game_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.game_hidden_dim, cfg.d_model),
        )
        self.schedule_hand_emb = nn.Embedding(cfg.max_hand_size + 2, cfg.game_hidden_dim)
        self.schedule_pos_emb = nn.Embedding(cfg.max_rounds, cfg.game_hidden_dim)
        self.schedule_status_emb = nn.Embedding(4, cfg.game_hidden_dim)
        schedule_layer = nn.TransformerEncoderLayer(
            d_model=cfg.game_hidden_dim,
            nhead=cfg.schedule_heads,
            dim_feedforward=4 * cfg.game_hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.schedule_encoder = nn.TransformerEncoder(
            schedule_layer,
            num_layers=cfg.schedule_layers,
            enable_nested_tensor=False,
        )
        self.schedule_projection = nn.Sequential(
            nn.LayerNorm(cfg.game_hidden_dim),
            nn.Linear(cfg.game_hidden_dim, cfg.d_model),
        )

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
        self.game_value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

        self.player_query_emb = nn.Embedding(cfg.max_players, cfg.d_model)
        self.player_mlp = nn.Sequential(
            nn.Linear(2 * cfg.d_model + cfg.player_feature_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.trick_count_head = nn.Linear(cfg.d_model, cfg.bid_count)

        self.owner_card_emb = nn.Embedding(NUM_CARDS, cfg.d_model)
        self.owner_card_mlp = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.owner_class_emb = nn.Embedding(
            cfg.owner_class_count,
            cfg.d_model,
        )
        self.owner_capacity_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, batch: ModelBatch, *, need_owner: bool = True) -> PlumpModelOutput:
        self._validate_batch(batch)
        event_emb = self._embed_events(batch.event_tokens)
        context_emb = self.context_mlp(batch.context_features)
        context_emb = context_emb + self.game_context_emb(batch.game_context_mask.long())
        if bool(batch.game_context_mask.any()):
            game_context = self.game_context_mlp(batch.game_context_features)
            game_context = game_context + self._schedule_context(batch)
            context_emb = context_emb + torch.where(
                batch.game_context_mask.unsqueeze(-1),
                game_context,
                torch.zeros_like(game_context),
            )
        x = torch.cat([context_emb.unsqueeze(1), event_emb], dim=1)

        context_valid = torch.ones(
            batch.event_valid_mask.shape[0], 1, dtype=torch.bool, device=batch.event_valid_mask.device
        )
        padding_mask = ~torch.cat([context_valid, batch.event_valid_mask], dim=1)
        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        state = self.final_norm(hidden[:, 0, :])

        bid_logits = self.bid_head(state)
        card_logits = self.card_head(state)
        masked_bid_logits = _masked_logits(bid_logits, batch.legal_bid_mask)
        masked_card_logits = _masked_logits(card_logits, batch.legal_card_mask)
        round_value = self.value_head(state)
        game_value = self.game_value_head(state)
        value = torch.where(
            batch.game_context_mask.unsqueeze(-1),
            game_value,
            round_value,
        )

        player_state = self._player_states(state, batch.player_features)
        trick_count_logits = self.trick_count_head(player_state)
        masked_trick_count_logits = _masked_logits(trick_count_logits, batch.final_trick_count_mask)

        owner_logits = None
        masked_owner_logits = None
        owner_pre_sinkhorn_probs = None
        owner_probs = None
        if need_owner:
            owner_logits = self._owner_logits(
                state,
                batch.owner_capacities,
            )
            masked_owner_logits = _masked_logits(owner_logits, batch.owner_valid_mask)
            owner_has_valid = batch.owner_valid_mask.any(dim=-1)
            owner_pre_sinkhorn_probs = (
                torch.softmax(masked_owner_logits.float(), dim=-1)
                .masked_fill(~owner_has_valid.unsqueeze(-1), 0.0)
            )
            owner_probs = masked_capacity_sinkhorn(
                owner_logits,
                batch.owner_valid_mask,
                batch.owner_capacities,
                iterations=self.config.owner_sinkhorn_iterations,
            )

        hit_bid_probs = self._hit_bid_probs(
            masked_trick_count_logits, batch.bid_values, batch.active_player_mask
        )
        return PlumpModelOutput(
            state=state,
            player_state=player_state,
            bid_logits=bid_logits,
            card_logits=card_logits,
            masked_bid_logits=masked_bid_logits,
            masked_card_logits=masked_card_logits,
            value=value,
            round_value=round_value,
            game_value=game_value,
            trick_count_logits=trick_count_logits,
            masked_trick_count_logits=masked_trick_count_logits,
            owner_logits=owner_logits,
            masked_owner_logits=masked_owner_logits,
            owner_pre_sinkhorn_probs=owner_pre_sinkhorn_probs,
            owner_probs=owner_probs,
            hit_bid_probs=hit_bid_probs,
            score_probs=hit_bid_probs,
        )

    def _schedule_context(self, batch: ModelBatch) -> Tensor:
        positions = torch.arange(
            self.config.max_rounds,
            device=batch.schedule_hand_sizes.device,
        ).unsqueeze(0)
        schedule = (
            self.schedule_hand_emb(batch.schedule_hand_sizes)
            + self.schedule_pos_emb(positions)
            + self.schedule_status_emb(batch.schedule_statuses)
        )
        safe_valid = batch.schedule_valid_mask.clone()
        empty = ~safe_valid.any(dim=-1)
        if empty.any():
            safe_valid[empty, 0] = True
        encoded = self.schedule_encoder(
            schedule,
            src_key_padding_mask=~safe_valid,
        )
        weights = batch.schedule_valid_mask.float()
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.schedule_projection(pooled)

    def _embed_events(self, event_tokens: Tensor) -> Tensor:
        token = event_tokens
        embedded = (
            self.type_emb(token[..., 0])
            + self.player_emb(token[..., 1])
            + self.rank_emb(token[..., 2])
            + self.suit_emb(token[..., 3])
            + self.card_emb(token[..., 4])
            + self.bid_emb(token[..., 5])
            + self.trick_emb(token[..., 6])
            + self.pos_emb(token[..., 7])
        )
        positions = torch.arange(token.shape[1], device=token.device).unsqueeze(0).expand(token.shape[0], -1)
        return embedded + self.abs_pos_emb(positions)

    def _player_states(self, state: Tensor, player_features: Tensor) -> Tensor:
        batch_size = state.shape[0]
        player_ids = torch.arange(self.config.max_players, device=state.device)
        player_embeddings = self.player_query_emb(player_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_expanded = state.unsqueeze(1).expand(-1, self.config.max_players, -1)
        return self.player_mlp(torch.cat([state_expanded, player_embeddings, player_features], dim=-1))

    def _owner_logits(
        self,
        state: Tensor,
        capacities: Tensor,
    ) -> Tensor:
        batch_size = state.shape[0]
        card_ids = torch.arange(NUM_CARDS, device=state.device)
        card_embeddings = self.owner_card_emb(card_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_expanded = state.unsqueeze(1).expand(-1, NUM_CARDS, -1)
        card_state = self.owner_card_mlp(
            torch.cat([state_expanded, card_embeddings], dim=-1)
        )

        owner_ids = torch.arange(
            self.config.owner_class_count,
            device=state.device,
        )
        owner_embeddings = self.owner_class_emb(owner_ids).unsqueeze(0)
        total_hidden = capacities.sum(dim=-1, keepdim=True).clamp_min(1.0)
        capacity_features = torch.stack(
            (
                capacities / float(NUM_CARDS),
                capacities / total_hidden,
            ),
            dim=-1,
        )
        owner_state = owner_embeddings + self.owner_capacity_mlp(
            capacity_features
        )
        return torch.einsum(
            "bcd,bod->bco",
            card_state.float(),
            owner_state.float(),
        ) / math.sqrt(self.config.d_model)

    def _hit_bid_probs(self, trick_logits: Tensor, bid_values: Tensor, active_mask: Tensor) -> Tensor:
        safe_logits = trick_logits.masked_fill(~active_mask[:, :, None], 0.0)
        probs = torch.softmax(safe_logits, dim=-1)
        has_bid = (bid_values >= 0) & active_mask
        safe_bids = bid_values.clamp(min=0, max=self.config.bid_count - 1)
        hits = probs.gather(dim=-1, index=safe_bids.unsqueeze(-1)).squeeze(-1)
        return hits.masked_fill(~has_bid, 0.0)

    def _validate_batch(self, batch: ModelBatch) -> None:
        batch_size = batch.event_tokens.shape[0]
        if batch.event_tokens.ndim != 3 or batch.event_tokens.shape[-1] != EVENT_TOKEN_WIDTH:
            raise ValueError(f"event_tokens must have shape [B, L, {EVENT_TOKEN_WIDTH}].")
        if batch.event_tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("event sequence is longer than model max_seq_len.")
        if batch.context_features.shape != (batch_size, self.config.context_dim):
            raise ValueError("context_features has the wrong shape.")
        if batch.game_context_mask.shape != (batch_size,):
            raise ValueError("game_context_mask has the wrong shape.")
        if batch.game_context_features.shape != (
            batch_size,
            self.config.game_feature_dim,
        ):
            raise ValueError("game_context_features has the wrong shape.")
        schedule_shape = (batch_size, self.config.max_rounds)
        if batch.schedule_hand_sizes.shape != schedule_shape:
            raise ValueError("schedule_hand_sizes has the wrong shape.")
        if batch.schedule_statuses.shape != schedule_shape:
            raise ValueError("schedule_statuses has the wrong shape.")
        if batch.schedule_valid_mask.shape != schedule_shape:
            raise ValueError("schedule_valid_mask has the wrong shape.")
        if batch.player_features.shape[1:] != (self.config.max_players, self.config.player_feature_dim):
            raise ValueError("player_features has the wrong shape.")
        if batch.final_trick_count_mask.shape[1:] != (
            self.config.max_players,
            self.config.bid_count,
        ):
            raise ValueError("final_trick_count_mask has the wrong shape.")
        if batch.owner_valid_mask.shape[1:] != (NUM_CARDS, self.config.owner_class_count):
            raise ValueError("owner_valid_mask has the wrong shape.")
        if batch.owner_capacities.shape != (
            batch_size,
            self.config.owner_class_count,
        ):
            raise ValueError("owner_capacities has the wrong shape.")
        hidden_rows = batch.owner_valid_mask.any(dim=-1).sum(dim=-1)
        if not torch.equal(
            hidden_rows,
            batch.owner_capacities.sum(dim=-1).long(),
        ):
            raise ValueError(
                "owner_capacities must sum to the hidden-card count."
            )


class PlumpSearchModel(PlumpTransformerModel):
    """Schema-v5 expert-iteration model with explicit legal-action Q heads."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        cfg = self.config
        self.bid_q_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.bid_count),
        )
        self.card_q_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, NUM_CARDS),
        )

    def forward(self, batch: ModelBatch, *, need_owner: bool = True) -> PlumpModelOutput:
        output = super().forward(batch, need_owner=need_owner)
        bid_q_values = self.bid_q_head(output.state).float()
        card_q_values = self.card_q_head(output.state).float()
        return replace(
            output,
            bid_q_values=bid_q_values,
            card_q_values=card_q_values,
            masked_bid_q_values=_masked_logits(
                bid_q_values,
                batch.legal_bid_mask,
            ),
            masked_card_q_values=_masked_logits(
                card_q_values,
                batch.legal_card_mask,
            ),
        )


def _masked_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def masked_capacity_sinkhorn(
    logits: Tensor,
    valid_mask: Tensor,
    capacities: Tensor,
    *,
    iterations: int,
) -> Tensor:
    """Project card-owner scores onto masked row and capacity constraints."""

    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive.")
    mask = valid_mask.bool()
    row_active = mask.any(dim=-1, keepdim=True)
    safe_logits = logits.float().masked_fill(~mask, -1e9)
    row_max = safe_logits.max(dim=-1, keepdim=True).values
    row_max = torch.where(row_active, row_max, torch.zeros_like(row_max))
    transport = (
        torch.exp(safe_logits - row_max)
        * mask.float()
    )
    target_columns = capacities.float().clamp_min(0.0)
    for _ in range(iterations):
        row_sums = transport.sum(dim=-1, keepdim=True)
        transport = torch.where(
            row_active,
            transport / row_sums.clamp_min(1e-12),
            torch.zeros_like(transport),
        )
        column_sums = transport.sum(dim=1)
        column_scale = torch.where(
            target_columns > 0.0,
            target_columns / column_sums.clamp_min(1e-12),
            torch.zeros_like(target_columns),
        )
        transport = transport * column_scale.unsqueeze(1)
        transport = transport * mask.float()
    return transport


V3_GAME_PARAMETER_PREFIXES = (
    "game_context_mlp.",
    "schedule_hand_emb.",
    "schedule_pos_emb.",
    "schedule_status_emb.",
    "schedule_encoder.",
    "schedule_projection.",
    "game_value_head.",
)

V4_OWNER_PARAMETER_PREFIXES = (
    "owner_card_emb.",
    "owner_card_mlp.",
    "owner_class_emb.",
    "owner_capacity_mlp.",
)

PRE_V4_OWNER_PARAMETER_PREFIXES = (
    "owner_card_emb.",
    "owner_head.",
)

V5_Q_PARAMETER_PREFIXES = (
    "bid_q_head.",
    "card_q_head.",
)


def load_v4_weights(
    model: "PlumpSearchModel",
    state_dict: Mapping[str, Tensor],
) -> dict[str, list[str]]:
    """Warm-start a schema-v5 search model from v4 with fresh Q heads."""

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    unexpected = sorted(key for key in state_dict if key not in current)
    mismatched = sorted(
        key
        for key, value in state_dict.items()
        if key in current and current[key].shape != value.shape
    )
    missing = sorted(key for key in current if key not in compatible)
    invalid_missing = [
        key
        for key in missing
        if not key.startswith(V5_Q_PARAMETER_PREFIXES)
    ]
    if unexpected or mismatched or invalid_missing:
        raise ValueError(
            "V4 warm start is not shape-compatible: "
            f"unexpected={unexpected}, mismatched={mismatched}, "
            f"invalid_missing={invalid_missing}"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "fresh": missing,
        "dropped": [],
    }


def load_v2_weights(
    model: PlumpTransformerModel,
    state_dict: Mapping[str, Tensor],
) -> dict[str, list[str]]:
    """Warm-start a v4 model from v2 while replacing the owner head."""

    return _load_pre_v4_weights(
        model,
        state_dict,
        allowed_fresh_prefixes=(
            *V3_GAME_PARAMETER_PREFIXES,
            *V4_OWNER_PARAMETER_PREFIXES,
        ),
    )


def load_v3_weights(
    model: PlumpTransformerModel,
    state_dict: Mapping[str, Tensor],
) -> dict[str, list[str]]:
    """Warm-start a v4 model from v3 while replacing the owner head."""

    return _load_pre_v4_weights(
        model,
        state_dict,
        allowed_fresh_prefixes=V4_OWNER_PARAMETER_PREFIXES,
    )


def _load_pre_v4_weights(
    model: PlumpTransformerModel,
    state_dict: Mapping[str, Tensor],
    *,
    allowed_fresh_prefixes: tuple[str, ...],
) -> dict[str, list[str]]:
    current = model.state_dict()
    dropped = sorted(
        key
        for key in state_dict
        if key.startswith(PRE_V4_OWNER_PARAMETER_PREFIXES)
    )
    candidates = {
        key: value
        for key, value in state_dict.items()
        if key not in dropped
    }
    compatible = {
        key: value
        for key, value in candidates.items()
        if key in current and current[key].shape == value.shape
    }
    unexpected = sorted(key for key in candidates if key not in current)
    mismatched = sorted(
        key
        for key, value in candidates.items()
        if key in current and current[key].shape != value.shape
    )
    missing = sorted(key for key in current if key not in compatible)
    invalid_missing = [
        key
        for key in missing
        if not key.startswith(allowed_fresh_prefixes)
    ]
    if unexpected or mismatched or invalid_missing:
        raise ValueError(
            "Pre-v4 warm start is not shape-compatible: "
            f"unexpected={unexpected}, mismatched={mismatched}, "
            f"invalid_missing={invalid_missing}"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "fresh": missing,
        "dropped": dropped,
    }
