"""Run one architecture forward pass from a live Plump observation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump import GameConfig, PlumpEnv
from plump.modeling import ModelConfig, encode_observation
from plump.modeling.torch_model import PlumpTransformerModel, best_torch_device, encoded_observations_to_batch


def main() -> None:
    env = PlumpEnv(GameConfig(num_players=4, hand_sizes=[3]), seed=11)
    env.reset()
    observation = env.get_observation(env.current_player())

    model_config = ModelConfig(max_seq_len=128)
    encoded = encode_observation(observation, model_config)
    device = best_torch_device()
    batch = encoded_observations_to_batch([encoded], device=device)
    model = PlumpTransformerModel(model_config).to(device)
    output = model(batch)

    print(f"device: {device}")
    print(f"state shape: {tuple(output.state.shape)}")
    print(f"bid logits shape: {tuple(output.masked_bid_logits.shape)}")
    print(f"card logits shape: {tuple(output.masked_card_logits.shape)}")
    print(f"value shape: {tuple(output.value.shape)}")
    print(f"trick count logits shape: {tuple(output.trick_count_logits.shape)}")
    print(f"masked trick count logits shape: {tuple(output.masked_trick_count_logits.shape)}")
    print(f"owner logits shape: {tuple(output.owner_logits.shape)}")
    print(f"owner capacities: {batch.owner_capacities[0].cpu().tolist()}")
    print(
        "projected owner counts: "
        f"{output.owner_probs[0].sum(dim=0).detach().cpu().tolist()}"
    )
    print(
        "maximum projected row error: "
        f"{(output.owner_probs[0].sum(dim=-1)[batch.owner_valid_mask[0].any(dim=-1)] - 1.0).abs().max().item():.3g}"
    )
    print(f"score probabilities shape: {tuple(output.score_probs.shape)}")


if __name__ == "__main__":
    main()
