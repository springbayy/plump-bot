"""Run a complete random Plump game."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plump import GameConfig, PlumpEnv


def main() -> None:
    rng = random.Random(7)
    env = PlumpEnv(GameConfig(num_players=4), seed=7)
    env.reset()

    print(f"Starting player: {env.current_player()}")
    print(f"Legal actions: {env.legal_actions()}")

    while not env.is_done():
        action = rng.choice(env.legal_actions())
        result = env.step(action)
        if result.info.get("round_ended"):
            print(f"Round {result.info['round_index']} rewards: {result.rewards}")

    print(f"Final scores: {env.state.cumulative_scores}")


if __name__ == "__main__":
    main()
