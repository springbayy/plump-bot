import copy
import random
import tempfile
import unittest
from pathlib import Path

import torch

from plump.cards import Card, Rank, Suit
from plump.env import PlumpEnv
from plump.information_search import (
    InformationSearchConfig,
    InformationSetSearch,
    information_set_key,
)
from plump.modeling import ModelConfig, PlumpSearchModel, encode_observation
from plump.modeling.torch_model import encoded_observations_to_batch
from plump.policies import HeuristicPolicy, ModelPolicy
from plump.state import BidAction, GameConfig
from plump.training import (
    ExpertIterationConfig,
    ExpertIterationTrainer,
    ExpertReplay,
    OpponentMix,
)


def _model_config():
    return ModelConfig(
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        context_hidden_dim=64,
        game_hidden_dim=32,
        schedule_heads=4,
    )


def _manual_env(swapped=False):
    hands = {
        0: [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.CLUBS, Rank.ACE),
        ],
        1: [
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.CLUBS, Rank.KING),
        ],
        2: [
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.QUEEN),
        ],
    }
    if swapped:
        hands[1], hands[2] = hands[2], hands[1]
    env = PlumpEnv(
        GameConfig(
            num_players=3,
            hand_sizes=[3],
            manual_hands=hands,
            bidding_start_players=[0],
        )
    )
    env.reset()
    return env


class ExpertIterationTest(unittest.TestCase):
    def _trainer(self, *, microbatch_size=4):
        model_config = _model_config()
        config = ExpertIterationConfig(
            player_counts=(3,),
            hand_sizes=(3,),
            rounds_per_configuration=1,
            minibatch_size=8,
            microbatch_size=microbatch_size,
            replay_capacity=100,
            precision="fp32",
            device="cpu",
            model_config=model_config,
            opponent_mix=OpponentMix(0.0, 1.0, 0.0, 0.0),
            search_config=InformationSearchConfig(
                min_determinizations=4,
                max_determinizations=4,
                node_budget=4_096,
            ),
        )
        return ExpertIterationTrainer(
            PlumpSearchModel(model_config),
            config,
        )

    def test_v5_model_emits_masked_q_heads(self):
        env = _manual_env()
        encoded = encode_observation(
            env.get_observation(0),
            _model_config(),
        )
        model = PlumpSearchModel(_model_config()).eval()
        with torch.no_grad():
            output = model(
                encoded_observations_to_batch(
                    [encoded],
                    device="cpu",
                )
            )

        self.assertEqual(output.bid_q_values.shape, (1, 11))
        self.assertEqual(output.card_q_values.shape, (1, 52))
        illegal = torch.tensor(encoded.legal_bid_mask).logical_not()
        self.assertTrue(
            torch.all(
                output.masked_bid_q_values[0, illegal]
                < -1e20
            )
        )

    def test_initialize_from_v4_checkpoint_imports_trunk_with_fresh_q_heads(self):
        from plump.modeling.torch_model import PlumpTransformerModel
        from plump.training import PPOTrainer, TrainingConfig

        model_config = _model_config()
        ppo = PPOTrainer(
            PlumpTransformerModel(model_config),
            TrainingConfig(
                player_counts=(3,),
                hand_sizes=(3,),
                device="cpu",
                model_config=model_config,
            ),
        )
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v4.pt"
            ppo.save_checkpoint(path, iteration=7)
            info = trainer.initialize_from_v4_checkpoint(path)

        self.assertEqual(info["source_iteration"], 7)
        v4_state = ppo.model.state_dict()
        v5_state = trainer.model.state_dict()
        self.assertTrue(
            torch.equal(v5_state["bid_head.weight"], v4_state["bid_head.weight"])
        )
        self.assertTrue(
            torch.equal(
                v5_state["transformer.layers.0.linear1.weight"],
                v4_state["transformer.layers.0.linear1.weight"],
            )
        )
        fresh = info["fresh_tensors"]
        self.assertTrue(fresh)
        self.assertTrue(
            all(
                key.startswith(("bid_q_head.", "card_q_head."))
                for key in fresh
            )
        )

    def test_deep_search_is_independent_of_true_hidden_deal(self):
        torch.manual_seed(7)
        model = PlumpSearchModel(_model_config())
        decisions = []
        for env in (_manual_env(False), _manual_env(True)):
            policy = ModelPolicy(
                copy.deepcopy(model),
                device="cpu",
                greedy=False,
            )
            search = InformationSetSearch(
                policy,
                opponent_policy_for_player=lambda _: HeuristicPolicy(),
                config=InformationSearchConfig(
                    min_determinizations=4,
                    max_determinizations=4,
                    node_budget=4_096,
                ),
            )
            decisions.append(
                search.search(
                    env.get_observation(0),
                    legal_actions=env.legal_actions(),
                    rng=random.Random(11),
                )
            )

        self.assertEqual(
            decisions[0].action_values,
            decisions[1].action_values,
        )
        self.assertEqual(
            decisions[0].action_probabilities,
            decisions[1].action_probabilities,
        )

    def test_information_set_key_changes_only_after_public_reveal(self):
        first = _manual_env(False)
        second = _manual_env(True)
        self.assertEqual(
            information_set_key(first.get_observation(0)),
            information_set_key(second.get_observation(0)),
        )
        first.step(BidAction(0, 1))
        second.step(BidAction(0, 1))
        first.step(BidAction(1, 0))
        second.step(BidAction(1, 1))
        self.assertNotEqual(
            information_set_key(first.get_observation(0)),
            information_set_key(second.get_observation(0)),
        )

    def test_cycle_routes_search_and_universal_targets(self):
        trainer = self._trainer()
        cycle = trainer.collect_cycle(cycle=1)

        self.assertEqual(len(cycle.outcomes), 1)
        self.assertTrue(cycle.samples)
        self.assertTrue(any(not row.forced for row in cycle.samples))
        for row in cycle.samples:
            self.assertIsNotNone(row.target_value)
            self.assertIsNotNone(row.final_trick_targets)
            self.assertEqual(len(row.owner_targets), 52)
            if row.forced:
                self.assertIsNone(row.search_policy)
        searched = [row for row in cycle.samples if not row.forced]
        self.assertTrue(
            all(row.search_policy is not None for row in searched)
        )

    def test_update_trains_search_and_auxiliary_losses(self):
        trainer = self._trainer()
        cycle = trainer.collect_cycle(cycle=1)
        for row in cycle.samples:
            if row.search_policy is not None:
                row.accepted = True
        trainer.add_cycle(cycle, cycle_index=1)

        stats = trainer.update(new_state_count=len(cycle.samples))

        self.assertGreater(stats.policy_loss, 0.0)
        self.assertGreaterEqual(stats.q_loss, 0.0)
        self.assertGreater(stats.value_loss, 0.0)
        self.assertGreater(stats.owner_loss, 0.0)

    def test_replay_keeps_exact_opponent_objective(self):
        trainer = self._trainer()
        cycle = trainer.collect_cycle(cycle=1)
        template = cycle.samples[0]
        replay = ExpertReplay(capacity=100)
        rows = []
        for arm in ("self", "heuristic", "mixed", "historical"):
            row = copy.deepcopy(template)
            row.opponent_arm = arm
            rows.append(row)
        replay.add_many(rows)

        selected = replay.balanced_sample(
            random.Random(7),
            1_000,
            {
                "self": 0.3,
                "heuristic": 0.3,
                "mixed": 0.3,
                "historical": 0.1,
            },
        )
        counts = {
            arm: sum(row.opponent_arm == arm for row in selected)
            for arm in ("self", "heuristic", "mixed", "historical")
        }

        self.assertEqual(
            counts,
            {
                "self": 300,
                "heuristic": 300,
                "mixed": 300,
                "historical": 100,
            },
        )

    def test_lockstep_self_play_collection_completes_with_partial_search(self):
        model_config = _model_config()
        config = ExpertIterationConfig(
            player_counts=(3,),
            hand_sizes=(3,),
            rounds_per_configuration=4,
            concurrent_episodes=3,
            play_search_fraction=0.0,
            minibatch_size=8,
            microbatch_size=4,
            replay_capacity=100,
            precision="fp32",
            device="cpu",
            model_config=model_config,
            opponent_mix=OpponentMix(1.0, 0.0, 0.0, 0.0),
            search_config=InformationSearchConfig(
                min_determinizations=4,
                max_determinizations=4,
                node_budget=4_096,
            ),
        )
        trainer = ExpertIterationTrainer(
            PlumpSearchModel(model_config),
            config,
        )

        cycle = trainer.collect_cycle(cycle=1)

        self.assertEqual(len(cycle.outcomes), 4)
        # 3 players x (1 bid + 3 plays) focal decisions per round.
        self.assertEqual(len(cycle.samples), 4 * 4)
        for row in cycle.samples:
            self.assertGreaterEqual(row.action_index, 0)
            self.assertIsNotNone(row.final_trick_targets)
            self.assertIsNotNone(row.target_value)
            if row.phase == "play" and not row.forced:
                # play_search_fraction=0 must skip play searches entirely.
                self.assertIsNone(row.search_policy)
        searched_bids = [
            row
            for row in cycle.samples
            if row.phase == "bid" and not row.forced
        ]
        self.assertTrue(searched_bids)
        self.assertTrue(
            all(row.search_policy is not None for row in searched_bids)
        )

    def test_checkpoint_is_strict_v5_and_restores_replay(self):
        trainer = self._trainer()
        cycle = trainer.collect_cycle(cycle=1)
        trainer.add_cycle(cycle, cycle_index=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v5.pt"
            trainer.save_checkpoint(path, cycle=1)
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["schema_version"], 5)
            self.assertEqual(
                payload["observation_schema_version"],
                4,
            )
            restored = self._trainer()
            info = restored.load_checkpoint(path)
            self.assertEqual(info["cycle"], 1)
            self.assertEqual(len(restored.replay), len(trainer.replay))

            old = Path(tmp) / "v4.pt"
            torch.save({"schema_version": 4}, old)
            with self.assertRaisesRegex(ValueError, "schema-v5"):
                restored.load_checkpoint(old)


if __name__ == "__main__":
    unittest.main()
