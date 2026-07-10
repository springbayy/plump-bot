import math
import unittest
import random

import torch

from plump.modeling import ModelConfig
from plump.modeling.torch_model import (
    PlumpTransformerModel,
    encoded_observations_to_batch,
)
from plump.search import SearchDecision
from plump.state import BidAction
from plump.training import (
    CounterfactualSearchRouter,
    PPOTrainer,
    SearchReplaySample,
    SearchTrustRegionUpdater,
    StratifiedReplayWindow,
    TrainingConfig,
)


class SearchRoutingTest(unittest.TestCase):
    def _ppo_trainer(self):
        model_config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            game_hidden_dim=32,
            schedule_heads=4,
        )
        return PPOTrainer(
            PlumpTransformerModel(model_config),
            TrainingConfig(
                player_counts=(3,),
                hand_sizes=(3,),
                rounds_per_configuration=1,
                num_envs=1,
                ppo_epochs=1,
                minibatch_size=64,
                self_play_fraction=1.0,
                heuristic_fraction=0.0,
                mixed_fraction=0.0,
                historical_fraction=0.0,
                device="cpu",
                model_config=model_config,
            ),
        )

    def test_replay_expires_samples_older_than_250_updates(self):
        trainer = self._ppo_trainer()
        sample = next(
            row
            for row in trainer.collect_rollouts().samples
            if row.phase == "bid"
        )
        target = [0.0] * trainer.config.model_config.bid_count
        legal = [
            index
            for index, valid in enumerate(sample.encoded.legal_bid_mask)
            if valid
        ]
        for index in legal:
            target[index] = 1.0 / len(legal)
        replay = StratifiedReplayWindow(capacity=10, max_age=250)
        replay.add(
            SearchReplaySample(
                sample.encoded,
                "bid",
                target,
                sample.spec,
                sample.bidding_position,
                iteration=1,
            )
        )
        replay.add(
            SearchReplaySample(
                sample.encoded,
                "bid",
                target,
                sample.spec,
                sample.bidding_position,
                iteration=300,
            )
        )

        rows = replay.balanced_samples(
            random.Random(1),
            current_iteration=300,
        )

        self.assertEqual([row.iteration for row in rows], [300])

    def test_phase_gate_and_regret_ramp_require_three_stable_diagnostics(self):
        trainer = self._ppo_trainer()
        sample = next(
            row
            for row in trainer.collect_rollouts().samples
            if row.phase == "bid"
        )
        legal = [
            BidAction(sample.acting_player, index)
            for index, valid in enumerate(sample.encoded.legal_bid_mask)
            if valid
        ][:2]
        keys = [f"bid:{action.bid}" for action in legal]
        decision = SearchDecision(
            action=legal[0],
            action_values={keys[0]: 1.0, keys[1]: 0.0},
            action_probabilities={keys[0]: 0.9, keys[1]: 0.1},
            action_regrets={keys[0]: 0.5, keys[1]: -0.5},
            action_stderr={keys[0]: 0.0, keys[1]: 0.0},
            prior_probabilities={keys[0]: 0.5, keys[1]: 0.5},
            split_half_argmax_agreement=True,
            target_js_divergence=0.0,
            accepted=True,
            samples_per_action=8,
            forward_passes=0,
        )
        router = CounterfactualSearchRouter(
            trainer.model,
            device="cpu",
            minimum_iteration=0,
            explained_variance_threshold=0.3,
            seed=4,
        )
        for _ in range(3):
            router.update_diagnostics(
                bid_explained_variance=0.4,
                play_explained_variance=0.4,
            )
        for iteration in range(1, 3):
            router._phase_report(
                "bid",
                True,
                [(sample, decision)],
                iteration,
            )
            self.assertEqual(router.regret_matching_fraction("bid"), 0.0)
        report = router._phase_report(
            "bid",
            True,
            [(sample, decision)],
            3,
        )

        self.assertTrue(report.gate_passed)
        self.assertFalse(sample.ppo_policy_enabled)
        self.assertGreater(router.regret_matching_fraction("bid"), 0.0)

    def test_search_update_respects_overall_and_stratum_kl_caps(self):
        trainer = self._ppo_trainer()
        sample = next(
            row
            for row in trainer.collect_rollouts().samples
            if row.phase == "bid"
        )
        legal = [
            index
            for index, valid in enumerate(sample.encoded.legal_bid_mask)
            if valid
        ]
        target = [0.0] * trainer.config.model_config.bid_count
        regrets = [0.0] * trainer.config.model_config.bid_count
        target[legal[0]] = 0.9
        target[legal[1]] = 0.1
        regrets[legal[0]] = 1.0
        replay = StratifiedReplayWindow(capacity=10)
        replay.add(
            SearchReplaySample(
                sample.encoded,
                "bid",
                target,
                sample.spec,
                sample.bidding_position,
                iteration=1,
                regrets=regrets,
            )
        )
        updater = SearchTrustRegionUpdater(
            trainer.model,
            device="cpu",
            learning_rate=1e-2,
            minibatch_size=8,
        )

        stats = updater.update(
            replay,
            phase="bid",
            iteration=1,
            regret_matching_fraction=0.5,
        )

        self.assertTrue(math.isfinite(stats.loss))
        if stats.applied:
            self.assertLessEqual(stats.kl, stats.kl_cap)
            self.assertLessEqual(
                stats.maximum_stratum_kl,
                stats.kl_cap,
            )

    def test_search_entropy_floor_uses_only_legal_actions(self):
        trainer = self._ppo_trainer()
        sample = next(
            row
            for row in trainer.collect_rollouts().samples
            if row.phase == "bid"
        )
        legal = [
            index
            for index, valid in enumerate(sample.encoded.legal_bid_mask)
            if valid
        ]
        illegal = next(
            index
            for index, valid in enumerate(sample.encoded.legal_bid_mask)
            if not valid
        )
        with torch.no_grad():
            trainer.model.bid_head.weight.zero_()
            trainer.model.bid_head.bias.zero_()
            trainer.model.bid_head.bias[legal[0]] = 10.0
            trainer.model.bid_head.bias[illegal] = 1_000.0

        target = [0.0] * trainer.config.model_config.bid_count
        for index in legal:
            target[index] = 1.0 / len(legal)
        replay = StratifiedReplayWindow(capacity=10)
        replay.add(
            SearchReplaySample(
                sample.encoded,
                "bid",
                target,
                sample.spec,
                sample.bidding_position,
                iteration=1,
                regrets=[0.0] * len(target),
            )
        )
        batch = encoded_observations_to_batch(
            [sample.encoded],
            device="cpu",
        )
        with torch.no_grad():
            masked_logits = trainer.model(batch).masked_bid_logits[0]
            legal_entropy = torch.distributions.Categorical(
                logits=masked_logits[legal].float()
            ).entropy()
        updater = SearchTrustRegionUpdater(
            trainer.model,
            device="cpu",
            learning_rate=0.0,
        )

        active = updater.update(
            replay,
            phase="bid",
            iteration=1,
            regret_matching_fraction=0.1,
        )
        inactive = updater.update(
            replay,
            phase="bid",
            iteration=1,
            regret_matching_fraction=0.0,
        )

        self.assertAlmostEqual(
            active.policy_entropy,
            float(legal_entropy),
            places=6,
        )
        self.assertGreater(active.entropy_floor_loss, 0.0)
        self.assertEqual(inactive.entropy_floor_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
