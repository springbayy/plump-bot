import csv
import copy
import dataclasses
import math
import tempfile
import unittest
from pathlib import Path

import torch

from plump.modeling import ModelConfig, SCHEMA_VERSION
from plump.modeling.torch_model import (
    PlumpTransformerModel,
    encoded_observations_to_batch,
)
from plump.policies import RandomPolicy
from plump.training import (
    PPOTrainer,
    PositionBaseline,
    TrainingConfig,
    TrainingRunLogger,
    compute_relative_rewards,
    training_config_snapshot,
)
from plump.training.ppo import (
    _explained_variance,
    _trick_implied_relative_values,
)


class TrainingTest(unittest.TestCase):
    def _trainer(self, *, player_counts=(3,), hand_sizes=(3,)) -> PPOTrainer:
        model_config = ModelConfig(
            max_seq_len=48,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            dropout=0.0,
        )
        config = TrainingConfig(
            player_counts=player_counts,
            hand_sizes=hand_sizes,
            rounds_per_configuration=1,
            num_envs=2,
            ppo_epochs=1,
            minibatch_size=64,
            learning_rate=1e-3,
            self_play_fraction=1.0,
            historical_fraction=0.0,
            heuristic_fraction=0.0,
            mixed_fraction=0.0,
            seed=3,
            device="cpu",
            model_config=model_config,
        )
        return PPOTrainer(PlumpTransformerModel(model_config), config)

    def test_compute_relative_rewards(self):
        rewards = compute_relative_rewards({0: 10, 1: 4, 2: 1})
        self.assertEqual(rewards, {0: 7.5, 1: -1.5, 2: -6.0})

    def test_balanced_schedule_and_macro_round_weights(self):
        trainer = self._trainer(player_counts=(3, 4), hand_sizes=(3, 4))
        schedule = trainer.balanced_round_specs()
        self.assertEqual(len(schedule), 4)
        self.assertEqual(set(schedule), set(trainer.config.specs))

        buffer = trainer.collect_rollouts()
        counts = {spec: 0 for spec in trainer.config.specs}
        for outcome in buffer.round_outcomes:
            counts[outcome.spec] += 1
        self.assertEqual(set(counts.values()), {1})
        # Self-play rounds train every seat, so each round's weight is split
        # evenly across its seat trajectories.
        for sample in buffer.samples:
            self.assertEqual(
                sample.round_weight,
                1.0 / (4.0 * sample.spec.num_players),
            )

    def test_historical_arm_falls_back_to_self_play_without_history(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            self_play_fraction=0.3,
            historical_fraction=0.2,
            heuristic_fraction=0.2,
            mixed_fraction=0.3,
        )

        fractions = trainer._effective_arm_fractions()

        self.assertEqual(fractions["historical"], 0.0)
        self.assertEqual(fractions["self"], 0.5)

    def test_opponent_mix_is_gradient_weighted_30_30_30_10(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            rounds_per_configuration=16,
            self_play_fraction=0.3,
            heuristic_fraction=0.3,
            mixed_fraction=0.3,
            historical_fraction=0.1,
        )
        trainer.historical_policies.append(RandomPolicy(19))

        buffer = trainer.collect_rollouts()
        arm_counts = {
            arm: sum(outcome.opponent_arm == arm for outcome in buffer.round_outcomes)
            for arm in ("self", "heuristic", "mixed", "historical")
        }
        weight_totals = {
            arm: sum(
                sample.round_weight
                for sample in buffer.samples
                if sample.opponent_arm == arm
            )
            for arm in ("self", "heuristic", "mixed", "historical")
        }
        total_weight = sum(weight_totals.values())

        self.assertEqual(arm_counts, {
            "self": 5,
            "heuristic": 5,
            "mixed": 5,
            "historical": 1,
        })
        self.assertAlmostEqual(weight_totals["self"] / total_weight, 0.3)
        self.assertAlmostEqual(weight_totals["heuristic"] / total_weight, 0.3)
        self.assertAlmostEqual(weight_totals["mixed"] / total_weight, 0.3)
        self.assertAlmostEqual(weight_totals["historical"] / total_weight, 0.1)

    def test_current_policy_seats_are_trained_and_frozen_arms_store_focal_only(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            rounds_per_configuration=16,
            self_play_fraction=0.3,
            heuristic_fraction=0.3,
            mixed_fraction=0.3,
            historical_fraction=0.1,
        )
        trainer.historical_policies.append(RandomPolicy(23))

        buffer = trainer.collect_rollouts()
        arms_by_episode = {
            outcome.episode_id: outcome.opponent_arm
            for outcome in buffer.round_outcomes
        }
        acting_players_by_episode: dict[int, set[int]] = {}
        for sample in buffer.samples:
            acting_players_by_episode.setdefault(sample.episode_id, set()).add(
                sample.acting_player
            )

        self.assertEqual(len(acting_players_by_episode), 16)
        num_players = trainer.config.specs[0].num_players
        decisions_per_seat = 1 + trainer.config.specs[0].hand_size
        for episode_id, players in acting_players_by_episode.items():
            arm = arms_by_episode[episode_id]
            if arm == "self":
                self.assertEqual(len(players), num_players)
            elif arm in ("heuristic", "historical"):
                self.assertEqual(len(players), 1)
            episode_samples = [
                sample
                for sample in buffer.samples
                if sample.episode_id == episode_id
            ]
            self.assertEqual(
                len(episode_samples),
                len(players) * decisions_per_seat,
            )

    def test_self_play_seat_returns_are_relative_and_sum_to_zero(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()

        last_return_by_player: dict[int, float] = {}
        for sample in buffer.samples:
            last_return_by_player[sample.acting_player] = float(
                sample.return_target
            )
        self.assertEqual(len(last_return_by_player), 3)
        self.assertAlmostEqual(
            sum(last_return_by_player.values()),
            0.0,
            places=6,
        )

    def test_mixed_arm_assigns_all_three_opponent_categories(self):
        trainer = self._trainer()
        historical = RandomPolicy(29)
        trainer.historical_policies.append(historical)
        categories = set()

        for episode_id in range(100):
            episode = trainer._new_active_episode(
                trainer.config.specs[0],
                "mixed",
                episode_id,
                iteration=1,
            )
            for policy in episode.opponent_policies.values():
                if policy is None:
                    categories.add("self")
                elif policy is trainer.heuristic_policy:
                    categories.add("heuristic")
                elif policy is historical:
                    categories.add("historical")

        self.assertEqual(categories, {"self", "heuristic", "historical"})

    def test_rollouts_assign_residual_value_and_auxiliary_targets(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()

        self.assertEqual(len(buffer.round_outcomes), 1)
        self.assertTrue(buffer.samples)
        self.assertTrue(trainer.position_baseline.values)
        for sample in buffer.samples:
            self.assertIsNotNone(sample.return_target)
            self.assertIsNotNone(sample.value_target)
            self.assertEqual(sample.position_intercept, 0.0)
            self.assertEqual(len(sample.final_trick_targets), 5)
            self.assertEqual(len(sample.final_bid_targets), 5)
            self.assertEqual(len(sample.owner_targets), 52)
            active_owner_targets = [target for target in sample.owner_targets if target >= 0]
            self.assertTrue(active_owner_targets)
            self.assertTrue(all(target < trainer.config.model_config.owner_class_count for target in active_owner_targets))
            self.assertFalse(sample.encoded.game_context_enabled)

    def test_heuristic_reward_uses_only_focal_round_rewards(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            self_play_fraction=0.0,
            historical_fraction=0.0,
            heuristic_fraction=1.0,
            mixed_fraction=0.0,
        )
        buffer = trainer.collect_rollouts()
        stats = trainer.summarize_rollout(buffer)

        self.assertEqual(stats.heuristic_rounds, 1)
        self.assertEqual(buffer.round_outcomes[0].opponent_arm, "heuristic")
        self.assertEqual(
            stats.heuristic_relative_reward,
            buffer.round_outcomes[0].focal_reward,
        )

    def test_trick_baseline_folds_implied_value_into_intercepts(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(trainer.config, trick_baseline=True)

        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)

        self.assertTrue(math.isfinite(stats.total_loss))
        play_intercepts = [
            sample.position_intercept
            for sample in buffer.samples
            if sample.phase == "play"
        ]
        self.assertTrue(any(abs(value) > 0.0 for value in play_intercepts))
        for sample in buffer.samples:
            self.assertAlmostEqual(
                sample.old_value,
                sample.position_intercept + sample.old_residual_value,
                places=4,
            )
            self.assertAlmostEqual(
                sample.value_target,
                sample.return_target - sample.position_intercept,
                places=4,
            )

    def test_position_baseline_is_lagged(self):
        baseline = PositionBaseline(decay=0.5)
        key = (3, 3, 0)
        self.assertEqual(baseline.get(key), 0.0)
        baseline.update_many([(key, 10.0), (key, 6.0)])
        self.assertEqual(baseline.get(key), 8.0)
        baseline.update_many([(key, 4.0)])
        self.assertEqual(baseline.get(key), 6.0)

    def test_update_and_diagnostics_are_finite(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)
        prediction = trainer.compute_prediction_stats(buffer, max_samples=8, minibatch_size=4)

        for value in (
            stats.total_loss,
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.auxiliary_loss,
            stats.trick_loss,
            stats.owner_loss,
            stats.owner_ce_loss,
            stats.owner_capacity_loss,
            prediction.value_mse,
            prediction.owner_brier,
            prediction.owner_opponent_accuracy,
            prediction.owner_opponent_true_prob,
            prediction.owner_capacity_mae,
            prediction.owner_raw_capacity_mae,
            prediction.hit_prob_brier,
            prediction.trick_implied_value_explained_variance,
            prediction.bid_trick_implied_value_explained_variance,
            prediction.play_trick_implied_value_explained_variance,
        ):
            self.assertTrue(math.isfinite(value))
        self.assertEqual(stats.configurations, 1)
        self.assertGreaterEqual(prediction.owner_accuracy, 0.0)
        self.assertLessEqual(prediction.owner_accuracy, 1.0)
        self.assertGreaterEqual(prediction.owner_opponent_accuracy, 0.0)
        self.assertLessEqual(prediction.owner_opponent_accuracy, 1.0)
        self.assertLess(prediction.owner_capacity_max_error, 1e-4)
        self.assertGreater(prediction.owner_raw_capacity_mae, 0.0)

    def test_trick_implied_value_uses_expected_relative_score(self):
        probabilities = torch.zeros((1, 3, 4))
        probabilities[0, 0, 1] = 0.8
        probabilities[0, 0, 0] = 0.2
        probabilities[0, 1, 0] = 0.5
        probabilities[0, 1, 1] = 0.5
        probabilities[0, 2, 2] = 0.25
        probabilities[0, 2, 0] = 0.75
        bids = torch.tensor([[1, 0, 2]])
        active = torch.tensor([[True, True, True]])

        implied = _trick_implied_relative_values(
            probabilities,
            bids,
            active,
        )

        self.assertAlmostEqual(float(implied[0]), 6.05, places=5)

    def test_explained_variance_ignores_constant_prediction_offset(self):
        self.assertAlmostEqual(
            _explained_variance(
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ),
            1.0,
        )

    def test_sinkhorn_iteration_count_must_be_positive(self):
        model_config = dataclasses.replace(
            self._trainer().config.model_config,
            owner_sinkhorn_iterations=0,
        )
        config = dataclasses.replace(
            self._trainer().config,
            model_config=model_config,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PPOTrainer(PlumpTransformerModel(model_config), config)

    def test_focal_decision_gae_bootstraps_across_opponent_actions(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        samples = [copy.deepcopy(buffer.samples[0]) for _ in range(3)]
        for sample, value in zip(samples, (1.0, 2.0, 3.0)):
            sample.old_value = value
            sample.position_intercept = 0.0

        trainer._assign_gae(
            samples,
            terminal_reward=10.0,
            gae_lambda=0.95,
        )

        self.assertAlmostEqual(samples[2].return_target, 10.0)
        self.assertAlmostEqual(samples[1].return_target, 9.65)
        self.assertAlmostEqual(samples[0].return_target, 9.2675)

    def test_routed_states_disable_only_ppo_policy_and_entropy(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            value_coef=1.0,
            trick_coef=0.0,
            owner_coef=0.0,
        )
        buffer = trainer.collect_rollouts()
        before = {
            key: value.detach().clone()
            for key, value in trainer.model.value_head.state_dict().items()
        }
        for sample in buffer.samples:
            sample.ppo_policy_enabled = False

        stats = trainer.update(buffer)

        self.assertEqual(stats.policy_loss, 0.0)
        self.assertEqual(stats.entropy, 0.0)
        self.assertTrue(
            any(
                not torch.equal(before[key], value)
                for key, value in trainer.model.value_head.state_dict().items()
            )
        )

    def test_ppo_entropy_uses_only_post_mask_legal_actions(self):
        trainer = self._trainer()
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
            trainer.model.bid_head.bias[illegal] = 1_000.0
            output = trainer.model(
                encoded_observations_to_batch(
                    [sample.encoded],
                    device="cpu",
                )
            )
            _, entropy = trainer._logprobs_and_entropy(
                output,
                [sample],
            )

        self.assertAlmostEqual(
            float(entropy[0]),
            math.log(len(legal)),
            places=5,
        )

    def test_complete_game_rollout_uses_masked_context_and_terminal_targets(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            training_mode="game",
            game_schedule=(3, 2, 3),
            games_per_player_seat=1,
        )
        buffer = trainer.collect_rollouts()

        self.assertEqual(len(buffer.round_outcomes), 9)
        self.assertTrue(buffer.samples)
        self.assertTrue(
            all(sample.encoded.game_context_enabled for sample in buffer.samples)
        )
        self.assertTrue(
            all(sample.return_target is not None for sample in buffer.samples)
        )
        episode_ids = {sample.episode_id for sample in buffer.samples}
        self.assertEqual(len(episode_ids), 3)

    def test_microbatch_accumulation_matches_logical_minibatch(self):
        full = self._trainer()
        buffer = full.collect_rollouts()
        micro_config = dataclasses.replace(full.config, microbatch_size=3)
        micro_model = copy.deepcopy(full.model)
        full.optimizer = torch.optim.SGD(full.model.parameters(), lr=1e-3)
        micro = PPOTrainer(
            micro_model,
            micro_config,
            optimizer=torch.optim.SGD(micro_model.parameters(), lr=1e-3),
        )
        micro.rng.setstate(full.rng.getstate())

        full_stats = full.update(buffer)
        micro_stats = micro.update(buffer)

        for full_parameter, micro_parameter in zip(
            full.model.parameters(),
            micro.model.parameters(),
        ):
            self.assertTrue(
                torch.allclose(full_parameter, micro_parameter, atol=2e-6, rtol=2e-5)
            )
        self.assertAlmostEqual(full_stats.total_loss, micro_stats.total_loss, places=5)
        self.assertAlmostEqual(full_stats.owner_loss, micro_stats.owner_loss, places=5)

    def test_checkpoint_is_v4_and_older_resume_is_rejected(self):
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v4.pt"
            trainer.save_checkpoint(path, iteration=7)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertIn("rules_fingerprint", payload)
            self.assertIn("position_baseline", payload)
            self.assertEqual(trainer.load_checkpoint(path)["iteration"], 7)

            legacy = Path(tmp) / "v1.pt"
            torch.save({"model_state_dict": {}}, legacy)
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                trainer.load_checkpoint(legacy)

    def test_logger_uses_only_v4_metrics(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        rollout = trainer.summarize_rollout(buffer)
        update = trainer.update(buffer)
        prediction = trainer.compute_prediction_stats(buffer, max_samples=4, minibatch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingRunLogger(tmp)
            logger.write_config({"training_config": training_config_snapshot(trainer.config)})
            logger.log_iteration(
                iteration=1,
                elapsed_sec=1.0,
                timings={"iteration_sec": 1.0},
                update=update,
                rollout=rollout,
                prediction=prediction,
                evaluation=None,
                checkpoint_path=Path(tmp) / "checkpoint.pt",
            )
            fields = next(csv.reader((Path(tmp) / "metrics.csv").open()))
            self.assertIn("loss_owner", fields)
            self.assertIn("loss_owner_ce", fields)
            self.assertIn("loss_owner_capacity", fields)
            self.assertIn("pred_owner_accuracy", fields)
            self.assertIn("pred_owner_opponent_accuracy", fields)
            self.assertIn("pred_owner_capacity_mae", fields)
            self.assertIn(
                "pred_trick_implied_value_explained_variance",
                fields,
            )
            self.assertIn(
                "search_bid_sampler_infeasible_rejection_rate",
                fields,
            )
            self.assertIn("search_bid_entropy_floor_loss", fields)
            self.assertNotIn("loss_point", fields)
            self.assertNotIn("loss_hand", fields)


if __name__ == "__main__":
    unittest.main()
