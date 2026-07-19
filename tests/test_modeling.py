import copy
import importlib.util
import unittest

from plump.cards import Card, Rank, Suit
from plump.env import PlumpEnv
from plump.modeling import EVENT_TOKEN_WIDTH, ModelConfig, card_id, encode_observation
from plump.modeling.torch_model import (
    V4_OWNER_PARAMETER_PREFIXES,
    PlumpTransformerModel,
    encoded_observations_to_batch,
    load_v3_weights,
    masked_capacity_sinkhorn,
)
from plump.rounds import descending_ascending_schedule
from plump.state import BidAction, GameConfig, GameEvent, PlayCardAction, TrumpPolicy


def _hands():
    return {
        0: [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.HEARTS, Rank.FOUR),
        ],
        1: [
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.HEARTS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FIVE),
        ],
        2: [
            Card(Suit.DIAMONDS, Rank.ACE),
            Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.HEARTS, Rank.SIX),
        ],
        3: [
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE),
            Card(Suit.CLUBS, Rank.SEVEN),
        ],
    }


def _env_with_history(rotation=0):
    hands = {(player + rotation) % 4: cards for player, cards in _hands().items()}
    env = PlumpEnv(
        GameConfig(
            num_players=4,
            hand_sizes=[3],
            manual_hands=hands,
            trump_policy=TrumpPolicy.NONE,
            forbid_total_bid_equals_hand_size=False,
            bidding_start_players=[rotation],
        )
    )
    env.reset()
    for player, bid in ((0, 1), (1, 0), (2, 1), (3, 0)):
        env.step(BidAction((player + rotation) % 4, bid))
    env.step(
        PlayCardAction(
            rotation,
            Card(Suit.SPADES, Rank.ACE),
        )
    )
    return env


class ModelingEncodingTest(unittest.TestCase):
    def test_five_player_ten_card_decision_history_tops_out_at_64_events(self):
        env = PlumpEnv(
            GameConfig(
                num_players=5,
                hand_sizes=[10],
                forbid_total_bid_equals_hand_size=False,
            ),
            seed=17,
        )
        env.reset()
        decision_lengths = []
        while not env.is_done():
            observation = env.get_observation(env.current_player())
            decision_lengths.append(
                sum(
                    event.round_index == observation.round_index
                    for event in observation.event_log
                )
            )
            env.step(env.legal_actions()[0])

        self.assertEqual(max(decision_lengths), 64)

    def test_event_buckets_never_truncate_and_fall_back_above_64(self):
        config = ModelConfig(max_seq_len=100)
        encoded = encode_observation(
            _env_with_history().get_observation(1),
            config,
        )
        for valid_length, expected_length in (
            (6, 8),
            (9, 16),
            (17, 32),
            (33, 64),
            (65, 100),
        ):
            candidate = copy.deepcopy(encoded)
            candidate.event_valid_mask = [
                index < valid_length for index in range(config.max_seq_len)
            ]
            batch = encoded_observations_to_batch(
                [candidate],
                device="cpu",
                event_length_buckets=(8, 16, 32, 64),
            )
            self.assertEqual(batch.max_valid_event_length, valid_length)
            self.assertEqual(batch.event_length, expected_length)
            self.assertEqual(int(batch.event_valid_mask.sum()), valid_length)

    def test_torch_and_numpy_batch_packing_match(self):
        import torch

        config = ModelConfig(max_seq_len=32)
        observations = [
            encode_observation(
                _env_with_history().get_observation(1),
                config,
                include_game_context=include_game_context,
            )
            for include_game_context in (False, True)
        ]
        torch_batch = encoded_observations_to_batch(
            observations,
            device="cpu",
            event_length_buckets=(8, 16, 32, 64),
            packing="torch",
        )
        numpy_batch = encoded_observations_to_batch(
            observations,
            device="cpu",
            event_length_buckets=(8, 16, 32, 64),
            packing="numpy",
        )
        for field in torch_batch.__dataclass_fields__:
            expected = getattr(torch_batch, field)
            actual = getattr(numpy_batch, field)
            if isinstance(expected, torch.Tensor):
                self.assertTrue(torch.equal(expected, actual), field)
            else:
                self.assertEqual(expected, actual, field)

    def test_bucketed_and_lean_forwards_match_full_outputs_and_gradients(self):
        import torch

        torch.manual_seed(23)
        config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            game_hidden_dim=32,
            schedule_heads=4,
            dropout=0.0,
            oracle_critic=True,
        )
        observation = _env_with_history().get_observation(1)
        encoded = [
            encode_observation(
                observation,
                config,
                include_game_context=include_game_context,
            )
            for include_game_context in (False, True)
        ]
        full_batch = encoded_observations_to_batch(encoded, device="cpu")
        bucketed_batch = encoded_observations_to_batch(
            encoded,
            device="cpu",
            event_length_buckets=(8, 16, 32, 64),
        )
        owner_targets = torch.full((2, 52), -100, dtype=torch.long)
        for batch_index, row_mask in enumerate(encoded):
            for card_index, valid_classes in enumerate(row_mask.owner_valid_mask):
                if any(valid_classes):
                    owner_targets[batch_index, card_index] = valid_classes.index(True)

        model = PlumpTransformerModel(config).eval()
        full = model(full_batch, privileged_owner_targets=owner_targets)
        bucketed = model(bucketed_batch, privileged_owner_targets=owner_targets)
        for field in (
            "bid_logits",
            "card_logits",
            "masked_bid_logits",
            "masked_card_logits",
            "value",
            "round_value",
            "game_value",
            "oracle_value",
            "trick_count_logits",
            "masked_trick_count_logits",
        ):
            torch.testing.assert_close(
                getattr(bucketed, field),
                getattr(full, field),
                atol=2e-5,
                rtol=2e-5,
            )

        policy = model.forward_policy(bucketed_batch)
        rollout = model.forward_rollout(
            bucketed_batch,
            privileged_owner_targets=owner_targets,
        )
        for lean, expected in (
            (policy.masked_bid_logits, bucketed.masked_bid_logits),
            (policy.masked_card_logits, bucketed.masked_card_logits),
            (rollout.masked_bid_logits, bucketed.masked_bid_logits),
            (rollout.masked_card_logits, bucketed.masked_card_logits),
            (rollout.value, bucketed.value),
            (rollout.oracle_value, bucketed.oracle_value),
            (
                rollout.masked_trick_count_logits,
                bucketed.masked_trick_count_logits,
            ),
        ):
            torch.testing.assert_close(lean, expected, atol=2e-5, rtol=2e-5)

        # A mixed context batch must retain the exact per-row full-forward
        # behavior, including selecting the round head for the local row and
        # the game head for the context-enabled row.
        for index in range(2):
            individual = model(
                encoded_observations_to_batch([encoded[index]], device="cpu"),
                privileged_owner_targets=owner_targets[index : index + 1],
            )
            torch.testing.assert_close(
                bucketed.value[index],
                individual.value[0],
                atol=2e-5,
                rtol=2e-5,
            )

        def objective(output):
            return (
                output.bid_logits.sum()
                + output.card_logits.sum()
                + output.value.sum()
                + output.oracle_value.sum()
                + output.trick_count_logits.sum()
            )

        model.zero_grad(set_to_none=True)
        objective(
            model(full_batch, privileged_owner_targets=owner_targets)
        ).backward()
        full_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        model.zero_grad(set_to_none=True)
        objective(
            model(bucketed_batch, privileged_owner_targets=owner_targets)
        ).backward()
        bucketed_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        self.assertEqual(set(full_gradients), set(bucketed_gradients))
        for name in full_gradients:
            torch.testing.assert_close(
                bucketed_gradients[name],
                full_gradients[name],
                atol=5e-5,
                rtol=5e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_schema_v4_shapes_masks_capacities_and_reserved_context(self):
        env = _env_with_history()
        config = ModelConfig(max_seq_len=32)
        encoded = encode_observation(env.get_observation(env.current_player()), config)

        self.assertEqual(len(encoded.event_tokens), config.max_seq_len)
        self.assertEqual(len(encoded.event_tokens[0]), EVENT_TOKEN_WIDTH)
        self.assertEqual(len(encoded.context_features), config.context_dim)
        self.assertEqual(len(encoded.player_features), config.max_players)
        self.assertFalse(encoded.game_context_enabled)
        self.assertEqual(encoded.current_player_relative, 0)
        self.assertEqual(encoded.bidding_position, 1)
        self.assertTrue(encoded.active_player_mask[:4])
        self.assertFalse(encoded.active_player_mask[4])
        self.assertTrue(encoded.legal_card_mask[card_id(Card(Suit.SPADES, Rank.KING))])
        self.assertFalse(encoded.legal_card_mask[card_id(Card(Suit.HEARTS, Rank.THREE))])
        self.assertEqual(encoded.bid_values[:4], [0, 1, 0, 1])
        self.assertEqual(len(encoded.owner_valid_mask), 52)
        self.assertEqual(len(encoded.owner_valid_mask[0]), config.owner_class_count)
        self.assertEqual(encoded.owner_capacities, [3, 3, 2, 0, 40])
        self.assertEqual(
            sum(encoded.owner_capacities),
            sum(any(row) for row in encoded.owner_valid_mask),
        )
        self.assertFalse(any(encoded.owner_valid_mask[card_id(Card(Suit.SPADES, Rank.KING))]))
        self.assertTrue(
            encoded.owner_valid_mask[card_id(Card(Suit.DIAMONDS, Rank.ACE))][0]
        )
        self.assertTrue(
            encoded.owner_valid_mask[card_id(Card(Suit.DIAMONDS, Rank.ACE))][
                config.undealt_owner_class
            ]
        )

    def test_trick_count_mask_uses_wins_and_unresolved_tricks(self):
        env = _env_with_history()
        env.step(PlayCardAction(1, Card(Suit.SPADES, Rank.KING)))
        env.step(PlayCardAction(2, Card(Suit.CLUBS, Rank.THREE)))
        env.step(PlayCardAction(3, Card(Suit.SPADES, Rank.TWO)))
        encoded = encode_observation(
            env.get_observation(env.current_player()),
            ModelConfig(max_seq_len=32),
        )

        self.assertEqual(encoded.final_trick_count_mask[0][:5], [False, True, True, True, False])
        self.assertEqual(encoded.final_trick_count_mask[1][:5], [True, True, True, False, False])

    def test_absolute_seat_rotation_is_invariant(self):
        config = ModelConfig(max_seq_len=32)
        original = encode_observation(
            _env_with_history(0).get_observation(1),
            config,
        )
        rotated = encode_observation(
            _env_with_history(1).get_observation(2),
            config,
        )

        self.assertEqual(original.event_tokens, rotated.event_tokens)
        self.assertEqual(original.context_features, rotated.context_features)
        self.assertEqual(original.player_features, rotated.player_features)
        self.assertEqual(original.legal_card_mask, rotated.legal_card_mask)

    def test_prior_rounds_are_ignored_but_reserved_context_can_change(self):
        config = ModelConfig(max_seq_len=32)
        observation = _env_with_history().get_observation(1)
        shifted = copy.deepcopy(observation)
        shifted.round_index = 5
        shifted.total_rounds = 12
        shifted.rounds_remaining = 6
        shifted.hand_size_schedule = [3] * 12
        shifted.scores = {0: 40, 1: 20, 2: 15, 3: 30}
        shifted.event_log = [
            GameEvent(type=event.type, round_index=5, player=event.player, card=event.card,
                      bid=event.bid, trick_index=event.trick_index,
                      position_in_trick=event.position_in_trick)
            for event in observation.event_log
        ]
        shifted.event_log.insert(0, GameEvent(type=observation.event_log[0].type, round_index=2))

        local_a = encode_observation(observation, config)
        local_b = encode_observation(shifted, config)
        game_a = encode_observation(observation, config, include_game_context=True)
        game_b = encode_observation(shifted, config, include_game_context=True)

        self.assertEqual(local_a.event_tokens, local_b.event_tokens)
        self.assertEqual(local_a.context_features, local_b.context_features)
        self.assertNotEqual(game_a.context_features, game_b.context_features)
        self.assertTrue(game_b.game_context_enabled)
        self.assertFalse(any(local_b.schedule_valid_mask))
        self.assertEqual(game_b.schedule_statuses[:5], [1] * 5)
        self.assertEqual(game_b.schedule_statuses[5], 2)

    def test_default_full_game_schedule_does_not_duplicate_minimum(self):
        self.assertEqual(
            descending_ascending_schedule(min_cards=3, max_cards=6),
            [6, 5, 4, 3, 4, 5, 6],
        )

    def test_v3_warm_start_replaces_only_owner_head_and_preserves_shared_outputs(self):
        import torch

        torch.manual_seed(7)
        config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            game_hidden_dim=32,
            schedule_heads=4,
        )
        source = PlumpTransformerModel(config).eval()
        v3_state = {
            key: value.clone()
            for key, value in source.state_dict().items()
            if not key.startswith(V4_OWNER_PARAMETER_PREFIXES)
        }
        v3_state["owner_card_emb.weight"] = torch.randn(
            52,
            config.d_model,
        )
        v3_state["owner_head.0.weight"] = torch.randn(
            config.d_model,
            2 * config.d_model,
        )
        v3_state["owner_head.0.bias"] = torch.randn(config.d_model)
        v3_state["owner_head.2.weight"] = torch.randn(
            config.owner_class_count,
            config.d_model,
        )
        v3_state["owner_head.2.bias"] = torch.randn(
            config.owner_class_count
        )
        target = PlumpTransformerModel(config).eval()
        migration = load_v3_weights(target, v3_state)
        self.assertTrue(migration["fresh"])
        self.assertTrue(
            all(
                key.startswith(V4_OWNER_PARAMETER_PREFIXES)
                for key in migration["fresh"]
            )
        )
        self.assertEqual(
            set(migration["dropped"]),
            {
                "owner_card_emb.weight",
                "owner_head.0.weight",
                "owner_head.0.bias",
                "owner_head.2.weight",
                "owner_head.2.bias",
            },
        )

        encoded = encode_observation(
            _env_with_history().get_observation(1),
            config,
        )
        batch = encoded_observations_to_batch([encoded], device="cpu")
        with torch.no_grad():
            expected = source(batch)
            actual = target(batch)
        self.assertTrue(torch.equal(expected.bid_logits, actual.bid_logits))
        self.assertTrue(torch.equal(expected.card_logits, actual.card_logits))
        self.assertTrue(torch.equal(expected.round_value, actual.round_value))
        self.assertTrue(
            torch.equal(
                expected.trick_count_logits,
                actual.trick_count_logits,
            )
        )

    def test_game_context_selects_game_value_head_only_when_enabled(self):
        import torch

        config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            game_hidden_dim=32,
            schedule_heads=4,
        )
        observation = _env_with_history().get_observation(1)
        local = encode_observation(observation, config)
        game = encode_observation(
            observation,
            config,
            include_game_context=True,
        )
        model = PlumpTransformerModel(config).eval()
        with torch.no_grad():
            local_output = model(
                encoded_observations_to_batch([local], device="cpu")
            )
            game_output = model(
                encoded_observations_to_batch([game], device="cpu")
            )
        self.assertTrue(
            torch.equal(local_output.value, local_output.round_value)
        )
        self.assertTrue(
            torch.equal(game_output.value, game_output.game_value)
        )

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_torch_model_forward_shapes(self):
        import torch

        from plump.modeling.torch_model import combined_action_logits

        config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
        )
        encoded = encode_observation(
            _env_with_history().get_observation(1),
            config,
        )
        batch = encoded_observations_to_batch([encoded], device="cpu")
        output = PlumpTransformerModel(config)(batch)

        self.assertEqual(output.state.shape, (1, config.d_model))
        self.assertEqual(output.bid_logits.shape, (1, config.bid_count))
        self.assertEqual(output.card_logits.shape, (1, 52))
        self.assertEqual(output.value.shape, (1, 1))
        self.assertEqual(
            output.trick_count_logits.shape,
            (1, config.max_players, config.bid_count),
        )
        self.assertEqual(
            output.owner_logits.shape,
            (1, 52, config.owner_class_count),
        )
        active_owner_rows = batch.owner_valid_mask.any(dim=-1)
        self.assertTrue(
            torch.allclose(
                output.owner_probs.sum(dim=-1)[active_owner_rows],
                torch.ones_like(
                    output.owner_probs.sum(dim=-1)[active_owner_rows]
                ),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                output.owner_probs.sum(dim=1),
                batch.owner_capacities,
                atol=1e-5,
            )
        )
        self.assertEqual(output.hit_bid_probs.shape, (1, config.max_players))
        self.assertEqual(output.score_probs.shape, (1, config.max_players))
        self.assertTrue(
            (output.masked_trick_count_logits[~batch.final_trick_count_mask] < -1e30).all()
        )
        self.assertTrue(
            (output.masked_owner_logits[~batch.owner_valid_mask] < -1e30).all()
        )
        self.assertTrue(torch.isfinite(output.value).all())

        bid_logits = combined_action_logits(
            output,
            torch.tensor([True]),
        )
        play_logits = combined_action_logits(
            output,
            torch.tensor([False]),
        )
        self.assertEqual(bid_logits.shape, (1, 52))
        self.assertTrue(torch.isneginf(bid_logits[:, config.bid_count:]).all())
        self.assertTrue(torch.equal(play_logits, output.masked_card_logits.float()))

    def test_owner_beliefs_do_not_depend_on_true_hidden_hands(self):
        import torch

        config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            game_hidden_dim=32,
            schedule_heads=4,
        )
        first = _env_with_history()
        second = copy.deepcopy(first)
        second.state.current_round.current_hands[2], (
            second.state.current_round.current_hands[3]
        ) = (
            second.state.current_round.current_hands[3],
            second.state.current_round.current_hands[2],
        )
        observations = [
            encode_observation(first.get_observation(1), config),
            encode_observation(second.get_observation(1), config),
        ]
        model = PlumpTransformerModel(config).eval()
        with torch.no_grad():
            output = model(
                encoded_observations_to_batch(
                    observations,
                    device="cpu",
                )
            )
        self.assertTrue(
            torch.allclose(
                output.owner_probs[0],
                output.owner_probs[1],
                atol=2e-7,
                rtol=1e-6,
            )
        )

    def test_masked_sinkhorn_satisfies_rows_columns_and_exclusions(self):
        import torch

        logits = torch.tensor(
            [
                [
                    [1.0, 2.0, -1.0],
                    [0.5, -0.5, 1.0],
                    [2.0, 1.0, 0.0],
                    [-1.0, 0.5, 2.0],
                ]
            ],
            requires_grad=True,
        )
        valid = torch.tensor(
            [
                [
                    [True, True, False],
                    [True, False, True],
                    [True, True, False],
                    [False, True, True],
                ]
            ]
        )
        capacities = torch.tensor([[1.0, 2.0, 1.0]])

        probabilities = masked_capacity_sinkhorn(
            logits,
            valid,
            capacities,
            iterations=64,
        )

        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones((1, 4)),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                capacities,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.equal(
                probabilities[~valid],
                torch.zeros_like(probabilities[~valid]),
            )
        )

        targets = torch.tensor([[1, 2, 0, 1]])
        selected = probabilities.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)
        loss = -selected.clamp_min(1e-12).log().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_sinkhorn_capacity_condition_changes_assignment(self):
        import torch

        logits = torch.tensor(
            [
                [
                    [2.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                ],
                [
                    [2.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                ],
            ]
        )
        valid = torch.ones_like(logits, dtype=torch.bool)
        capacities = torch.tensor([[2.0, 2.0], [1.0, 3.0]])

        probabilities = masked_capacity_sinkhorn(
            logits,
            valid,
            capacities,
            iterations=64,
        )

        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                capacities,
                atol=1e-5,
            )
        )
        self.assertFalse(torch.allclose(probabilities[0], probabilities[1]))

class OracleCriticModelTest(unittest.TestCase):
    def _config(self, *, oracle_critic: bool) -> ModelConfig:
        return ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            oracle_critic=oracle_critic,
        )

    def test_oracle_head_is_gated_by_config(self):
        plain = PlumpTransformerModel(self._config(oracle_critic=False))
        self.assertFalse(
            [name for name, _ in plain.named_parameters() if name.startswith("oracle_")]
        )
        oracle = PlumpTransformerModel(self._config(oracle_critic=True))
        self.assertTrue(
            [name for name, _ in oracle.named_parameters() if name.startswith("oracle_")]
        )

    def test_oracle_value_requires_head_and_targets(self):
        import torch

        config = self._config(oracle_critic=True)
        encoded = encode_observation(
            _env_with_history().get_observation(1),
            config,
        )
        batch = encoded_observations_to_batch([encoded], device="cpu")
        model = PlumpTransformerModel(config)
        targets = torch.full((1, 52), -100, dtype=torch.long)
        targets[0, card_id(Card(Suit.DIAMONDS, Rank.ACE))] = 0
        targets[0, card_id(Card(Suit.SPADES, Rank.TWO))] = 1

        output = model(batch, privileged_owner_targets=targets)
        self.assertEqual(output.oracle_value.shape, (1, 1))
        self.assertTrue(torch.isfinite(output.oracle_value).all())
        self.assertIsNone(model(batch).oracle_value)

        plain = PlumpTransformerModel(self._config(oracle_critic=False))
        with self.assertRaises(ValueError):
            plain(batch, privileged_owner_targets=targets)

    def test_oracle_value_depends_on_privileged_targets_not_policy(self):
        import torch

        config = self._config(oracle_critic=True)
        encoded = encode_observation(
            _env_with_history().get_observation(1),
            config,
        )
        batch = encoded_observations_to_batch([encoded], device="cpu")
        model = PlumpTransformerModel(config)
        # Same class counts, swapped card-to-owner assignment: only the
        # private "who holds what" association differs. A pooling that
        # collapses to public information would leave the output unchanged.
        ace = card_id(Card(Suit.DIAMONDS, Rank.ACE))
        deuce = card_id(Card(Suit.SPADES, Rank.TWO))
        targets_a = torch.full((1, 52), -100, dtype=torch.long)
        targets_a[0, ace] = 0
        targets_a[0, deuce] = 1
        targets_b = torch.full((1, 52), -100, dtype=torch.long)
        targets_b[0, ace] = 1
        targets_b[0, deuce] = 0

        output_a = model(batch, privileged_owner_targets=targets_a)
        output_b = model(batch, privileged_owner_targets=targets_b)
        self.assertFalse(
            torch.allclose(output_a.oracle_value, output_b.oracle_value)
        )
        # The policy and plain value must be unaffected by privileged inputs.
        self.assertTrue(torch.equal(output_a.bid_logits, output_b.bid_logits))
        self.assertTrue(torch.equal(output_a.card_logits, output_b.card_logits))
        self.assertTrue(torch.equal(output_a.value, output_b.value))

    def test_v4_warm_start_drops_oracle_head_weights(self):
        from plump.modeling.torch_model import PlumpSearchModel, load_v4_weights

        source = PlumpTransformerModel(self._config(oracle_critic=True))
        target = PlumpSearchModel(self._config(oracle_critic=False))

        migration = load_v4_weights(target, source.state_dict())

        self.assertTrue(migration["dropped"])
        self.assertTrue(
            all(key.startswith("oracle_") for key in migration["dropped"])
        )
        self.assertTrue(
            all(
                key.startswith(("bid_q_head.", "card_q_head."))
                for key in migration["fresh"]
            )
        )

    def test_suit_presence_head_gated_and_dropped_on_v4_warm_start(self):
        import dataclasses

        import torch

        from plump.modeling.torch_model import PlumpSearchModel, load_v4_weights

        config = dataclasses.replace(
            self._config(oracle_critic=False),
            suit_presence_head=True,
        )
        model = PlumpTransformerModel(config)
        self.assertTrue(
            [
                name
                for name, _ in model.named_parameters()
                if name.startswith("suit_presence_head.")
            ]
        )
        encoded = encode_observation(
            _env_with_history().get_observation(1),
            config,
        )
        batch = encoded_observations_to_batch([encoded], device="cpu")
        output = model(batch)
        self.assertEqual(
            output.suit_presence_logits.shape,
            (1, config.max_players, 4),
        )
        self.assertTrue(torch.isfinite(output.suit_presence_logits).all())

        plain = PlumpTransformerModel(self._config(oracle_critic=False))
        self.assertIsNone(plain(batch).suit_presence_logits)

        target = PlumpSearchModel(self._config(oracle_critic=False))
        migration = load_v4_weights(target, model.state_dict())
        self.assertTrue(
            all(
                key.startswith("suit_presence_head.")
                for key in migration["dropped"]
            )
        )
        self.assertTrue(migration["dropped"])


if __name__ == "__main__":
    unittest.main()
