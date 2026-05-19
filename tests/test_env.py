import unittest

from plump.cards import Card, Rank, Suit
from plump.env import PlumpEnv
from plump.state import BidAction, EventType, GameConfig, IllegalActionError, Phase, PlayCardAction, TrumpPolicy


class EnvTest(unittest.TestCase):
    def test_illegal_play_rejected_when_player_can_follow_suit(self):
        hands = {
            0: [Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.TWO)],
            1: [Card(Suit.SPADES, Rank.TWO), Card(Suit.HEARTS, Rank.ACE)],
        }
        env = PlumpEnv(
            GameConfig(
                num_players=2,
                hand_sizes=[2],
                manual_hands=hands,
                manual_trump_suit=None,
                trump_policy=TrumpPolicy.NONE,
                forbid_total_bid_equals_hand_size=False,
            )
        )
        env.reset()
        env.step(BidAction(0, 1))
        env.step(BidAction(1, 0))
        env.step(PlayCardAction(0, Card(Suit.SPADES, Rank.ACE)))

        with self.assertRaises(IllegalActionError):
            env.step(PlayCardAction(1, Card(Suit.HEARTS, Rank.ACE)))

    def test_full_mini_round_scores_and_ends_game(self):
        hands = {
            0: [Card(Suit.SPADES, Rank.ACE)],
            1: [Card(Suit.SPADES, Rank.KING)],
            2: [Card(Suit.HEARTS, Rank.TWO)],
            3: [Card(Suit.SPADES, Rank.THREE)],
        }
        env = PlumpEnv(
            GameConfig(
                num_players=4,
                hand_sizes=[1],
                manual_hands=hands,
                trump_policy=TrumpPolicy.NONE,
                forbid_total_bid_equals_hand_size=False,
            )
        )
        state = env.reset()

        self.assertEqual(state.phase, Phase.BIDDING)
        self.assertEqual(env.current_player(), 0)

        env.step(BidAction(0, 1))
        env.step(BidAction(1, 0))
        env.step(BidAction(2, 0))
        env.step(BidAction(3, 0))

        self.assertEqual(env.phase(), Phase.PLAYING)
        self.assertEqual(env.current_player(), 0)

        env.step(PlayCardAction(0, Card(Suit.SPADES, Rank.ACE)))
        env.step(PlayCardAction(1, Card(Suit.SPADES, Rank.KING)))
        env.step(PlayCardAction(2, Card(Suit.HEARTS, Rank.TWO)))
        result = env.step(PlayCardAction(3, Card(Suit.SPADES, Rank.THREE)))

        first_round = result.state.rounds[0]
        self.assertTrue(result.done)
        self.assertEqual(result.state.phase, Phase.GAME_OVER)
        self.assertEqual(first_round.tricks[0].winner, 0)
        self.assertEqual(first_round.tricks_won, {0: 1, 1: 0, 2: 0, 3: 0})
        self.assertEqual(first_round.round_scores, {0: 11, 1: 5, 2: 5, 3: 5})
        self.assertEqual(result.rewards, {0: 11, 1: 5, 2: 5, 3: 5})
        self.assertIn(EventType.ROUND_END, [event.type for event in result.state.event_log])

    def test_observation_hides_opponent_hands_and_includes_legal_actions(self):
        hands = {
            0: [Card(Suit.SPADES, Rank.ACE)],
            1: [Card(Suit.HEARTS, Rank.KING)],
        }
        env = PlumpEnv(
            GameConfig(
                num_players=2,
                hand_sizes=[1],
                manual_hands=hands,
                trump_policy=TrumpPolicy.NONE,
                forbid_total_bid_equals_hand_size=False,
            )
        )
        env.reset()

        obs0 = env.get_observation(0)

        self.assertEqual(obs0.my_hand, [Card(Suit.SPADES, Rank.ACE)])
        self.assertEqual(obs0.legal_bids, [0, 1])
        self.assertEqual(obs0.played_cards_total, [])
        self.assertFalse(any(Card(Suit.HEARTS, Rank.KING) in cards for cards in obs0.played_cards_by_player.values()))


if __name__ == "__main__":
    unittest.main()
