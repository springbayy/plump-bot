import unittest

from plump.gui.app import GuiController
from plump.state import Phase


class GuiControllerTest(unittest.TestCase):
    def test_new_game_honors_human_bid_position(self):
        controller = GuiController()

        view = controller.new_game(opponents=3, hand_size=3, human_bid_position=3, seed=1)

        round_state = controller.game.env.state.current_round
        self.assertEqual(round_state.bidding_order, [2, 3, 0, 1])
        self.assertEqual(view["current_player"], 0)
        self.assertEqual(view["phase"], Phase.BIDDING.value)
        self.assertEqual(len(view["bids"]), 2)
        self.assertTrue(view["legal_bids"])

    def test_human_bid_advances_random_models_to_next_human_turn_or_done(self):
        controller = GuiController()
        controller.new_game(opponents=2, hand_size=3, human_bid_position=1, seed=2)

        view = controller.bid(0)

        self.assertEqual(view["phase"], Phase.PLAYING.value)
        self.assertEqual(len(view["players"]), 3)

    def test_invalid_setup_rejected(self):
        controller = GuiController()

        with self.assertRaises(ValueError):
            controller.new_game(opponents=6, hand_size=9, human_bid_position=1, seed=1)

        with self.assertRaises(ValueError):
            controller.new_game(opponents=5, hand_size=2, human_bid_position=1, seed=1)

        with self.assertRaises(ValueError):
            controller.new_game(opponents=2, hand_size=2, human_bid_position=1, seed=1)


if __name__ == "__main__":
    unittest.main()
