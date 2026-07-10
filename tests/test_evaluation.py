import math
import unittest

from plump.evaluation import (
    DealBank,
    check_round_policy_compatibility,
    evaluate_full_games,
    evaluate_paired,
    evaluate_policy,
)
from plump.policies import HeuristicPolicy, RandomPolicy


class EvaluationTest(unittest.TestCase):
    def test_deal_bank_is_deterministic_and_rotates_all_positions(self):
        first = DealBank.generate(
            player_counts=(3,),
            hand_sizes=(3,),
            deals_per_configuration=1,
            seed=9,
        )
        second = DealBank.generate(
            player_counts=(3,),
            hand_sizes=(3,),
            deals_per_configuration=1,
            seed=9,
        )
        self.assertEqual(first.deals, second.deals)

        report = evaluate_policy(
            HeuristicPolicy(),
            RandomPolicy(1),
            first,
            bootstrap_samples=20,
            seed=5,
        )
        serial = evaluate_policy(
            HeuristicPolicy(),
            RandomPolicy(1),
            first,
            bootstrap_samples=20,
            seed=5,
            batch_size=1,
        )
        self.assertEqual(report.rounds, 9)
        self.assertEqual({cell.bidding_position for cell in report.cells}, {0, 1, 2})
        self.assertTrue(math.isfinite(report.macro_relative_reward))
        self.assertEqual(report.results, serial.results)

    def test_paired_identical_policies_have_zero_delta(self):
        bank = DealBank.generate(
            player_counts=(3,),
            hand_sizes=(3,),
            deals_per_configuration=1,
            seed=4,
        )
        report = evaluate_paired(
            HeuristicPolicy(),
            HeuristicPolicy(),
            RandomPolicy(2),
            bank,
            bootstrap_samples=20,
            seed=8,
        )
        self.assertEqual(report.macro_relative_reward_delta, 0.0)
        self.assertEqual(report.worst_cell_delta, 0.0)
        self.assertFalse(report.passes_gate())

    def test_full_game_performance_is_a_separate_measure(self):
        compatibility = check_round_policy_compatibility(
            HeuristicPolicy(),
            RandomPolicy(3),
            num_players=3,
            hand_sizes=(3, 4, 3),
            seed=6,
        )
        report = evaluate_full_games(
            HeuristicPolicy(),
            RandomPolicy(3),
            num_players=3,
            games=2,
            hand_sizes=(3,),
            seed=6,
        )
        self.assertTrue(compatibility.completed)
        self.assertEqual(compatibility.rounds, 3)
        self.assertEqual(report.games, 2)
        self.assertTrue(math.isfinite(report.average_cumulative_relative_score))
        self.assertEqual(set(report.focal_seat_relative_score), {"0", "1"})
        self.assertEqual(
            set(report.initial_bidding_position_relative_score),
            {"0"},
        )


if __name__ == "__main__":
    unittest.main()
