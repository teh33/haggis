import unittest

from haggis import Card, HaggisState, InvariantError, Move, Rank, Suit, deal, validate_combination
from haggis.tournament import play_hand
from haggis.bots import GreedySheddingBot


def c(rank, suit=Suit.CLUBS, wild=False):
    return Card(Rank(rank), suit, wild)


class InvariantTests(unittest.TestCase):
    def test_new_deal_satisfies_full_deck_invariants(self):
        state = HaggisState.new_deal(seed=1)

        report = state.validate_invariants(full_deck=True)

        self.assertTrue(report.ok, report.errors)

    def test_duplicate_card_across_zones_is_reported(self):
        duplicated = c(7)
        state = HaggisState(hands=((duplicated,), (duplicated,)))

        report = state.validate_invariants()

        self.assertFalse(report.ok)
        self.assertIn("duplicate cards", report.errors[0])

    def test_full_deck_invariants_detect_lost_cards(self):
        dealt = deal(seed=2)
        state = HaggisState.from_deal(dealt)
        state_with_lost_card = HaggisState(
            hands=(state.hands[0][1:], state.hands[1]),
            haggis=state.haggis,
        )

        report = state_with_lost_card.validate_invariants(full_deck=True)

        self.assertFalse(report.ok)
        self.assertTrue(any("missing cards" in error for error in report.errors))

    def test_assert_invariants_raises_for_invalid_state(self):
        invalid = HaggisState(hands=((c(7),), ()), hand_winner=0)

        with self.assertRaises(InvariantError):
            invalid.assert_invariants()

    def test_generated_legal_moves_are_playable_on_seeded_deals(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                state = HaggisState.new_deal(seed=seed).assert_invariants(full_deck=True)
                for move in state.legal_moves():
                    next_state = state.apply_move(move)
                    self.assertTrue(next_state.validate_invariants(full_deck=True).ok)

    def test_generated_response_moves_are_playable(self):
        state = HaggisState(
            hands=((c(8), c(9), c(11, wild=True), c(12, wild=True)), (c(7, Suit.HEARTS), c(2))),
            current_player=1,
        )
        state = state.apply_move(Move((c(7, Suit.HEARTS),), validate_combination((c(7, Suit.HEARTS),))))

        for move in state.legal_moves():
            with self.subTest(move=move):
                next_state = state.apply_move(move)
                self.assertTrue(next_state.validate_invariants().ok)

    def test_tournament_self_play_checks_invariants(self):
        result = play_hand((GreedySheddingBot(), GreedySheddingBot()), seed=3)

        self.assertIn(result.winner, (0, 1))
        self.assertIn(0, result.cards_remaining)


if __name__ == "__main__":
    unittest.main()
