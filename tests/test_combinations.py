import unittest

from haggis import Card, CombinationType, Rank, Suit, can_beat, possible_combinations, validate_combination


def c(rank, suit=Suit.CLUBS, wild=False):
    return Card(Rank(rank), suit, wild)


class CombinationTests(unittest.TestCase):
    def test_sets_may_use_higher_wilds_as_lower_rank_cards(self):
        combo = validate_combination((c(7, Suit.CLUBS), c(7, Suit.HEARTS), c(11, wild=True)))

        self.assertIsNotNone(combo)
        self.assertEqual(combo.type, CombinationType.SET)
        self.assertEqual(combo.rank, 7)

    def test_multiple_wilds_are_not_an_ordinary_set(self):
        self.assertIsNone(validate_combination((c(11, wild=True), c(11, Suit.HEARTS, wild=True))))

    def test_single_wild_can_be_played_as_a_single_at_its_printed_rank(self):
        combo = validate_combination((c(12, wild=True),))

        self.assertEqual(combo.type, CombinationType.SET)
        self.assertEqual(combo.rank, 12)

    def test_sequence_requires_consistent_suit_pattern(self):
        valid = validate_combination(
            (
                c(7, Suit.HEARTS),
                c(7, Suit.SPADES),
                c(8, Suit.HEARTS),
                c(8, Suit.SPADES),
            )
        )
        invalid = validate_combination(
            (
                c(7, Suit.HEARTS),
                c(7, Suit.SPADES),
                c(8, Suit.DIAMONDS),
                c(8, Suit.SPADES),
            )
        )

        self.assertEqual(valid.type, CombinationType.SEQUENCE)
        self.assertIsNone(invalid)

    def test_wild_cards_have_multiple_possible_interpretations(self):
        combos = possible_combinations((c(6, Suit.CLUBS), c(11, wild=True), c(12, wild=True)))

        self.assertIn((CombinationType.SET, 6), {(combo.type, combo.rank) for combo in combos})
        self.assertIn((CombinationType.SEQUENCE, 8), {(combo.type, combo.rank) for combo in combos})

    def test_set_interpretation_can_beat_same_size_set(self):
        previous = validate_combination((c(5, Suit.CLUBS), c(5, Suit.DIAMONDS), c(11, wild=True)))
        new = validate_combination((c(6, Suit.CLUBS), c(11, wild=True), c(12, wild=True)))

        self.assertIsNotNone(previous)
        self.assertIsNotNone(new)
        self.assertEqual(previous.type, CombinationType.SET)
        self.assertEqual(previous.rank, 5)
        self.assertEqual(new.type, CombinationType.SEQUENCE)
        self.assertEqual(new.rank, 8)
        self.assertTrue(can_beat(new, previous))

    def test_wild_sequence_interpretation_takes_precedence_over_lower_set(self):
        previous = validate_combination((c(7), c(9), c(13, wild=True)))
        new = validate_combination((c(10), c(12, wild=True), c(13, wild=True)))

        self.assertIsNotNone(previous)
        self.assertIsNotNone(new)
        self.assertEqual(previous.type, CombinationType.SEQUENCE)
        self.assertEqual(previous.rank, 9)
        self.assertEqual(new.type, CombinationType.SEQUENCE)
        self.assertEqual(new.rank, 12)
        self.assertTrue(can_beat(new, previous))

    def test_wilds_can_fill_sequence_gaps_with_required_suit(self):
        combo = validate_combination((c(7, Suit.HEARTS), c(8, Suit.HEARTS), c(11, wild=True)))

        self.assertIsNotNone(combo)
        self.assertEqual(combo.type, CombinationType.SEQUENCE)
        self.assertEqual(combo.rank, 9)
        self.assertEqual(combo.sequence_width, 1)
        self.assertEqual(combo.sequence_length, 3)

    def test_bombs_are_ranked_by_official_order(self):
        cases = [
            ((c(3, Suit.CLUBS), c(5, Suit.DIAMONDS), c(7, Suit.HEARTS), c(9, Suit.SPADES)), 1),
            ((c(11, wild=True), c(12, wild=True)), 2),
            ((c(11, wild=True), c(13, wild=True)), 3),
            ((c(12, wild=True), c(13, wild=True)), 4),
            ((c(11, wild=True), c(12, wild=True), c(13, wild=True)), 5),
            ((c(3, Suit.HEARTS), c(5, Suit.HEARTS), c(7, Suit.HEARTS), c(9, Suit.HEARTS)), 6),
        ]

        for cards, expected_rank in cases:
            with self.subTest(expected_rank=expected_rank):
                combo = validate_combination(cards)
                self.assertEqual(combo.type, CombinationType.BOMB)
                self.assertEqual(combo.bomb_rank, expected_rank)

    def test_wilds_cannot_make_three_five_seven_nine_bombs(self):
        self.assertIsNone(validate_combination((c(3), c(5), c(7), c(11, wild=True))))

    def test_bomb_beats_non_bomb_and_only_higher_bomb_beats_bomb(self):
        pair = validate_combination((c(7, Suit.CLUBS), c(7, Suit.SPADES)))
        low_bomb = validate_combination((c(11, wild=True), c(12, wild=True)))
        high_bomb = validate_combination((c(12, wild=True), c(13, wild=True)))

        self.assertTrue(can_beat(low_bomb, pair))
        self.assertFalse(can_beat(pair, low_bomb))
        self.assertTrue(can_beat(high_bomb, low_bomb))
        self.assertFalse(can_beat(low_bomb, high_bomb))

    def test_sequences_must_match_width_and_length_to_beat(self):
        single_run = validate_combination((c(7, Suit.CLUBS), c(8, Suit.CLUBS), c(9, Suit.CLUBS)))
        pair_run = validate_combination(
            (c(8, Suit.CLUBS), c(8, Suit.SPADES), c(9, Suit.CLUBS), c(9, Suit.SPADES))
        )
        higher_single_run = validate_combination((c(8, Suit.CLUBS), c(9, Suit.CLUBS), c(10, Suit.CLUBS)))

        self.assertFalse(can_beat(pair_run, single_run))
        self.assertTrue(can_beat(higher_single_run, single_run))


if __name__ == "__main__":
    unittest.main()
