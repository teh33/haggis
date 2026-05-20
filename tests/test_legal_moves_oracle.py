from itertools import combinations
from random import Random
import unittest

from haggis import Card, Rank, Suit, can_beat, legal_moves, player_wilds, standard_deck, validate_combination
from haggis.engine import Move


def c(rank, suit=Suit.CLUBS, wild=False):
    return Card(Rank(rank), suit, wild)


def brute_force_legal_moves(hand, previous=None):
    moves = {}
    for size in range(1, len(hand) + 1):
        for cards in combinations(hand, size):
            combination = validate_combination(cards)
            if combination is None or not can_beat(combination, previous):
                continue
            move = Move(cards=tuple(sorted(cards)), combination=combination)
            moves[_move_key(move)] = move

    ordered = sorted(moves.values(), key=lambda move: move.combination.sort_key())
    if previous is not None:
        ordered.append(Move.pass_turn())
    return tuple(ordered)


def move_keys(moves):
    return {_move_key(move) for move in moves}


def _move_key(move):
    if move.is_pass:
        return ("pass",)
    combo = move.combination
    return (
        tuple(card.short_name() for card in move.cards),
        str(combo.type),
        combo.rank,
        combo.bomb_rank,
        combo.sequence_width,
        combo.sequence_length,
    )


class LegalMoveOracleTests(unittest.TestCase):
    def assert_matches_oracle(self, hand, previous=None, *, context):
        optimized = legal_moves(tuple(sorted(hand)), previous)
        oracle = brute_force_legal_moves(tuple(sorted(hand)), previous)
        optimized_keys = move_keys(optimized)
        oracle_keys = move_keys(oracle)

        missing = sorted(oracle_keys - optimized_keys)
        extra = sorted(optimized_keys - oracle_keys)
        self.assertEqual(
            optimized_keys,
            oracle_keys,
            msg=(
                f"legal move oracle mismatch in {context}\n"
                f"hand={[card.short_name() for card in sorted(hand)]}\n"
                f"previous={previous}\n"
                f"missing={missing}\n"
                f"extra={extra}"
            ),
        )

    def test_curated_leading_hands_match_brute_force_oracle(self):
        hands = [
            # Simple sets and single-suit runs.
            (c(3), c(3, Suit.HEARTS), c(4), c(5), c(6), c(7)),
            # Paired sequences with a consistent suit pattern.
            (c(7, Suit.HEARTS), c(7, Suit.SPADES), c(8, Suit.HEARTS), c(8, Suit.SPADES), c(9)),
            # Low and high 3/5/7/9 bombs.
            (c(3), c(5, Suit.DIAMONDS), c(7, Suit.HEARTS), c(9, Suit.SPADES), c(3, Suit.HEARTS), c(5, Suit.HEARTS), c(7, Suit.HEARTS), c(9, Suit.HEARTS)),
            # Wild bombs and wild-assisted sets/sequences.
            (c(7), c(8), c(9), c(11, wild=True), c(12, wild=True), c(13, wild=True)),
        ]

        for index, hand in enumerate(hands):
            with self.subTest(index=index):
                self.assert_matches_oracle(hand, context=f"curated leader {index}")

    def test_curated_response_hands_match_brute_force_oracle(self):
        cases = [
            (
                (c(6), c(7), c(8), c(9), c(11, wild=True), c(12, wild=True)),
                validate_combination((c(5, Suit.HEARTS),)),
                "single response with wild bombs",
            ),
            (
                (c(7, Suit.HEARTS), c(7, Suit.SPADES), c(8, Suit.HEARTS), c(8, Suit.SPADES), c(9, Suit.HEARTS), c(9, Suit.SPADES)),
                validate_combination((c(6, Suit.CLUBS), c(6, Suit.DIAMONDS))),
                "pair response and paired sequences",
            ),
            (
                (c(3), c(5, Suit.DIAMONDS), c(7, Suit.HEARTS), c(9, Suit.SPADES), c(11, wild=True), c(13, wild=True)),
                validate_combination((c(11, Suit.HEARTS, wild=True), c(12, Suit.HEARTS, wild=True))),
                "bomb response",
            ),
            (
                (c(8, Suit.CLUBS), c(9, Suit.CLUBS), c(10, Suit.CLUBS), c(11, wild=True), c(12, wild=True)),
                validate_combination((c(5, Suit.CLUBS), c(6, Suit.CLUBS), c(7, Suit.CLUBS))),
                "sequence response with wild gaps",
            ),
        ]

        for hand, previous, context in cases:
            with self.subTest(context=context):
                self.assert_matches_oracle(hand, previous, context=context)

    def test_seeded_small_leading_hands_match_brute_force_oracle(self):
        deck = list(standard_deck()) + list(player_wilds(0))
        rng = Random(20260520)

        for sample_index in range(30):
            hand_size = rng.randint(5, 8)
            hand = tuple(sorted(rng.sample(deck, hand_size)))
            with self.subTest(sample_index=sample_index, hand=[card.short_name() for card in hand]):
                self.assert_matches_oracle(hand, context=f"seeded leader {sample_index}")

    def test_seeded_small_response_hands_match_brute_force_oracle(self):
        deck = list(standard_deck()) + list(player_wilds(0))
        previous_options = [
            validate_combination((c(5, Suit.CLUBS),)),
            validate_combination((c(6, Suit.CLUBS), c(6, Suit.HEARTS))),
            validate_combination((c(5, Suit.CLUBS), c(6, Suit.CLUBS), c(7, Suit.CLUBS))),
            validate_combination((c(11, Suit.HEARTS, wild=True), c(12, Suit.HEARTS, wild=True))),
        ]
        rng = Random(20260521)

        for sample_index in range(30):
            hand_size = rng.randint(5, 8)
            hand = tuple(sorted(rng.sample(deck, hand_size)))
            previous = previous_options[sample_index % len(previous_options)]
            with self.subTest(sample_index=sample_index, previous=previous, hand=[card.short_name() for card in hand]):
                self.assert_matches_oracle(hand, previous, context=f"seeded response {sample_index}")


if __name__ == "__main__":
    unittest.main()
