import unittest

from haggis import Rank, Suit, deal, player_wilds, standard_deck


class CardAndDealTests(unittest.TestCase):
    def test_standard_deck_has_two_through_ten_in_four_suits(self):
        deck = standard_deck()

        self.assertEqual(len(deck), 36)
        self.assertEqual({card.rank for card in deck}, set(Rank(value) for value in range(2, 11)))
        for rank in range(2, 11):
            self.assertEqual(sum(1 for card in deck if card.rank == Rank(rank)), 4)
        for card in deck:
            self.assertFalse(card.is_wild)

    def test_deal_gives_each_player_fourteen_cards_three_wilds_and_haggis(self):
        dealt = deal(seed=7)

        self.assertEqual([len(hand) for hand in dealt.hands], [17, 17])
        self.assertEqual(len(dealt.haggis), 8)
        for player_index, hand in enumerate(dealt.hands):
            wilds = [card for card in hand if card.is_wild]
            self.assertEqual(tuple(wilds), player_wilds(player_index))
            self.assertEqual(len([card for card in hand if not card.is_wild]), 14)

    def test_wild_card_suits_are_stable_but_rules_irrelevant(self):
        self.assertEqual({card.suit for card in player_wilds(0)}, {Suit.HEARTS})
        self.assertEqual({card.suit for card in player_wilds(1)}, {Suit.SPADES})


if __name__ == "__main__":
    unittest.main()
