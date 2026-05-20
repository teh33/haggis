from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from random import Random


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13

    @property
    def label(self) -> str:
        return {11: "J", 12: "Q", 13: "K"}.get(int(self), str(int(self)))


class Suit(StrEnum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"

    @property
    def symbol(self) -> str:
        return {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠",
        }[self]


SUITS: tuple[Suit, ...] = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
NATURAL_RANKS: tuple[Rank, ...] = tuple(Rank(value) for value in range(2, 11))
WILD_RANKS: tuple[Rank, ...] = (Rank.JACK, Rank.QUEEN, Rank.KING)
POINT_VALUES: dict[Rank, int] = {
    Rank.THREE: 1,
    Rank.FIVE: 1,
    Rank.SEVEN: 1,
    Rank.NINE: 1,
    Rank.JACK: 2,
    Rank.QUEEN: 3,
    Rank.KING: 5,
}


@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit
    is_wild: bool = False

    @property
    def points(self) -> int:
        return POINT_VALUES.get(self.rank, 0)

    @property
    def is_point_card(self) -> bool:
        return self.points > 0

    def short_name(self) -> str:
        suffix = "*" if self.is_wild else ""
        return f"{self.rank.label}{self.suit.symbol}{suffix}"

    def __str__(self) -> str:
        return self.short_name()


Hand = tuple[Card, ...]


def standard_deck() -> tuple[Card, ...]:
    """Return the 36 shuffled-card deck: ranks 2-10 in four suits."""
    return tuple(Card(rank, suit) for rank in NATURAL_RANKS for suit in SUITS)


def player_wilds(player_index: int) -> tuple[Card, ...]:
    """Return one J/Q/K wild set for a player.

    Suits are irrelevant for wild cards, but stable suits keep physical cards distinct
    in logs and tests.
    """
    suit = Suit.HEARTS if player_index == 0 else Suit.SPADES
    return tuple(Card(rank, suit, is_wild=True) for rank in WILD_RANKS)


@dataclass(frozen=True)
class Deal:
    hands: tuple[Hand, Hand]
    haggis: Hand


def deal(seed: int | None = None, rng: Random | None = None) -> Deal:
    """Deal a two-player Haggis hand: 14 natural cards + J/Q/K wilds each."""
    random = rng if rng is not None else Random(seed)
    deck = list(standard_deck())
    random.shuffle(deck)

    hand0 = tuple(sorted((*deck[:14], *player_wilds(0))))
    hand1 = tuple(sorted((*deck[14:28], *player_wilds(1))))
    return Deal(hands=(hand0, hand1), haggis=tuple(deck[28:]))


def point_total(cards: tuple[Card, ...] | list[Card]) -> int:
    return sum(card.points for card in cards)
