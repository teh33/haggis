from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from .cards import Card, Rank, SUITS, Suit


class CombinationType(StrEnum):
    SET = "set"
    SEQUENCE = "sequence"
    BOMB = "bomb"


@dataclass(frozen=True)
class Combination:
    cards: tuple[Card, ...]
    type: CombinationType
    rank: int
    bomb_rank: int = 0
    sequence_width: int = 0
    sequence_length: int = 0

    @property
    def card_count(self) -> int:
        return len(self.cards)

    def sort_key(self) -> tuple[int, int, int, int, tuple[str, ...]]:
        type_order = {
            CombinationType.SET: 0,
            CombinationType.SEQUENCE: 1,
            CombinationType.BOMB: 2,
        }[self.type]
        return (
            type_order,
            self.card_count,
            self.sequence_width,
            self.rank if self.type != CombinationType.BOMB else self.bomb_rank,
            tuple(card.short_name() for card in self.cards),
        )


def validate_combination(cards: tuple[Card, ...] | list[Card]) -> Combination | None:
    candidates = possible_combinations(cards)
    return max(candidates, key=lambda combination: combination.sort_key()) if candidates else None


def possible_combinations(cards: tuple[Card, ...] | list[Card]) -> tuple[Combination, ...]:
    ordered = tuple(sorted(cards))
    if not ordered:
        return ()

    candidates: list[Combination] = []
    bomb_rank = bomb_rank_for(ordered)
    if bomb_rank:
        candidates.append(Combination(ordered, CombinationType.BOMB, rank=bomb_rank, bomb_rank=bomb_rank))

    sequence = sequence_shape_for(ordered)
    if sequence is not None:
        high_rank, width, length = sequence
        candidates.append(
            Combination(
                ordered,
                CombinationType.SEQUENCE,
                rank=high_rank,
                sequence_width=width,
                sequence_length=length,
            )
        )

    set_rank = set_rank_for(ordered)
    if set_rank is not None:
        candidates.append(Combination(ordered, CombinationType.SET, rank=set_rank))

    return tuple(candidates)


def bomb_rank_for(cards: tuple[Card, ...]) -> int:
    ranks = sorted(int(card.rank) for card in cards)

    if ranks == [Rank.JACK, Rank.QUEEN]:
        return 2
    if ranks == [Rank.JACK, Rank.KING]:
        return 3
    if ranks == [Rank.QUEEN, Rank.KING]:
        return 4
    if ranks == [Rank.JACK, Rank.QUEEN, Rank.KING]:
        return 5

    if ranks == [Rank.THREE, Rank.FIVE, Rank.SEVEN, Rank.NINE]:
        if any(card.is_wild for card in cards):
            return 0
        suits = {card.suit for card in cards}
        if len(suits) == 4:
            return 1
        if len(suits) == 1:
            return 6

    return 0


def set_rank_for(cards: tuple[Card, ...]) -> int | None:
    if not 1 <= len(cards) <= 7:
        return None

    non_wild = [card for card in cards if not card.is_wild]
    wilds = [card for card in cards if card.is_wild]

    if len(cards) == 1:
        return int(cards[0].rank)

    if not non_wild:
        # Multiple wild cards are bombs when valid, never ordinary sets.
        return None

    target_rank = int(non_wild[0].rank)
    if any(int(card.rank) != target_rank for card in non_wild):
        return None
    if any(int(wild.rank) <= target_rank for wild in wilds):
        return None

    return target_rank


def sequence_shape_for(cards: tuple[Card, ...]) -> tuple[int, int, int] | None:
    if len(cards) < 3:
        return None

    non_wild = [card for card in cards if not card.is_wild]
    wilds = [card for card in cards if card.is_wild]
    if not non_wild:
        return None

    total = len(cards)
    possible_shapes: list[tuple[int, int, int]] = []

    for width in range(1, min(4, total) + 1):
        if total % width != 0:
            continue
        length = total // width
        if width == 1 and length < 3:
            continue
        if width > 1 and length < 2:
            continue

        for start_rank in range(2, 14 - length + 1):
            ranks = tuple(range(start_rank, start_rank + length))
            for suit_pattern in combinations(SUITS, width):
                if _cards_fit_sequence(non_wild, wilds, ranks, suit_pattern):
                    possible_shapes.append((ranks[-1], width, length))

    if not possible_shapes:
        return None

    # Prefer the interpretation that makes the played combination strongest; legal
    # move generation/comparison uses this canonical rank.
    return max(possible_shapes, key=lambda shape: (shape[0], shape[1], shape[2]))


def _cards_fit_sequence(
    non_wild: list[Card],
    wilds: list[Card],
    ranks: tuple[int, ...],
    suit_pattern: tuple[Suit, ...],
) -> bool:
    required_slots = {(rank, suit) for rank in ranks for suit in suit_pattern}
    occupied: set[tuple[int, Suit]] = set()

    for card in non_wild:
        slot = (int(card.rank), card.suit)
        if slot not in required_slots or slot in occupied:
            return False
        occupied.add(slot)

    missing = tuple(sorted(required_slots - occupied))
    if len(missing) != len(wilds):
        return False

    return _wilds_can_fill(tuple(sorted(wilds, key=lambda card: int(card.rank))), missing)


def _wilds_can_fill(wilds: tuple[Card, ...], slots: tuple[tuple[int, Suit], ...]) -> bool:
    if not wilds:
        return not slots

    wild = wilds[0]
    for index, (rank, _suit) in enumerate(slots):
        if rank < int(wild.rank):
            remaining = slots[:index] + slots[index + 1 :]
            if _wilds_can_fill(wilds[1:], remaining):
                return True
    return False


def can_beat(new: Combination, previous: Combination | None) -> bool:
    if previous is None:
        return True

    if new.type == CombinationType.BOMB:
        if previous.type == CombinationType.BOMB:
            return new.bomb_rank > previous.bomb_rank
        return True

    if previous.type == CombinationType.BOMB:
        return False
    if new.type != previous.type:
        return False
    if new.card_count != previous.card_count:
        return False

    if new.type == CombinationType.SEQUENCE:
        if new.sequence_width != previous.sequence_width:
            return False
        if new.sequence_length != previous.sequence_length:
            return False

    return new.rank > previous.rank
