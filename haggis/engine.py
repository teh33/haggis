from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from collections import Counter

from .cards import Card, Deal, Hand, deal as deal_cards, player_wilds, point_total, standard_deck
from .combinations import Combination, CombinationType, can_beat, validate_combination


@dataclass(frozen=True)
class Move:
    cards: tuple[Card, ...] = ()
    combination: Combination | None = None

    @property
    def is_pass(self) -> bool:
        return self.combination is None

    @classmethod
    def pass_turn(cls) -> "Move":
        return cls()


@dataclass(frozen=True)
class HandScore:
    points: tuple[int, int]
    winner: int


class InvariantError(ValueError):
    """Raised when a HaggisState violates structural/card-conservation rules."""


@dataclass(frozen=True)
class InvariantReport:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise InvariantError("; ".join(self.errors))


@dataclass(frozen=True)
class HaggisState:
    hands: tuple[Hand, Hand]
    haggis: Hand = ()
    captured: tuple[Hand, Hand] = ((), ())
    current_player: int = 0
    last_combination: Combination | None = None
    last_player: int | None = None
    trick_cards: Hand = ()
    bets: tuple[int, int] = (0, 0)
    has_played: tuple[bool, bool] = (False, False)
    hand_winner: int | None = None

    @classmethod
    def new_deal(cls, seed: int | None = None, dealer: int = 1) -> "HaggisState":
        dealt = deal_cards(seed=seed)
        leader = 1 - dealer
        return cls(hands=dealt.hands, haggis=dealt.haggis, current_player=leader)

    @classmethod
    def from_deal(cls, dealt: Deal, current_player: int = 0) -> "HaggisState":
        return cls(hands=dealt.hands, haggis=dealt.haggis, current_player=current_player)

    def validate_invariants(self, *, full_deck: bool = False) -> InvariantReport:
        errors: list[str] = []

        if len(self.hands) != 2:
            errors.append("state must have exactly two player hands")
        if len(self.captured) != 2:
            errors.append("state must have exactly two captured piles")
        if len(self.bets) != 2:
            errors.append("state must have exactly two bets")
        if len(self.has_played) != 2:
            errors.append("state must have exactly two has_played flags")
        if self.current_player not in (0, 1):
            errors.append("current_player must be 0 or 1")
        if self.last_player not in (None, 0, 1):
            errors.append("last_player must be None, 0, or 1")
        if self.hand_winner not in (None, 0, 1):
            errors.append("hand_winner must be None, 0, or 1")
        if any(bet not in (0, 15, 30) for bet in self.bets):
            errors.append("bets must be 0, 15, or 30")
        if (self.last_combination is None) != (self.last_player is None):
            errors.append("last_combination and last_player must be set together")
        if self.last_combination is None and self.trick_cards:
            errors.append("trick_cards require a last_combination")
        if self.hand_winner is not None and self.hands[self.hand_winner]:
            errors.append("hand_winner must have an empty hand")
        if self.hand_winner is not None and self.trick_cards:
            errors.append("completed hands cannot have pending trick cards")

        zones = [*self.hands, self.haggis, *self.captured, self.trick_cards]
        all_cards = tuple(card for zone in zones for card in zone)
        duplicates = [card.short_name() for card, count in Counter(all_cards).items() if count > 1]
        if duplicates:
            errors.append(f"duplicate cards across state zones: {', '.join(sorted(duplicates))}")

        if full_deck:
            expected = Counter((*standard_deck(), *player_wilds(0), *player_wilds(1)))
            actual = Counter(all_cards)
            missing = sorted((card.short_name() for card, count in (expected - actual).items() for _ in range(count)))
            extra = sorted((card.short_name() for card, count in (actual - expected).items() for _ in range(count)))
            if missing:
                errors.append(f"missing cards from full deck: {', '.join(missing)}")
            if extra:
                errors.append(f"unexpected cards outside full deck: {', '.join(extra)}")

        return InvariantReport(tuple(errors))

    def assert_invariants(self, *, full_deck: bool = False) -> "HaggisState":
        self.validate_invariants(full_deck=full_deck).raise_if_invalid()
        return self

    def legal_moves(self) -> tuple[Move, ...]:
        if self.hand_winner is not None:
            return ()
        return legal_moves(self.hands[self.current_player], self.last_combination)

    def place_bet(self, player: int, amount: int) -> "HaggisState":
        if amount not in (0, 15, 30):
            raise ValueError("bet must be 0, 15, or 30")
        if self.has_played[player]:
            raise ValueError("player cannot bet after playing cards")
        bets = list(self.bets)
        bets[player] = amount
        return replace(self, bets=tuple(bets))

    def apply_move(self, move: Move) -> "HaggisState":
        if self.hand_winner is not None:
            raise ValueError("hand is already complete")

        player = self.current_player
        opponent = 1 - player

        if move.is_pass:
            if self.last_combination is None or self.last_player is None:
                raise ValueError("cannot pass when leading a trick")
            winner = self.last_player
            capture_player = 1 - winner if self.last_combination.type == CombinationType.BOMB else winner
            captured = _add_cards(self.captured, capture_player, self.trick_cards)
            return replace(
                self,
                captured=captured,
                current_player=winner,
                last_combination=None,
                last_player=None,
                trick_cards=(),
            )

        if move.combination is None:
            raise ValueError("non-pass moves need a combination")
        if not can_beat(move.combination, self.last_combination):
            raise ValueError("move cannot beat current combination")

        hand = self.hands[player]
        new_hand = _remove_cards(hand, move.cards)
        hands = list(self.hands)
        hands[player] = new_hand
        has_played = list(self.has_played)
        has_played[player] = True
        trick_cards = (*self.trick_cards, *move.cards)

        if not new_hand:
            capture_player = opponent if move.combination.type == CombinationType.BOMB else player
            captured = _add_cards(self.captured, capture_player, trick_cards)
            return replace(
                self,
                hands=tuple(hands),
                captured=captured,
                current_player=player,
                last_combination=move.combination,
                last_player=player,
                trick_cards=(),
                has_played=tuple(has_played),
                hand_winner=player,
            )

        return replace(
            self,
            hands=tuple(hands),
            current_player=opponent,
            last_combination=move.combination,
            last_player=player,
            trick_cards=trick_cards,
            has_played=tuple(has_played),
        )

    def score_hand(self, winner: int | None = None) -> HandScore:
        if winner is None:
            if self.hand_winner is None:
                raise ValueError("winner required before hand is complete")
            winner = self.hand_winner

        loser = 1 - winner
        points = [0, 0]

        points[winner] += len(self.hands[loser]) * 5
        points[0] += point_total(self.captured[0])
        points[1] += point_total(self.captured[1])
        points[winner] += point_total(self.hands[loser]) + point_total(self.haggis)

        for player, bet in enumerate(self.bets):
            if bet:
                points[player if player == winner else 1 - player] += bet

        return HandScore(points=tuple(points), winner=winner)


def legal_moves(hand: Hand, previous: Combination | None = None) -> tuple[Move, ...]:
    moves: dict[tuple, Move] = {}
    for cards in _candidate_card_sets(hand):
        combination = validate_combination(cards)
        if combination is None or not can_beat(combination, previous):
            continue
        key = (
            tuple(cards),
            combination.type,
            combination.rank,
            combination.bomb_rank,
            combination.sequence_width,
            combination.sequence_length,
        )
        moves[key] = Move(cards=tuple(cards), combination=combination)

    ordered = sorted(moves.values(), key=lambda move: move.combination.sort_key())
    if previous is not None:
        ordered.append(Move.pass_turn())
    return tuple(ordered)


def _candidate_card_sets(hand: Hand) -> tuple[tuple[Card, ...], ...]:
    """Generate plausible legal card groups without enumerating every subset.

    A full 17-card hand has 131k subsets; doing that every turn makes simulation
    unusably slow. Haggis combinations have narrow structure, so generate sets,
    sequences, and bombs directly and let `validate_combination` remain the final
    authority.
    """
    candidates: set[tuple[Card, ...]] = set()
    candidates.update(_set_candidates(hand))
    candidates.update(_bomb_candidates(hand))
    candidates.update(_sequence_candidates(hand))
    return tuple(sorted(candidates, key=lambda cards: (len(cards), tuple(card.short_name() for card in cards))))


def _set_candidates(hand: Hand) -> set[tuple[Card, ...]]:
    candidates: set[tuple[Card, ...]] = {(card,) for card in hand}
    naturals = [card for card in hand if not card.is_wild]
    wilds = [card for card in hand if card.is_wild]

    for target_rank in range(2, 14):
        same_rank = [card for card in naturals if int(card.rank) == target_rank]
        compatible_wilds = [card for card in wilds if int(card.rank) > target_rank]
        if not same_rank:
            continue

        for natural_count in range(1, len(same_rank) + 1):
            for natural_cards in combinations(same_rank, natural_count):
                max_wild_count = min(len(compatible_wilds), 7 - natural_count)
                for wild_count in range(0, max_wild_count + 1):
                    for wild_cards in combinations(compatible_wilds, wild_count):
                        candidates.add(tuple(sorted((*natural_cards, *wild_cards))))

    return candidates


def _bomb_candidates(hand: Hand) -> set[tuple[Card, ...]]:
    candidates: set[tuple[Card, ...]] = set()

    by_rank: dict[int, list[Card]] = {}
    for card in hand:
        by_rank.setdefault(int(card.rank), []).append(card)

    for ranks in ((11, 12), (11, 13), (12, 13), (11, 12, 13)):
        rank_cards = [by_rank.get(rank, []) for rank in ranks]
        if all(rank_cards):
            for chosen in _cartesian_cards(rank_cards):
                candidates.add(tuple(sorted(chosen)))

    for suit_cards in _natural_3579_bombs(hand, same_suit=True):
        candidates.add(tuple(sorted(suit_cards)))
    for suit_cards in _natural_3579_bombs(hand, same_suit=False):
        candidates.add(tuple(sorted(suit_cards)))

    return candidates


def _natural_3579_bombs(hand: Hand, *, same_suit: bool) -> list[tuple[Card, ...]]:
    natural = [card for card in hand if not card.is_wild]
    by_rank = {rank: [card for card in natural if int(card.rank) == rank] for rank in (3, 5, 7, 9)}
    if not all(by_rank.values()):
        return []

    bombs = []
    for chosen in _cartesian_cards([by_rank[3], by_rank[5], by_rank[7], by_rank[9]]):
        suit_count = len({card.suit for card in chosen})
        if (same_suit and suit_count == 1) or (not same_suit and suit_count == 4):
            bombs.append(tuple(chosen))
    return bombs


def _sequence_candidates(hand: Hand) -> set[tuple[Card, ...]]:
    candidates: set[tuple[Card, ...]] = set()
    naturals = [card for card in hand if not card.is_wild]
    wilds = [card for card in hand if card.is_wild]
    natural_by_slot = {(int(card.rank), card.suit): card for card in naturals}

    for width in range(1, 5):
        min_length = 3 if width == 1 else 2
        max_length = min(11, len(hand) // width)
        for length in range(min_length, max_length + 1):
            total_cards = width * length
            if total_cards > len(hand):
                continue
            for start_rank in range(2, 13 - length + 1):
                ranks = tuple(range(start_rank, start_rank + length))
                for suit_pattern in combinations({card.suit for card in hand}, width):
                    slots = tuple((rank, suit) for rank in ranks for suit in suit_pattern)
                    for assigned_cards in _assign_sequence_slots(natural_by_slot, wilds, slots):
                        candidates.add(tuple(sorted(assigned_cards)))

    return candidates


def _assign_sequence_slots(
    natural_by_slot: dict[tuple[int, object], Card],
    wilds: list[Card],
    slots: tuple[tuple[int, object], ...],
) -> tuple[tuple[Card, ...], ...]:
    assignments: list[tuple[Card, ...]] = []

    def backtrack(slot_index: int, remaining_wilds: tuple[Card, ...], chosen: tuple[Card, ...]) -> None:
        if slot_index == len(slots):
            assignments.append(chosen)
            return

        slot = slots[slot_index]
        natural = natural_by_slot.get(slot)
        if natural is not None:
            backtrack(slot_index + 1, remaining_wilds, (*chosen, natural))

        rank, _suit = slot
        for index, wild in enumerate(remaining_wilds):
            if rank < int(wild.rank):
                backtrack(
                    slot_index + 1,
                    remaining_wilds[:index] + remaining_wilds[index + 1 :],
                    (*chosen, wild),
                )

    backtrack(0, tuple(wilds), ())
    return tuple(assignments)


def _assign_wilds_to_slots(wilds: list[Card], slots: tuple[tuple[int, object], ...]) -> tuple[tuple[Card, ...], ...]:
    if not slots:
        return ((),)

    assignments: list[tuple[Card, ...]] = []

    def backtrack(slot_index: int, remaining_wilds: tuple[Card, ...], chosen: tuple[Card, ...]) -> None:
        if slot_index == len(slots):
            assignments.append(chosen)
            return
        rank, _suit = slots[slot_index]
        for index, wild in enumerate(remaining_wilds):
            if rank < int(wild.rank):
                backtrack(
                    slot_index + 1,
                    remaining_wilds[:index] + remaining_wilds[index + 1 :],
                    (*chosen, wild),
                )

    backtrack(0, tuple(wilds), ())
    return tuple(assignments)


def _cartesian_cards(groups: list[list[Card]]) -> tuple[tuple[Card, ...], ...]:
    if not groups:
        return ((),)
    result: list[tuple[Card, ...]] = []
    first, *rest = groups
    for card in first:
        for suffix in _cartesian_cards(rest):
            result.append((card, *suffix))
    return tuple(result)


def _remove_cards(hand: Hand, cards: tuple[Card, ...]) -> Hand:
    remaining = list(hand)
    for card in cards:
        try:
            remaining.remove(card)
        except ValueError as exc:
            raise ValueError(f"card {card} is not in current player's hand") from exc
    return tuple(remaining)


def _add_cards(captured: tuple[Hand, Hand], player: int, cards: Hand) -> tuple[Hand, Hand]:
    updated = list(captured)
    updated[player] = (*updated[player], *cards)
    return tuple(updated)
