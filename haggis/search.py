from __future__ import annotations

from random import Random

from .cards import Card
from .combinations import CombinationType
from .engine import HaggisState, Move


def bet_amount_for_hand(hand: tuple[Card, ...], *, aggression: int) -> int:
    """Deterministic pre-play betting heuristic.

    Haggis bets are high leverage: a failed 30-point bet often decides a 350-point
    game. Keep ordinary baseline bots conservative so self-play data contains a
    useful spread of 0/15/30 decisions instead of teaching every model to always
    slam 30.
    """
    point_total = sum(card.points for card in hand)
    high_cards = sum(1 for card in hand if int(card.rank) >= 9 and not card.is_wild)
    rank_counts: dict[int, int] = {}
    for card in hand:
        if not card.is_wild:
            rank_counts[int(card.rank)] = rank_counts.get(int(card.rank), 0) + 1
    pairs = sum(1 for count in rank_counts.values() if count >= 2)
    triples = sum(1 for count in rank_counts.values() if count >= 3)

    strength = point_total * 1.2 + high_cards * 1.5 + pairs * 1.5 + triples * 3.0 + aggression * 1.0
    if strength >= 42:
        return 30
    if strength >= 36:
        return 15
    return 0


def sample_determinization(state: HaggisState, rng: Random) -> HaggisState:
    player = state.current_player
    opponent = 1 - player
    unknown_pool = list(sorted((*state.hands[opponent], *state.haggis)))
    rng.shuffle(unknown_pool)

    opponent_count = len(state.hands[opponent])
    sampled_opponent = tuple(sorted(unknown_pool[:opponent_count]))
    sampled_haggis = tuple(sorted(unknown_pool[opponent_count:]))
    hands = list(state.hands)
    hands[opponent] = sampled_opponent
    return HaggisState(
        hands=tuple(hands),
        haggis=sampled_haggis,
        captured=state.captured,
        current_player=state.current_player,
        last_combination=state.last_combination,
        last_player=state.last_player,
        trick_cards=state.trick_cards,
        bets=state.bets,
        has_played=state.has_played,
        hand_winner=state.hand_winner,
    )


def root_candidates(state: HaggisState, legal_moves: tuple[Move, ...], max_root_moves: int) -> tuple[Move, ...]:
    if len(legal_moves) <= max_root_moves:
        return legal_moves

    hand_size = len(state.hands[state.current_player])
    playable = [move for move in legal_moves if not move.is_pass]
    pass_moves = [move for move in legal_moves if move.is_pass]
    ranked = sorted(playable, key=lambda move: point_aware_key(move, hand_size))
    candidates = ranked[:max_root_moves]
    if pass_moves and len(candidates) < max_root_moves:
        candidates.append(pass_moves[0])
    return tuple(candidates)


def rollout_value(
    state: HaggisState,
    root_player: int,
    rng: Random,
    max_rollout_turns: int,
    *,
    rollout_policy: object | None = None,
) -> float:
    turns = 0
    while state.hand_winner is None and turns < max_rollout_turns:
        move = choose_rollout_move(state, rng, rollout_policy=rollout_policy)
        state = state.apply_move(move)
        turns += 1

    if state.hand_winner is not None:
        score = state.score_hand().points
        return float(score[root_player] - score[1 - root_player])
    return float(static_value(state, root_player))


def choose_rollout_move(state: HaggisState, rng: Random, *, rollout_policy: object | None = None) -> Move:
    legal_moves = state.legal_moves()
    playable = [move for move in legal_moves if not move.is_pass]
    if not playable:
        return Move.pass_turn()

    hand_size = len(state.hands[state.current_player])
    winning_moves = [move for move in playable if len(move.cards) == hand_size]
    if winning_moves:
        return min(winning_moves, key=low_commitment_key)

    if rollout_policy is not None:
        return rollout_policy.choose_move(state, tuple(playable))  # type: ignore[attr-defined]

    ranked = sorted(playable, key=lambda move: point_aware_key(move, hand_size))
    top_count = min(3, len(ranked))
    return rng.choice(ranked[:top_count])


def is_bomb(move: Move) -> bool:
    return bool(move.combination and move.combination.type == CombinationType.BOMB)


def static_value(state: HaggisState, player: int) -> int:
    opponent = 1 - player
    return (
        (len(state.hands[opponent]) - len(state.hands[player])) * 10
        + sum(card.points for card in state.captured[player])
        - sum(card.points for card in state.captured[opponent])
        + sum(card.points for card in state.hands[opponent])
        - sum(card.points for card in state.hands[player])
    )


def search_tiebreak_key(move: Move) -> tuple[int, int, int, int, tuple[str, ...]]:
    if move.is_pass or move.combination is None:
        return (-1, 0, 0, 0, ())
    return (
        len(move.cards),
        -sum(card.points for card in move.cards),
        -move.combination.bomb_rank,
        -move.combination.rank,
        tuple(card.short_name() for card in move.cards),
    )


def point_aware_key(move: Move, hand_size: int) -> tuple[int, int, int, int, int, tuple[str, ...]]:
    combination = move.combination
    assert combination is not None
    point_risk = sum(card.points for card in move.cards)
    wild_count = sum(1 for card in move.cards if card.is_wild)
    empties_hand = len(move.cards) == hand_size
    bomb_penalty = 8 if combination.type == CombinationType.BOMB else 0

    return (
        0 if empties_hand else 1,
        point_risk * 4 + wild_count * 3 + bomb_penalty,
        -len(move.cards),
        combination.bomb_rank,
        combination.rank,
        tuple(card.short_name() for card in move.cards),
    )


def low_bomb_key(move: Move) -> tuple[int, int, int, tuple[str, ...]]:
    combination = move.combination
    assert combination is not None
    return (
        combination.bomb_rank,
        sum(card.points for card in move.cards),
        len(move.cards),
        tuple(card.short_name() for card in move.cards),
    )


def low_commitment_key(move: Move) -> tuple[int, int, int, int, tuple[str, ...]]:
    combination = move.combination
    assert combination is not None
    return (
        1 if combination.type == CombinationType.BOMB else 0,
        sum(card.points for card in move.cards),
        combination.bomb_rank,
        combination.rank,
        tuple(card.short_name() for card in move.cards),
    )
