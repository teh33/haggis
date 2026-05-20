from __future__ import annotations

from pathlib import Path
from random import Random
from math import log, sqrt

from .cards import Card
from .combinations import CombinationType
from .engine import HaggisState, Move


class RandomBot:
    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")
        return self.rng.choice(moves)


class GreedySheddingBot:
    """Simple baseline: shed the most cards, then the lowest-ranking combo."""

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=0)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        playable = [move for move in moves if not move.is_pass]
        if not playable:
            return Move.pass_turn()
        return min(
            playable,
            key=lambda move: (
                -len(move.cards),
                move.combination.bomb_rank if move.combination else 0,
                move.combination.rank if move.combination else 0,
                sum(card.points for card in move.cards),
            ),
        )


class PointAwareBot:
    """Baseline that sheds cards while avoiding unnecessary point-card donation."""

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        playable = [move for move in moves if not move.is_pass]
        if not playable:
            return Move.pass_turn()

        hand_size = len(state.hands[state.current_player])
        return min(playable, key=lambda move: _point_aware_key(move, hand_size))


class BombControlBot:
    """Conservative bomb baseline that saves bombs for threats and endgames."""

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        playable = [move for move in moves if not move.is_pass]
        if not playable:
            return Move.pass_turn()

        player = state.current_player
        opponent = 1 - player
        hand_size = len(state.hands[player])
        opponent_cards = len(state.hands[opponent])
        bombs = [move for move in playable if _is_bomb(move)]
        non_bombs = [move for move in playable if not _is_bomb(move)]

        winning_moves = [move for move in playable if len(move.cards) == hand_size]
        if winning_moves:
            return min(winning_moves, key=_low_commitment_key)

        if state.last_combination and state.last_combination.type == CombinationType.BOMB:
            if bombs:
                return min(bombs, key=_low_bomb_key)
            return Move.pass_turn()

        if non_bombs:
            return min(non_bombs, key=lambda move: _point_aware_key(move, hand_size))

        if state.last_combination is not None and opponent_cards > 5:
            return Move.pass_turn()

        return min(bombs, key=_low_bomb_key) if bombs else Move.pass_turn()


class UCBInformationSetBot:
    """Information-set root search with UCB1 allocation over legal moves."""

    def __init__(
        self,
        simulations: int = 6,
        seed: int | None = None,
        exploration: float = 1.4,
        max_rollout_turns: int = 120,
        max_root_moves: int = 4,
    ):
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        if max_rollout_turns < 1:
            raise ValueError("max_rollout_turns must be at least 1")
        if max_root_moves < 1:
            raise ValueError("max_root_moves must be at least 1")
        self.simulations = simulations
        self.rng = Random(seed)
        self.exploration = exploration
        self.max_rollout_turns = max_rollout_turns
        self.max_root_moves = max_root_moves

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=_low_commitment_key)

        root_moves, visits, values = self.search_root(state)
        scored = [
            (values[index] / visits[index], visits[index], _search_tiebreak_key(move), move)
            for index, move in enumerate(root_moves)
        ]
        return max(scored, key=lambda item: (item[0], item[1], item[2]))[3]

    def search_root(self, state: HaggisState) -> tuple[tuple[Move, ...], list[int], list[float]]:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        root_player = state.current_player
        root_moves = _root_candidates(state, legal_moves, min(self.max_root_moves, self.simulations))
        visits = [0 for _ in root_moves]
        values = [0.0 for _ in root_moves]

        for simulation_index in range(max(self.simulations, len(root_moves))):
            move_index = self._select_root_move(visits, values, simulation_index + 1)
            move = root_moves[move_index]
            determinized = self.sample_determinization(state)
            next_state = determinized.apply_move(move)
            value = _rollout_value(next_state, root_player, self.rng, self.max_rollout_turns)
            visits[move_index] += 1
            values[move_index] += value

        return root_moves, visits, values

    def _select_root_move(self, visits: list[int], values: list[float], total_visits: int) -> int:
        for index, visit_count in enumerate(visits):
            if visit_count == 0:
                return index

        log_total = log(max(total_visits, 2))
        scores = []
        for index, visit_count in enumerate(visits):
            exploitation = values[index] / visit_count
            exploration = self.exploration * sqrt(log_total / visit_count)
            scores.append((exploitation + exploration, -index, index))
        return max(scores)[2]

    def sample_determinization(self, state: HaggisState) -> HaggisState:
        player = state.current_player
        opponent = 1 - player
        unknown_pool = list(sorted((*state.hands[opponent], *state.haggis)))
        self.rng.shuffle(unknown_pool)

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


class InformationSetRolloutBot:
    """Hidden-information rollout bot using sampled determinizations.

    Root moves are legal moves from the real information set. Each simulation
    samples a plausible split of the unknown pool (opponent hand + haggis), then
    rolls out from that determinized state.
    """

    def __init__(
        self,
        simulations_per_move: int = 2,
        seed: int | None = None,
        max_rollout_turns: int = 120,
        max_root_moves: int = 8,
    ):
        if simulations_per_move < 1:
            raise ValueError("simulations_per_move must be at least 1")
        if max_rollout_turns < 1:
            raise ValueError("max_rollout_turns must be at least 1")
        if max_root_moves < 1:
            raise ValueError("max_root_moves must be at least 1")
        self.simulations_per_move = simulations_per_move
        self.rng = Random(seed)
        self.max_rollout_turns = max_rollout_turns
        self.max_root_moves = max_root_moves

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=_low_commitment_key)

        root_moves = _root_candidates(state, legal_moves, self.max_root_moves)
        scored: list[tuple[float, tuple[int, int, int, int, tuple[str, ...]], Move]] = []
        for move in root_moves:
            total = 0.0
            for _ in range(self.simulations_per_move):
                determinized = self.sample_determinization(state)
                next_state = determinized.apply_move(move)
                total += _rollout_value(next_state, player, self.rng, self.max_rollout_turns)
            scored.append((total / self.simulations_per_move, _search_tiebreak_key(move), move))

        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def sample_determinization(self, state: HaggisState) -> HaggisState:
        player = state.current_player
        opponent = 1 - player
        unknown_pool = list(sorted((*state.hands[opponent], *state.haggis)))
        self.rng.shuffle(unknown_pool)

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


class MonteCarloRolloutBot:
    """Root Monte Carlo rollout bot for imperfect-but-useful tactical search.

    It scores a small set of promising root moves by simulated playouts. Rollouts
    use a stochastic heuristic policy and terminal score margin as value.
    """

    def __init__(
        self,
        simulations_per_move: int = 2,
        seed: int | None = None,
        max_rollout_turns: int = 120,
        max_root_moves: int = 8,
    ):
        if simulations_per_move < 1:
            raise ValueError("simulations_per_move must be at least 1")
        if max_rollout_turns < 1:
            raise ValueError("max_rollout_turns must be at least 1")
        if max_root_moves < 1:
            raise ValueError("max_root_moves must be at least 1")
        self.simulations_per_move = simulations_per_move
        self.rng = Random(seed)
        self.max_rollout_turns = max_rollout_turns
        self.max_root_moves = max_root_moves

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=_low_commitment_key)

        root_moves = _root_candidates(state, legal_moves, self.max_root_moves)
        scored: list[tuple[float, tuple[int, int, int, int, tuple[str, ...]], Move]] = []
        for move in root_moves:
            total = 0.0
            for _ in range(self.simulations_per_move):
                next_state = state.apply_move(move)
                total += _rollout_value(next_state, player, self.rng, self.max_rollout_turns)
            scored.append((total / self.simulations_per_move, _search_tiebreak_key(move), move))

        return max(scored, key=lambda item: (item[0], item[1]))[2]


class PolicyBot:
    """Bot backed by a saved linear action-ranking policy model."""

    def __init__(self, model_path: str | Path = "models/linear_policy.json"):
        from .policy import LinearPolicy

        self.model_path = Path(model_path)
        self.policy = LinearPolicy.load(self.model_path)

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")
        return self.policy.choose_move(state, moves)


class EndgameSearchBot:
    """Perfect-information minimax bot for small endgame states.

    This deliberately ignores hidden information. It is a tactical endgame
    baseline and a stepping stone toward information-set search.
    """

    def __init__(self, max_cards: int = 8, max_depth: int = 40, fallback: object | None = None):
        self.max_cards = max_cards
        self.max_depth = max_depth
        self.fallback = fallback if fallback is not None else BombControlBot()

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return _bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")

        remaining_cards = sum(len(hand) for hand in state.hands)
        if remaining_cards > self.max_cards:
            return self.fallback.choose_move(state)  # type: ignore[attr-defined]

        root_player = state.current_player
        cache: dict[tuple[HaggisState, int], int] = {}

        def value(position: HaggisState, depth: int) -> int:
            cache_key = (position, depth)
            if cache_key in cache:
                return cache[cache_key]

            if position.hand_winner is not None:
                score = position.score_hand().points
                result = score[root_player] - score[1 - root_player]
                cache[cache_key] = result
                return result

            if depth <= 0:
                result = _static_value(position, root_player)
                cache[cache_key] = result
                return result

            child_values = [value(position.apply_move(move), depth - 1) for move in position.legal_moves()]
            result = max(child_values) if position.current_player == root_player else min(child_values)
            cache[cache_key] = result
            return result

        scored_moves = [
            (value(state.apply_move(move), self.max_depth - 1), _search_tiebreak_key(move), move)
            for move in moves
        ]
        return max(scored_moves, key=lambda item: (item[0], item[1]))[2]


def _bet_amount_for_hand(hand: tuple[Card, ...], *, aggression: int) -> int:
    """Simple deterministic pre-play betting heuristic."""
    wild_count = sum(1 for card in hand if card.is_wild)
    point_total = sum(card.points for card in hand)
    high_cards = sum(1 for card in hand if int(card.rank) >= 8)
    rank_counts: dict[int, int] = {}
    for card in hand:
        if not card.is_wild:
            rank_counts[int(card.rank)] = rank_counts.get(int(card.rank), 0) + 1
    max_same_rank = max(rank_counts.values(), default=0)

    strength = point_total + wild_count * 6 + high_cards * 2 + max_same_rank * 3 + aggression * 3
    if strength >= 42:
        return 30
    if strength >= 32:
        return 15
    return 0


def _root_candidates(state: HaggisState, legal_moves: tuple[Move, ...], max_root_moves: int) -> tuple[Move, ...]:
    if len(legal_moves) <= max_root_moves:
        return legal_moves

    hand_size = len(state.hands[state.current_player])
    playable = [move for move in legal_moves if not move.is_pass]
    pass_moves = [move for move in legal_moves if move.is_pass]
    ranked = sorted(playable, key=lambda move: _point_aware_key(move, hand_size))
    candidates = ranked[:max_root_moves]
    if pass_moves and len(candidates) < max_root_moves:
        candidates.append(pass_moves[0])
    return tuple(candidates)


def _rollout_value(state: HaggisState, root_player: int, rng: Random, max_rollout_turns: int) -> float:
    turns = 0
    while state.hand_winner is None and turns < max_rollout_turns:
        move = _choose_rollout_move(state, rng)
        state = state.apply_move(move)
        turns += 1

    if state.hand_winner is not None:
        score = state.score_hand().points
        return float(score[root_player] - score[1 - root_player])
    return float(_static_value(state, root_player))


def _choose_rollout_move(state: HaggisState, rng: Random) -> Move:
    legal_moves = state.legal_moves()
    playable = [move for move in legal_moves if not move.is_pass]
    if not playable:
        return Move.pass_turn()

    hand_size = len(state.hands[state.current_player])
    winning_moves = [move for move in playable if len(move.cards) == hand_size]
    if winning_moves:
        return min(winning_moves, key=_low_commitment_key)

    ranked = sorted(playable, key=lambda move: _point_aware_key(move, hand_size))
    top_count = min(3, len(ranked))
    return rng.choice(ranked[:top_count])


def _is_bomb(move: Move) -> bool:
    return bool(move.combination and move.combination.type == CombinationType.BOMB)


def _static_value(state: HaggisState, player: int) -> int:
    opponent = 1 - player
    return (
        (len(state.hands[opponent]) - len(state.hands[player])) * 10
        + sum(card.points for card in state.captured[player])
        - sum(card.points for card in state.captured[opponent])
        + sum(card.points for card in state.hands[opponent])
        - sum(card.points for card in state.hands[player])
    )


def _search_tiebreak_key(move: Move) -> tuple[int, int, int, int, tuple[str, ...]]:
    if move.is_pass or move.combination is None:
        return (-1, 0, 0, 0, ())
    return (
        len(move.cards),
        -sum(card.points for card in move.cards),
        -move.combination.bomb_rank,
        -move.combination.rank,
        tuple(card.short_name() for card in move.cards),
    )


def _point_aware_key(move: Move, hand_size: int) -> tuple[int, int, int, int, int, tuple[str, ...]]:
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


def _low_bomb_key(move: Move) -> tuple[int, int, int, tuple[str, ...]]:
    combination = move.combination
    assert combination is not None
    return (
        combination.bomb_rank,
        sum(card.points for card in move.cards),
        len(move.cards),
        tuple(card.short_name() for card in move.cards),
    )


def _low_commitment_key(move: Move) -> tuple[int, int, int, int, tuple[str, ...]]:
    combination = move.combination
    assert combination is not None
    return (
        1 if combination.type == CombinationType.BOMB else 0,
        sum(card.points for card in move.cards),
        combination.bomb_rank,
        combination.rank,
        tuple(card.short_name() for card in move.cards),
    )
