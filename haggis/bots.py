from __future__ import annotations

from pathlib import Path
from random import Random
from math import log, sqrt

from .combinations import CombinationType
from .engine import HaggisState, Move
from .search import (
    bet_amount_for_hand,
    is_bomb,
    low_bomb_key,
    low_commitment_key,
    point_aware_key,
    root_candidates,
    rollout_value,
    sample_determinization,
    search_tiebreak_key,
    static_value,
)


class RandomBot:
    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")
        return self.rng.choice(moves)


class GreedySheddingBot:
    """Simple baseline: shed the most cards, then the lowest-ranking combo."""

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return bet_amount_for_hand(state.hands[player], aggression=0)

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
        return bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        playable = [move for move in moves if not move.is_pass]
        if not playable:
            return Move.pass_turn()

        hand_size = len(state.hands[state.current_player])
        return min(playable, key=lambda move: point_aware_key(move, hand_size))


class BombControlBot:
    """Conservative bomb baseline that saves bombs for threats and endgames."""

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        playable = [move for move in moves if not move.is_pass]
        if not playable:
            return Move.pass_turn()

        player = state.current_player
        opponent = 1 - player
        hand_size = len(state.hands[player])
        opponent_cards = len(state.hands[opponent])
        bombs = [move for move in playable if is_bomb(move)]
        non_bombs = [move for move in playable if not is_bomb(move)]

        winning_moves = [move for move in playable if len(move.cards) == hand_size]
        if winning_moves:
            return min(winning_moves, key=low_commitment_key)

        if state.last_combination and state.last_combination.type == CombinationType.BOMB:
            if bombs:
                return min(bombs, key=low_bomb_key)
            return Move.pass_turn()

        if non_bombs:
            return min(non_bombs, key=lambda move: point_aware_key(move, hand_size))

        if state.last_combination is not None and opponent_cards > 5:
            return Move.pass_turn()

        return min(bombs, key=low_bomb_key) if bombs else Move.pass_turn()


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
        return bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=low_commitment_key)

        root_moves, visits, values = self.search_root(state)
        scored = [
            (values[index] / visits[index], visits[index], search_tiebreak_key(move), move)
            for index, move in enumerate(root_moves)
        ]
        return max(scored, key=lambda item: (item[0], item[1], item[2]))[3]

    def search_root(self, state: HaggisState) -> tuple[tuple[Move, ...], list[int], list[float]]:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        root_player = state.current_player
        root_moves = root_candidates(state, legal_moves, min(self.max_root_moves, self.simulations))
        visits = [0 for _ in root_moves]
        values = [0.0 for _ in root_moves]

        for simulation_index in range(max(self.simulations, len(root_moves))):
            move_index = self._select_root_move(visits, values, simulation_index + 1)
            move = root_moves[move_index]
            determinized = sample_determinization(state, self.rng)
            next_state = determinized.apply_move(move)
            value = rollout_value(next_state, root_player, self.rng, self.max_rollout_turns)
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
        return sample_determinization(state, self.rng)


class TreeInformationSetBot:
    """Shallow information-set tree search over sampled determinizations."""

    def __init__(
        self,
        simulations: int = 8,
        seed: int | None = None,
        exploration: float = 1.4,
        max_depth: int = 2,
        max_rollout_turns: int = 80,
        max_root_moves: int = 4,
        max_child_moves: int = 4,
        model_path: str | Path | None = None,
    ):
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if max_rollout_turns < 1:
            raise ValueError("max_rollout_turns must be at least 1")
        if max_root_moves < 1:
            raise ValueError("max_root_moves must be at least 1")
        if max_child_moves < 1:
            raise ValueError("max_child_moves must be at least 1")
        self.simulations = simulations
        self.rng = Random(seed)
        self.exploration = exploration
        self.max_depth = max_depth
        self.max_rollout_turns = max_rollout_turns
        self.max_root_moves = max_root_moves
        self.max_child_moves = max_child_moves
        self.policy = None
        if model_path is not None:
            from .policy import LinearPolicy

            self.policy = LinearPolicy.load(model_path)

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=low_commitment_key)

        root_moves, visits, values = self.search_root(state)
        scored = [
            (values[index] / visits[index], visits[index], search_tiebreak_key(move), move)
            for index, move in enumerate(root_moves)
        ]
        return max(scored, key=lambda item: (item[0], item[1], item[2]))[3]

    def search_root(self, state: HaggisState) -> tuple[tuple[Move, ...], list[int], list[float]]:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        root_player = state.current_player
        root_moves = root_candidates(state, legal_moves, min(self.max_root_moves, self.simulations))
        visits = [0 for _ in root_moves]
        values = [0.0 for _ in root_moves]
        children: list[_TreeSearchNode] = [_TreeSearchNode() for _ in root_moves]

        for simulation_index in range(max(self.simulations, len(root_moves))):
            move_index = self._select_index(visits, values, simulation_index + 1)
            move = root_moves[move_index]
            determinized = sample_determinization(state, self.rng)
            next_state = determinized.apply_move(move)
            value = self._tree_value(next_state, root_player, depth=self.max_depth - 1, node=children[move_index])
            visits[move_index] += 1
            values[move_index] += value

        return root_moves, visits, values

    def _tree_value(self, state: HaggisState, root_player: int, *, depth: int, node: _TreeSearchNode) -> float:
        if state.hand_winner is not None:
            score = state.score_hand().points
            return float(score[root_player] - score[1 - root_player])
        if depth <= 0:
            return rollout_value(state, root_player, self.rng, self.max_rollout_turns, rollout_policy=self.policy)

        legal_moves = root_candidates(state, state.legal_moves(), self.max_child_moves)
        if not legal_moves:
            return float(static_value(state, root_player))
        for move in legal_moves:
            node.children.setdefault(move, _TreeSearchNode())

        move = self._select_tree_move(node, legal_moves, state.current_player == root_player)
        child = node.children[move]
        child_state = state.apply_move(move)
        value = self._tree_value(child_state, root_player, depth=depth - 1, node=child)
        child.visits += 1
        child.value_sum += value
        return value

    def _select_tree_move(self, node: _TreeSearchNode, legal_moves: tuple[Move, ...], maximizing: bool) -> Move:
        total_visits = sum(child.visits for child in node.children.values()) + 1
        return max(
            legal_moves,
            key=lambda candidate: self._tree_score(node.children[candidate], total_visits, maximizing, candidate),
        )

    def _tree_score(self, node: _TreeSearchNode, total_visits: int, maximizing: bool, move: Move) -> tuple[float, tuple[int, int, int, int, tuple[str, ...]]]:
        if node.visits == 0:
            mean = float("inf") if maximizing else float("-inf")
        else:
            mean = node.value_sum / node.visits
            if not maximizing:
                mean = -mean
            mean += self.exploration * sqrt(log(max(total_visits, 2)) / node.visits)
        return (mean, search_tiebreak_key(move))

    def _select_index(self, visits: list[int], values: list[float], total_visits: int) -> int:
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


class _TreeSearchNode:
    def __init__(self) -> None:
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[Move, _TreeSearchNode] = {}


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
        return bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=low_commitment_key)

        root_moves = root_candidates(state, legal_moves, self.max_root_moves)
        scored: list[tuple[float, tuple[int, int, int, int, tuple[str, ...]], Move]] = []
        for move in root_moves:
            total = 0.0
            for _ in range(self.simulations_per_move):
                determinized = sample_determinization(state, self.rng)
                next_state = determinized.apply_move(move)
                total += rollout_value(next_state, player, self.rng, self.max_rollout_turns)
            scored.append((total / self.simulations_per_move, search_tiebreak_key(move), move))

        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def sample_determinization(self, state: HaggisState) -> HaggisState:
        return sample_determinization(state, self.rng)


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
        return bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=low_commitment_key)

        root_moves = root_candidates(state, legal_moves, self.max_root_moves)
        scored: list[tuple[float, tuple[int, int, int, int, tuple[str, ...]], Move]] = []
        for move in root_moves:
            total = 0.0
            for _ in range(self.simulations_per_move):
                next_state = state.apply_move(move)
                total += rollout_value(next_state, player, self.rng, self.max_rollout_turns)
            scored.append((total / self.simulations_per_move, search_tiebreak_key(move), move))

        return max(scored, key=lambda item: (item[0], item[1]))[2]


class PolicyRolloutBot:
    """Search bot that uses a trained linear policy for rollout move selection.

    Policy-guided rollouts are deterministic, so each root candidate needs only
    one playout regardless of the configured simulation budget.
    """

    def __init__(
        self,
        model_path: str | Path = "models/linear_policy.json",
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
        from .policy import LinearPolicy

        self.model_path = Path(model_path)
        self.policy = LinearPolicy.load(self.model_path)
        self.simulations_per_move = simulations_per_move
        self.rng = Random(seed)
        self.max_rollout_turns = max_rollout_turns
        self.max_root_moves = max_root_moves

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return bet_amount_for_hand(state.hands[player], aggression=2)

    def choose_move(self, state: HaggisState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("no legal moves available")

        player = state.current_player
        playable = [move for move in legal_moves if not move.is_pass]
        immediate_wins = [move for move in playable if len(move.cards) == len(state.hands[player])]
        if immediate_wins:
            return min(immediate_wins, key=low_commitment_key)

        root_moves = root_candidates(state, legal_moves, self.max_root_moves)
        scored: list[tuple[float, tuple[int, int, int, int, tuple[str, ...]], Move]] = []
        for move in root_moves:
            next_state = state.apply_move(move)
            value = rollout_value(
                next_state,
                player,
                self.rng,
                self.max_rollout_turns,
                rollout_policy=self.policy,
            )
            scored.append((value, search_tiebreak_key(move), move))

        return max(scored, key=lambda item: (item[0], item[1]))[2]


class PolicyBot:
    """Bot backed by a saved linear action-ranking policy model."""

    def __init__(self, model_path: str | Path = "models/linear_policy.json"):
        from .policy import LinearPolicy

        self.model_path = Path(model_path)
        self.policy = LinearPolicy.load(self.model_path)

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return bet_amount_for_hand(state.hands[player], aggression=1)

    def choose_move(self, state: HaggisState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")
        return self.policy.choose_move(state, moves)


class TorchPolicyBot:
    """Bot backed by a PyTorch action-ranking policy model."""

    def __init__(self, model_path: str | Path = "models/torch_policy.pt"):
        from .torch_policy import load_torch_policy

        self.model_path = Path(model_path)
        self.policy = load_torch_policy(self.model_path)

    def choose_bet(self, state: HaggisState, player: int) -> int:
        return bet_amount_for_hand(state.hands[player], aggression=1)

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
        return bet_amount_for_hand(state.hands[player], aggression=2)

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
                result = static_value(position, root_player)
                cache[cache_key] = result
                return result

            child_values = [value(position.apply_move(move), depth - 1) for move in position.legal_moves()]
            result = max(child_values) if position.current_player == root_player else min(child_values)
            cache[cache_key] = result
            return result

        scored_moves = [
            (value(state.apply_move(move), self.max_depth - 1), search_tiebreak_key(move), move)
            for move in moves
        ]
        return max(scored_moves, key=lambda item: (item[0], item[1]))[2]
