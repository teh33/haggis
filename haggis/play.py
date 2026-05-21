from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from .cards import Card
from .combinations import CombinationType
from .engine import HaggisState, Move
from .tournament import BOT_TYPES, Bot, make_bot

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


@dataclass(frozen=True)
class HumanPlayer:
    input_fn: InputFn = input
    output_fn: OutputFn = print

    def choose_bet(self, state: HaggisState, player: int) -> int:
        self.output_fn(f"Your hand: {format_cards(state.hands[player])}")
        return prompt_bet(self.input_fn, self.output_fn)

    def choose_move(self, state: HaggisState) -> Move:
        return prompt_move(state, self.input_fn, self.output_fn)


@dataclass(frozen=True)
class PlayerCpuResult:
    winner: int
    score: tuple[int, int]
    turns: int
    bets: tuple[int, int]


def prompt_bet(input_fn: InputFn = input, output_fn: OutputFn = print) -> int:
    while True:
        raw = input_fn("Bet? [0/15/30, default 0]: ").strip()
        if raw == "":
            return 0
        try:
            amount = int(raw)
        except ValueError:
            output_fn("Enter 0, 15, or 30.")
            continue
        if amount in (0, 15, 30):
            return amount
        output_fn("Enter 0, 15, or 30.")


def prompt_move(state: HaggisState, input_fn: InputFn = input, output_fn: OutputFn = print) -> Move:
    legal_moves = state.legal_moves()
    if not legal_moves:
        raise ValueError("no legal moves available")

    output_fn(format_state_for_human(state))
    output_fn("Legal moves:")
    for index, move in enumerate(legal_moves, 1):
        output_fn(f"  {index:>2}. {format_move(move)}")

    while True:
        raw = input_fn("Choose move number, cards, or q to quit: ").strip()
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise KeyboardInterrupt("player quit")
        if lowered in {"p", "pass"}:
            pass_move = next((move for move in legal_moves if move.is_pass), None)
            if pass_move is not None:
                return pass_move
            output_fn("Pass is not legal right now.")
            continue
        try:
            selected = int(raw)
        except ValueError:
            selected_move = _move_by_card_names(raw, legal_moves)
            if selected_move is not None:
                return selected_move
            output_fn("Enter a legal move number, exact card names like '3♣ 3♦', 'pass', or 'q'.")
            continue
        if 1 <= selected <= len(legal_moves):
            return legal_moves[selected - 1]
        output_fn(f"Enter a number from 1 to {len(legal_moves)}.")


def play_player_vs_cpu(
    *,
    cpu_bot: str = "policy-rollout",
    human_player: int = 0,
    seed: int = 1,
    max_turns: int = 500,
    policy_model: str | None = None,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
    enable_betting: bool = True,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> PlayerCpuResult:
    if human_player not in (0, 1):
        raise ValueError("human_player must be 0 or 1")

    cpu_player = 1 - human_player
    human = HumanPlayer(input_fn=input_fn, output_fn=output_fn)
    cpu = make_bot(
        cpu_bot,
        seed=seed * 2 + cpu_player,
        policy_model=policy_model,
        search_simulations=search_simulations,
        search_root_moves=search_root_moves,
        search_rollout_turns=search_rollout_turns,
    )
    players: list[Bot] = [cpu, cpu]
    players[human_player] = human

    state = HaggisState.new_deal(seed=seed, dealer=1).assert_invariants(full_deck=True)
    if enable_betting:
        for player, bot in enumerate(players):
            chooser = getattr(bot, "choose_bet", None)
            amount = chooser(state, player) if chooser is not None else 0
            state = state.place_bet(player, amount)

    turns = 0
    while state.hand_winner is None:
        if turns >= max_turns:
            raise RuntimeError(f"hand exceeded {max_turns} turns")
        player = state.current_player
        output_fn(f"\nTurn {turns + 1}: {'You' if player == human_player else 'CPU'} to act")
        move = players[player].choose_move(state)
        if move not in state.legal_moves():
            raise ValueError(f"player {player} chose an illegal move: {move}")
        output_fn(f"{'You' if player == human_player else 'CPU'} played: {format_move(move)}")
        state = state.apply_move(move).assert_invariants(full_deck=True)
        turns += 1

    score = state.score_hand()
    output_fn("\nHand complete.")
    output_fn(f"Winner: {'You' if score.winner == human_player else 'CPU'}")
    output_fn(f"Score: You {score.points[human_player]} - CPU {score.points[cpu_player]}")
    return PlayerCpuResult(winner=score.winner, score=score.points, turns=turns, bets=state.bets)


def format_state_for_human(state: HaggisState) -> str:
    player = state.current_player
    opponent = 1 - player
    lines = [
        f"Your hand ({len(state.hands[player])}): {format_cards(state.hands[player])}",
        f"Opponent cards: {len(state.hands[opponent])}",
        f"Captured points: you {sum(card.points for card in state.captured[player])}, opponent {sum(card.points for card in state.captured[opponent])}",
        f"Trick points: {sum(card.points for card in state.trick_cards)}",
    ]
    if state.last_combination is None:
        lines.append("You are leading the trick.")
    else:
        assert state.last_player is not None
        lines.append(f"Current trick: player {state.last_player} led {format_combination(state.last_combination)}")
    return "\n".join(lines)


def format_cards(cards: tuple[Card, ...]) -> str:
    return " ".join(card.short_name() for card in cards) if cards else "—"


def format_move(move: Move) -> str:
    if move.is_pass:
        return "pass"
    assert move.combination is not None
    return f"{format_cards(move.cards)} ({format_combination(move.combination)}, {sum(card.points for card in move.cards)} pts)"


def format_combination(combination) -> str:
    if combination.type == CombinationType.BOMB:
        return f"bomb rank {combination.bomb_rank}"
    return f"{combination.type.value} rank {combination.rank}"


def _move_by_card_names(raw: str, legal_moves: tuple[Move, ...]) -> Move | None:
    requested = sorted(token.strip().lower() for token in raw.replace(",", " ").split() if token.strip())
    if not requested:
        return None
    for move in legal_moves:
        names = sorted(card.short_name().lower() for card in move.cards)
        ascii_names = sorted(_ascii_card_name(card).lower() for card in move.cards)
        if requested in (names, ascii_names):
            return move
    return None


def _ascii_card_name(card: Card) -> str:
    suffix = "*" if card.is_wild else ""
    return f"{card.rank.label}{card.suit.value}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play an interactive Haggis hand against a CPU bot")
    parser.add_argument("--cpu", choices=sorted(BOT_TYPES), default="policy-rollout", help="CPU bot to play against")
    parser.add_argument("--human-player", type=int, choices=(0, 1), default=0, help="Seat for the human player")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--no-betting", action="store_true")
    parser.add_argument("--policy-model", default="models/linear_policy.json")
    parser.add_argument("--search-simulations", type=int)
    parser.add_argument("--search-root-moves", type=int, default=4)
    parser.add_argument("--search-rollout-turns", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        play_player_vs_cpu(
            cpu_bot=args.cpu,
            human_player=args.human_player,
            seed=args.seed,
            max_turns=args.max_turns,
            policy_model=args.policy_model,
            search_simulations=args.search_simulations,
            search_root_moves=args.search_root_moves,
            search_rollout_turns=args.search_rollout_turns,
            enable_betting=not args.no_betting,
        )
    except KeyboardInterrupt:
        print("\nGame cancelled.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
