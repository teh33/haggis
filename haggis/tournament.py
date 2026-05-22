from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .bots import BombControlBot, EndgameSearchBot, GreedySheddingBot, InformationSetRolloutBot, MonteCarloRolloutBot, PointAwareBot, PolicyBot, PolicyRolloutBot, RandomBot, TorchPolicyBot, TreeInformationSetBot, UCBInformationSetBot
from .combinations import CombinationType
from .engine import HaggisState, Move


class Bot(Protocol):
    def choose_move(self, state: HaggisState) -> Move: ...


@dataclass(frozen=True)
class BetStats:
    placed: tuple[int, int]
    succeeded: tuple[int, int]
    failed: tuple[int, int]


@dataclass(frozen=True)
class HandResult:
    winner: int
    score: tuple[int, int]
    turns: int
    passes: int
    bombs: int
    cards_remaining: tuple[int, int]
    bets: tuple[int, int] = (0, 0)

    @property
    def bet_stats(self) -> BetStats:
        placed = tuple(1 if bet else 0 for bet in self.bets)
        succeeded = tuple(1 if bet and player == self.winner else 0 for player, bet in enumerate(self.bets))
        failed = tuple(1 if bet and player != self.winner else 0 for player, bet in enumerate(self.bets))
        return BetStats(placed=placed, succeeded=succeeded, failed=failed)


@dataclass(frozen=True)
class GameHandRecord:
    hand: HandResult
    dealer: int
    seed: int
    cumulative_score: tuple[int, int]


@dataclass(frozen=True)
class GameResult:
    bot_names: tuple[str, str]
    target_score: int
    hand_records: tuple[GameHandRecord, ...]

    @property
    def hands(self) -> tuple[HandResult, ...]:
        return tuple(record.hand for record in self.hand_records)

    @property
    def winner(self) -> int:
        scores = self.total_score
        if scores[0] == scores[1]:
            raise ValueError("game has no winner because scores are tied")
        return 0 if scores[0] > scores[1] else 1

    @property
    def total_score(self) -> tuple[int, int]:
        if not self.hand_records:
            return (0, 0)
        return self.hand_records[-1].cumulative_score

    @property
    def hand_wins(self) -> tuple[int, int]:
        return (
            sum(1 for hand in self.hands if hand.winner == 0),
            sum(1 for hand in self.hands if hand.winner == 1),
        )

    @property
    def score_margin(self) -> int:
        scores = self.total_score
        return scores[0] - scores[1]


@dataclass(frozen=True)
class MatchResult:
    bot_names: tuple[str, str]
    hands: tuple[HandResult, ...]

    @property
    def hand_wins(self) -> tuple[int, int]:
        return (
            sum(1 for hand in self.hands if hand.winner == 0),
            sum(1 for hand in self.hands if hand.winner == 1),
        )

    @property
    def total_score(self) -> tuple[int, int]:
        return (
            sum(hand.score[0] for hand in self.hands),
            sum(hand.score[1] for hand in self.hands),
        )

    @property
    def score_margin(self) -> int:
        scores = self.total_score
        return scores[0] - scores[1]

    @property
    def total_turns(self) -> int:
        return sum(hand.turns for hand in self.hands)

    @property
    def total_passes(self) -> int:
        return sum(hand.passes for hand in self.hands)

    @property
    def total_bombs(self) -> int:
        return sum(hand.bombs for hand in self.hands)

    @property
    def total_bets_placed(self) -> tuple[int, int]:
        return _sum_pair(hand.bet_stats.placed for hand in self.hands)

    @property
    def total_bets_succeeded(self) -> tuple[int, int]:
        return _sum_pair(hand.bet_stats.succeeded for hand in self.hands)

    @property
    def total_bets_failed(self) -> tuple[int, int]:
        return _sum_pair(hand.bet_stats.failed for hand in self.hands)

    @property
    def average_turns(self) -> float:
        return self.total_turns / len(self.hands) if self.hands else 0.0


BOT_TYPES = {
    "bomb-control": BombControlBot,
    "endgame-search": EndgameSearchBot,
    "greedy": GreedySheddingBot,
    "information-set": InformationSetRolloutBot,
    "monte-carlo": MonteCarloRolloutBot,
    "point-aware": PointAwareBot,
    "policy": PolicyBot,
    "policy-rollout": PolicyRolloutBot,
    "random": RandomBot,
    "torch-policy": TorchPolicyBot,
    "tree-information-set": TreeInformationSetBot,
    "ucb-information-set": UCBInformationSetBot,
}


def make_bot(
    name: str,
    seed: int | None = None,
    policy_model: str | None = None,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
    torch_bet_model: str | None = None,
    bot_a_bet_model: str | None = None,
    bot_b_bet_model: str | None = None,
    enable_betting: bool = True,
) -> Bot:
    try:
        bot_type = BOT_TYPES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BOT_TYPES))
        raise ValueError(f"unknown bot {name!r}; expected one of: {choices}") from exc

    if bot_type is RandomBot:
        return RandomBot(seed=seed)
    if bot_type is MonteCarloRolloutBot:
        return MonteCarloRolloutBot(
            seed=seed,
            simulations_per_move=search_simulations or 2,
            max_root_moves=search_root_moves or 8,
            max_rollout_turns=search_rollout_turns or 120,
        )
    if bot_type is InformationSetRolloutBot:
        return InformationSetRolloutBot(
            seed=seed,
            simulations_per_move=search_simulations or 2,
            max_root_moves=search_root_moves or 8,
            max_rollout_turns=search_rollout_turns or 120,
        )
    if bot_type is UCBInformationSetBot:
        return UCBInformationSetBot(
            seed=seed,
            simulations=search_simulations or 2,
            max_root_moves=search_root_moves or 3,
            max_rollout_turns=search_rollout_turns or 80,
        )
    if bot_type is TreeInformationSetBot:
        return TreeInformationSetBot(
            seed=seed,
            simulations=search_simulations or 4,
            max_root_moves=search_root_moves or 4,
            max_rollout_turns=search_rollout_turns or 80,
            model_path=policy_model,
        )
    if bot_type is PolicyBot:
        return PolicyBot(model_path=policy_model or "models/linear_policy.json")
    if bot_type is TorchPolicyBot:
        if not policy_model:
            raise ValueError("torch-policy requires --policy-model")
        return TorchPolicyBot(model_path=policy_model, bet_model_path=torch_bet_model)
    if bot_type is PolicyRolloutBot:
        return PolicyRolloutBot(
            model_path=policy_model or "models/linear_policy.json",
            seed=seed,
            simulations_per_move=search_simulations or 2,
            max_root_moves=search_root_moves or 8,
            max_rollout_turns=search_rollout_turns or 120,
        )
    return bot_type()


def play_hand(
    bots: tuple[Bot, Bot],
    *,
    seed: int | None = None,
    dealer: int = 1,
    max_turns: int = 500,
    enable_betting: bool = True,
) -> HandResult:
    state = HaggisState.new_deal(seed=seed, dealer=dealer).assert_invariants(full_deck=True)
    turns = 0
    passes = 0
    bombs = 0

    while state.hand_winner is None:
        if turns >= max_turns:
            raise RuntimeError(f"hand exceeded {max_turns} turns")

        player = state.current_player
        if enable_betting and not state.has_played[player]:
            state = _place_player_bet(state, bots[player], player).assert_invariants(full_deck=True)

        legal = set(state.legal_moves())
        move = bots[player].choose_move(state)
        if move not in legal:
            raise ValueError(f"bot {player} chose an illegal move: {move}")

        turns += 1
        if move.is_pass:
            passes += 1
        elif move.combination and move.combination.type == CombinationType.BOMB:
            bombs += 1

        state = state.apply_move(move).assert_invariants(full_deck=True)

    score = state.score_hand()
    return HandResult(
        winner=score.winner,
        score=score.points,
        turns=turns,
        passes=passes,
        bombs=bombs,
        cards_remaining=(len(state.hands[0]), len(state.hands[1])),
        bets=state.bets,
    )


def run_game(
    bot_a: str,
    bot_b: str,
    *,
    target_score: int = 350,
    seed: int = 1,
    max_hands: int = 100,
    max_turns: int = 500,
    policy_model: str | None = None,
    bot_a_policy_model: str | None = None,
    bot_b_policy_model: str | None = None,
    torch_policy_model: str | None = None,
    torch_bet_model: str | None = None,
    bot_a_bet_model: str | None = None,
    bot_b_bet_model: str | None = None,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
    enable_betting: bool = True,
) -> GameResult:
    if target_score < 1:
        raise ValueError("target_score must be at least 1")
    if max_hands < 1:
        raise ValueError("max_hands must be at least 1")

    bots = (
        make_bot(
            bot_a,
            seed=seed * 2 + 1,
            policy_model=bot_a_policy_model or _model_for_bot_name(bot_a, policy_model=policy_model, torch_policy_model=torch_policy_model),
            torch_bet_model=bot_a_bet_model or torch_bet_model,
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        ),
        make_bot(
            bot_b,
            seed=seed * 2 + 2,
            policy_model=bot_b_policy_model or _model_for_bot_name(bot_b, policy_model=policy_model, torch_policy_model=torch_policy_model),
            torch_bet_model=bot_b_bet_model or torch_bet_model,
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        ),
    )
    # Per published two-player rules, the trailing player leads each next hand;
    # if tied, the previous hand winner leads. `dealer` is the non-leader.
    cumulative = [0, 0]
    dealer = 1
    records: list[GameHandRecord] = []

    for hand_index in range(max_hands):
        hand_seed = seed + hand_index
        hand = play_hand(bots, seed=hand_seed, dealer=dealer, max_turns=max_turns, enable_betting=enable_betting)
        cumulative[0] += hand.score[0]
        cumulative[1] += hand.score[1]
        records.append(
            GameHandRecord(
                hand=hand,
                dealer=dealer,
                seed=hand_seed,
                cumulative_score=tuple(cumulative),
            )
        )

        if max(cumulative) >= target_score and cumulative[0] != cumulative[1]:
            return GameResult(bot_names=(bot_a, bot_b), target_score=target_score, hand_records=tuple(records))

        dealer = _next_dealer(tuple(cumulative), last_hand_winner=hand.winner)

    raise RuntimeError(f"game did not finish within {max_hands} hands")


def _next_dealer(cumulative_score: tuple[int, int], *, last_hand_winner: int) -> int:
    if cumulative_score[0] > cumulative_score[1]:
        return 0
    if cumulative_score[1] > cumulative_score[0]:
        return 1
    return 1 - last_hand_winner


def format_game_summary(result: GameResult) -> str:
    scores = result.total_score
    wins = result.hand_wins
    return "\n".join(
        (
            f"Haggis game: {result.bot_names[0]} vs {result.bot_names[1]}",
            f"Target score: {result.target_score}",
            f"Hands: {len(result.hands)}",
            f"Winner: {result.bot_names[result.winner]} (player {result.winner})",
            f"Hand wins: {wins[0]} - {wins[1]}",
            f"Final score: {scores[0]} - {scores[1]} (margin {result.score_margin:+d})",
        )
    )


def run_match(
    bot_a: str,
    bot_b: str,
    *,
    hands: int,
    seed: int = 1,
    max_turns: int = 500,
    policy_model: str | None = None,
    bot_a_policy_model: str | None = None,
    bot_b_policy_model: str | None = None,
    torch_policy_model: str | None = None,
    torch_bet_model: str | None = None,
    bot_a_bet_model: str | None = None,
    bot_b_bet_model: str | None = None,
    enable_betting: bool = True,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
) -> MatchResult:
    if hands < 1:
        raise ValueError("hands must be at least 1")

    bots = (
        make_bot(
            bot_a,
            seed=seed * 2 + 1,
            policy_model=bot_a_policy_model or _model_for_bot_name(bot_a, policy_model=policy_model, torch_policy_model=torch_policy_model),
            torch_bet_model=bot_a_bet_model or torch_bet_model,
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        ),
        make_bot(
            bot_b,
            seed=seed * 2 + 2,
            policy_model=bot_b_policy_model or _model_for_bot_name(bot_b, policy_model=policy_model, torch_policy_model=torch_policy_model),
            torch_bet_model=bot_b_bet_model or torch_bet_model,
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        ),
    )
    results = []

    for hand_index in range(hands):
        dealer = hand_index % 2
        hand_seed = seed + hand_index
        results.append(play_hand(bots, seed=hand_seed, dealer=dealer, max_turns=max_turns, enable_betting=enable_betting))

    return MatchResult(bot_names=(bot_a, bot_b), hands=tuple(results))


def format_summary(result: MatchResult) -> str:
    scores = result.total_score
    wins = result.hand_wins
    hand_count = len(result.hands)
    return "\n".join(
        (
            f"Haggis tournament: {result.bot_names[0]} vs {result.bot_names[1]}",
            f"Hands: {hand_count}",
            f"Hand wins: {wins[0]} - {wins[1]}",
            f"Score: {scores[0]} - {scores[1]} (margin {result.score_margin:+d})",
            f"Average turns/hand: {result.average_turns:.2f}",
            f"Passes: {result.total_passes}",
            f"Bombs played: {result.total_bombs}",
            f"Bets placed: {result.total_bets_placed[0]} - {result.total_bets_placed[1]}",
            f"Bets succeeded: {result.total_bets_succeeded[0]} - {result.total_bets_succeeded[1]}",
            f"Bets failed: {result.total_bets_failed[0]} - {result.total_bets_failed[1]}",
        )
    )


def tournament_to_metrics(result: MatchResult, *, config: dict | None = None) -> dict:
    return {
        "type": "tournament",
        "config": config or {},
        "bot_names": list(result.bot_names),
        "hands": len(result.hands),
        "hand_wins": list(result.hand_wins),
        "score": list(result.total_score),
        "score_margin": result.score_margin,
        "total_turns": result.total_turns,
        "average_turns": result.average_turns,
        "passes": result.total_passes,
        "bombs": result.total_bombs,
        "bets_placed": list(result.total_bets_placed),
        "bets_succeeded": list(result.total_bets_succeeded),
        "bets_failed": list(result.total_bets_failed),
        "hand_records": [_hand_to_dict(hand) for hand in result.hands],
    }


def game_to_metrics(result: GameResult, *, config: dict | None = None) -> dict:
    return {
        "type": "game",
        "config": config or {},
        "bot_names": list(result.bot_names),
        "target_score": result.target_score,
        "winner": result.winner,
        "hands": len(result.hands),
        "hand_wins": list(result.hand_wins),
        "score": list(result.total_score),
        "score_margin": result.score_margin,
        "hand_records": [
            {
                "dealer": record.dealer,
                "seed": record.seed,
                "cumulative_score": list(record.cumulative_score),
                "hand": _hand_to_dict(record.hand),
            }
            for record in result.hand_records
        ],
    }


def _model_for_bot_name(bot_name: str, *, policy_model: str | None, torch_policy_model: str | None) -> str | None:
    if bot_name == "torch-policy":
        return torch_policy_model or policy_model
    return policy_model


def write_metrics(metrics: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hand_to_dict(hand: HandResult) -> dict:
    return {
        "winner": hand.winner,
        "score": list(hand.score),
        "turns": hand.turns,
        "passes": hand.passes,
        "bombs": hand.bombs,
        "cards_remaining": list(hand.cards_remaining),
        "bets": list(hand.bets),
        "bets_placed": list(hand.bet_stats.placed),
        "bets_succeeded": list(hand.bet_stats.succeeded),
        "bets_failed": list(hand.bet_stats.failed),
    }


def _run_config(args: argparse.Namespace, *, mode: str) -> dict:
    return {
        "mode": mode,
        "bot_a": args.bot_a,
        "bot_b": args.bot_b,
        "hands": args.hands,
        "target_score": args.target_score,
        "max_hands": args.max_hands,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "betting": not args.no_betting,
        "policy_model": args.policy_model,
        "torch_policy_model": args.torch_policy_model,
        "search_simulations": args.search_simulations,
        "search_root_moves": args.search_root_moves,
        "search_rollout_turns": args.search_rollout_turns,
    }


def _place_player_bet(state: HaggisState, bot: Bot, player: int) -> HaggisState:
    chooser = getattr(bot, "choose_bet", None)
    amount = chooser(state, player) if chooser is not None else 0
    return state.place_bet(player, amount)


def _place_initial_bets(state: HaggisState, bots: tuple[Bot, Bot]) -> HaggisState:
    for player, bot in enumerate(bots):
        state = _place_player_bet(state, bot, player)
    return state


def _sum_pair(pairs) -> tuple[int, int]:
    total0 = 0
    total1 = 0
    for pair in pairs:
        total0 += pair[0]
        total1 += pair[1]
    return (total0, total1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Haggis bot-vs-bot tournaments")
    parser.add_argument("--bot-a", choices=sorted(BOT_TYPES), default="greedy")
    parser.add_argument("--bot-b", choices=sorted(BOT_TYPES), default="random")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--target-score", type=int, help="Play an official target-score game instead of a fixed-hand tournament (default game target: 350)")
    parser.add_argument("--max-hands", type=int, default=100, help="Maximum hands for target-score games")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--no-betting", action="store_true", help="Disable bot pre-play betting")
    parser.add_argument("--policy-model", default="models/linear_policy.json", help="Model path when using policy or policy-rollout bots")
    parser.add_argument("--torch-policy-model", help="Model path when using torch-policy bots")
    parser.add_argument("--torch-bet-model", help="Optional bet model path when using torch-policy bots")
    parser.add_argument("--search-simulations", type=int, help="Simulation budget for rollout/search bots")
    parser.add_argument("--search-root-moves", type=int, help="Maximum root moves considered by rollout/search bots")
    parser.add_argument("--search-rollout-turns", type=int, help="Maximum turns per rollout for rollout/search bots")
    parser.add_argument("--output-json", help="Optional path to write machine-readable tournament/game metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_score is not None:
        game = run_game(
            args.bot_a,
            args.bot_b,
            target_score=args.target_score,
            seed=args.seed,
            max_hands=args.max_hands,
            max_turns=args.max_turns,
            policy_model=args.policy_model,
            torch_policy_model=args.torch_policy_model,
            torch_bet_model=args.torch_bet_model,
            enable_betting=not args.no_betting,
            search_simulations=args.search_simulations,
            search_root_moves=args.search_root_moves,
            search_rollout_turns=args.search_rollout_turns,
        )
        print(format_game_summary(game))
        if args.output_json:
            write_metrics(game_to_metrics(game, config=_run_config(args, mode="game")), args.output_json)
        return 0

    result = run_match(
        args.bot_a,
        args.bot_b,
        hands=args.hands,
        seed=args.seed,
        max_turns=args.max_turns,
        policy_model=args.policy_model,
        torch_policy_model=args.torch_policy_model,
        torch_bet_model=args.torch_bet_model,
        enable_betting=not args.no_betting,
        search_simulations=args.search_simulations,
        search_root_moves=args.search_root_moves,
        search_rollout_turns=args.search_rollout_turns,
    )
    print(format_summary(result))
    if args.output_json:
        write_metrics(tournament_to_metrics(result, config=_run_config(args, mode="tournament")), args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
