from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .tournament import BOT_TYPES, make_bot
from .engine import HaggisState


@dataclass(frozen=True)
class StateBenchmark:
    seed: int
    current_player: int
    hand_sizes: tuple[int, int]
    legal_moves: int
    legal_move_seconds: float


@dataclass(frozen=True)
class BotBenchmark:
    bot: str
    decisions: int
    total_seconds: float
    average_seconds: float
    moves: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    seeds: tuple[int, ...]
    state_benchmarks: tuple[StateBenchmark, ...]
    bot_benchmarks: tuple[BotBenchmark, ...]


def run_benchmark(
    *,
    bots: tuple[str, ...] | list[str] = ("random", "greedy", "point-aware", "bomb-control"),
    states: int = 5,
    seed: int = 1,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
) -> BenchmarkResult:
    if states < 1:
        raise ValueError("states must be at least 1")
    bot_names = tuple(bots)
    unknown = sorted(set(bot_names) - set(BOT_TYPES))
    if unknown:
        raise ValueError(f"unknown bot(s): {', '.join(unknown)}")

    seeds = tuple(seed + index for index in range(states))
    benchmark_states = tuple(HaggisState.new_deal(seed=state_seed) for state_seed in seeds)
    state_benchmarks = tuple(_benchmark_state(state_seed, state) for state_seed, state in zip(seeds, benchmark_states))
    bot_benchmarks = tuple(
        _benchmark_bot(
            bot_name,
            benchmark_states,
            seed=seed,
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        )
        for bot_name in bot_names
    )
    return BenchmarkResult(seeds=seeds, state_benchmarks=state_benchmarks, bot_benchmarks=bot_benchmarks)


def benchmark_to_metrics(result: BenchmarkResult, *, config: dict | None = None) -> dict:
    return {
        "config": config or {},
        "seeds": list(result.seeds),
        "states": [
            {
                "seed": state.seed,
                "current_player": state.current_player,
                "hand_sizes": list(state.hand_sizes),
                "legal_moves": state.legal_moves,
                "legal_move_seconds": state.legal_move_seconds,
            }
            for state in result.state_benchmarks
        ],
        "bots": [
            {
                "bot": bot.bot,
                "decisions": bot.decisions,
                "total_seconds": bot.total_seconds,
                "average_seconds": bot.average_seconds,
                "moves": list(bot.moves),
            }
            for bot in result.bot_benchmarks
        ],
    }


def write_benchmark_metrics(result: BenchmarkResult, path: str | Path, *, config: dict | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark_to_metrics(result, config=config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_benchmark_summary(result: BenchmarkResult) -> str:
    legal_total = sum(state.legal_moves for state in result.state_benchmarks)
    legal_seconds = sum(state.legal_move_seconds for state in result.state_benchmarks)
    lines = [
        "Haggis benchmark",
        f"States: {len(result.state_benchmarks)}",
        f"Legal moves: {legal_total} generated in {legal_seconds:.6f}s",
        "Bot decisions:",
    ]
    for bot in result.bot_benchmarks:
        lines.append(
            f"  {bot.bot}: {bot.decisions} decisions in {bot.total_seconds:.6f}s "
            f"avg={bot.average_seconds:.6f}s"
        )
    return "\n".join(lines)


def _benchmark_state(seed: int, state: HaggisState) -> StateBenchmark:
    started = perf_counter()
    moves = state.legal_moves()
    elapsed = perf_counter() - started
    return StateBenchmark(
        seed=seed,
        current_player=state.current_player,
        hand_sizes=(len(state.hands[0]), len(state.hands[1])),
        legal_moves=len(moves),
        legal_move_seconds=elapsed,
    )


def _benchmark_bot(
    bot_name: str,
    states: tuple[HaggisState, ...],
    *,
    seed: int,
    search_simulations: int | None,
    search_root_moves: int | None,
    search_rollout_turns: int | None,
) -> BotBenchmark:
    bot = make_bot(
        bot_name,
        seed=seed,
        search_simulations=search_simulations,
        search_root_moves=search_root_moves,
        search_rollout_turns=search_rollout_turns,
    )
    selected: list[str] = []
    started = perf_counter()
    for state in states:
        move = bot.choose_move(state)
        selected.append(_move_id(move))
    elapsed = perf_counter() - started
    return BotBenchmark(
        bot=bot_name,
        decisions=len(states),
        total_seconds=elapsed,
        average_seconds=elapsed / len(states),
        moves=tuple(selected),
    )


def _move_id(move) -> str:
    if move.is_pass:
        return "pass"
    cards = " ".join(card.short_name() for card in move.cards)
    combo = move.combination
    return f"{combo.type}:{combo.rank}:{combo.bomb_rank}:{cards}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Haggis legal move generation and bot decision latency")
    parser.add_argument("--bots", default="random,greedy,point-aware,bomb-control")
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--search-simulations", type=int)
    parser.add_argument("--search-root-moves", type=int)
    parser.add_argument("--search-rollout-turns", type=int)
    parser.add_argument("--output-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bots = tuple(bot.strip() for bot in args.bots.split(",") if bot.strip())
    result = run_benchmark(
        bots=bots,
        states=args.states,
        seed=args.seed,
        search_simulations=args.search_simulations,
        search_root_moves=args.search_root_moves,
        search_rollout_turns=args.search_rollout_turns,
    )
    print(format_benchmark_summary(result))
    if args.output_json:
        write_benchmark_metrics(
            result,
            args.output_json,
            config={
                "bots": list(bots),
                "states": args.states,
                "seed": args.seed,
                "search_simulations": args.search_simulations,
                "search_root_moves": args.search_root_moves,
                "search_rollout_turns": args.search_rollout_turns,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
