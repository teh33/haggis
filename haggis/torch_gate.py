from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import benchmark_to_metrics, run_benchmark
from .tournament import game_to_metrics, run_game


@dataclass(frozen=True)
class TorchGateGame:
    seed: int
    winner: str
    score_margin_for_challenger: int
    score: tuple[int, int]
    hands: int
    path: str


@dataclass(frozen=True)
class TorchGateResult:
    passed: bool
    champion: str
    challenger: str
    target_score: int
    games: tuple[TorchGateGame, ...]
    challenger_wins: int
    champion_wins: int
    average_margin: float
    benchmark: dict
    promoted_to: str | None


def run_torch_gate(
    *,
    champion: str | Path,
    challenger: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...],
    policy_model: str = "models/linear_policy.json",
    target_score: int = 350,
    max_hands: int = 30,
    search_root_moves: int | None = 3,
    search_rollout_turns: int | None = 16,
    challenger_bet_model: str | Path | None = None,
    champion_bet_model: str | Path | None = None,
    promote_to: str | Path | None = None,
    require_wins: int | None = None,
    min_promotion_games: int = 60,
) -> TorchGateResult:
    if not seeds:
        raise ValueError("at least one seed is required")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    required_wins = require_wins if require_wins is not None else len(seeds) // 2 + 1
    if (challenger_bet_model is None) != (champion_bet_model is None):
        raise ValueError("challenger and champion bet models must be provided together")
    if promote_to is not None and len(seeds) < min_promotion_games:
        raise ValueError(f"--promote-to requires at least {min_promotion_games} gate games; got {len(seeds)}")
    if promote_to is not None and required_wins <= len(seeds) // 2:
        raise ValueError("--promote-to requires require_wins to be a strict majority")
    games: list[TorchGateGame] = []
    for seed in seeds:
        game = run_game(
            "torch-policy",
            "torch-policy",
            target_score=target_score,
            seed=seed,
            max_hands=max_hands,
            policy_model=policy_model,
            bot_a_policy_model=str(challenger),
            bot_b_policy_model=str(champion),
            bot_a_bet_model=str(challenger_bet_model) if challenger_bet_model else None,
            bot_b_bet_model=str(champion_bet_model) if champion_bet_model else None,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        )
        metrics = game_to_metrics(
            game,
            config={
                "challenger": str(challenger),
                "champion": str(champion),
                "policy_model": policy_model,
                "challenger_bet_model": str(challenger_bet_model) if challenger_bet_model else None,
                "champion_bet_model": str(champion_bet_model) if champion_bet_model else None,
                "target_score": target_score,
                "max_hands": max_hands,
                "seed": seed,
                "search_root_moves": search_root_moves,
                "search_rollout_turns": search_rollout_turns,
            },
        )
        path = output_root / f"game-{seed}.json"
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        games.append(
            TorchGateGame(
                seed=seed,
                winner="challenger" if game.winner == 0 else "champion",
                score_margin_for_challenger=game.score_margin,
                score=game.total_score,
                hands=game.hands,
                path=str(path),
            )
        )

    challenger_wins = sum(1 for game in games if game.winner == "challenger")
    champion_wins = len(games) - challenger_wins
    average_margin = sum(game.score_margin_for_challenger for game in games) / len(games)
    benchmark = benchmark_to_metrics(
        run_benchmark(
            bots=("torch-policy",),
            states=3,
            seed=max(seeds) + 10_000,
            policy_model=policy_model,
            torch_policy_model=str(challenger),
            torch_bet_model=str(challenger_bet_model) if challenger_bet_model else None,
        ),
        config={"torch_policy_model": str(challenger), "policy_model": policy_model},
    )
    passed = challenger_wins >= required_wins and average_margin > 0
    promoted_path = None
    if passed and promote_to is not None:
        promoted_path = str(promote_to)
        Path(promoted_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(challenger, promoted_path)

    result = TorchGateResult(
        passed=passed,
        champion=str(champion),
        challenger=str(challenger),
        target_score=target_score,
        games=tuple(games),
        challenger_wins=challenger_wins,
        champion_wins=champion_wins,
        average_margin=average_margin,
        benchmark=benchmark,
        promoted_to=promoted_path,
    )
    (output_root / "gate_report.json").write_text(json.dumps(_result_payload(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _result_payload(result: TorchGateResult) -> dict:
    payload = asdict(result)
    payload["games"] = [asdict(game) for game in result.games]
    return payload


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    return tuple(seeds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate a torch challenger against a frozen torch champion using 350-point games")
    parser.add_argument("--champion", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="200,201,202,203,204", help="Comma-separated seeds; inclusive ranges like 7600:7659 are accepted")
    parser.add_argument("--policy-model", default="models/linear_policy.json")
    parser.add_argument("--target-score", type=int, default=350)
    parser.add_argument("--max-hands", type=int, default=30)
    parser.add_argument("--search-root-moves", type=int, default=3)
    parser.add_argument("--search-rollout-turns", type=int, default=16)
    parser.add_argument("--challenger-bet-model", help="Optional bet model to use for torch-policy bots during the gate")
    parser.add_argument("--champion-bet-model", help="Optional bet model to use for torch-policy bots during the gate")
    parser.add_argument("--promote-to", help="Copy challenger here if the gate passes; requires at least 60 seeds by default")
    parser.add_argument("--require-wins", type=int, help="Minimum challenger wins required; defaults to majority")
    parser.add_argument("--min-promotion-games", type=int, default=60, help="Minimum number of gate games required when --promote-to is used")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_torch_gate(
        champion=args.champion,
        challenger=args.challenger,
        output_dir=args.output_dir,
        seeds=_parse_seeds(args.seeds),
        policy_model=args.policy_model,
        target_score=args.target_score,
        max_hands=args.max_hands,
        search_root_moves=args.search_root_moves,
        search_rollout_turns=args.search_rollout_turns,
        challenger_bet_model=args.challenger_bet_model,
        champion_bet_model=args.champion_bet_model,
        promote_to=args.promote_to,
        require_wins=args.require_wins,
        min_promotion_games=args.min_promotion_games,
    )
    print(json.dumps(_result_payload(result), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
