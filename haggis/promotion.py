from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .benchmark import benchmark_to_metrics, run_benchmark
from .ladder import ladder_to_metrics, run_ladder

DEFAULT_BASELINES = ("point-aware", "bomb-control", "information-set", "greedy")


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    champion_rating: float
    challenger_rating: float
    rating_delta: float
    champion_win_rate: float
    challenger_win_rate: float
    win_rate_delta: float
    champion_seconds_per_decision: float
    challenger_seconds_per_decision: float
    speed_ratio: float
    reasons: tuple[str, ...]


def run_promotion_gate(
    *,
    champion_model: str | Path,
    challenger_model: str | Path,
    output_dir: str | Path,
    hands: int = 12,
    seed: int = 1,
    baselines: tuple[str, ...] = DEFAULT_BASELINES,
    search_simulations: int | None = 8,
    search_root_moves: int | None = 4,
    search_rollout_turns: int | None = 40,
    min_rating_delta: float = 0.0,
    min_win_rate_delta: float = 0.0,
    max_speed_ratio: float = 1.25,
    benchmark_states: int = 25,
    promote_to: str | Path | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    champion_model = Path(champion_model)
    challenger_model = Path(challenger_model)
    if not champion_model.exists():
        raise FileNotFoundError(f"champion model not found: {champion_model}")
    if not challenger_model.exists():
        raise FileNotFoundError(f"challenger model not found: {challenger_model}")

    champion_ladder = run_ladder(
        ("policy-rollout", *baselines),
        hands_per_match=hands,
        seed=seed,
        policy_model=str(champion_model),
        search_simulations=search_simulations,
        search_root_moves=search_root_moves,
        search_rollout_turns=search_rollout_turns,
    )
    challenger_ladder = run_ladder(
        ("policy-rollout", *baselines),
        hands_per_match=hands,
        seed=seed,
        policy_model=str(challenger_model),
        search_simulations=search_simulations,
        search_root_moves=search_root_moves,
        search_rollout_turns=search_rollout_turns,
    )
    champion_benchmark = run_benchmark(
        bots=("policy-rollout",),
        states=benchmark_states,
        seed=seed,
        policy_model=str(champion_model),
        search_simulations=search_simulations,
        search_root_moves=search_root_moves,
        search_rollout_turns=search_rollout_turns,
    )
    challenger_benchmark = run_benchmark(
        bots=("policy-rollout",),
        states=benchmark_states,
        seed=seed,
        policy_model=str(challenger_model),
        search_simulations=search_simulations,
        search_root_moves=search_root_moves,
        search_rollout_turns=search_rollout_turns,
    )

    config = {
        "champion_model": str(champion_model),
        "challenger_model": str(challenger_model),
        "hands": hands,
        "seed": seed,
        "baselines": list(baselines),
        "search_simulations": search_simulations,
        "search_root_moves": search_root_moves,
        "search_rollout_turns": search_rollout_turns,
        "min_rating_delta": min_rating_delta,
        "min_win_rate_delta": min_win_rate_delta,
        "max_speed_ratio": max_speed_ratio,
        "benchmark_states": benchmark_states,
    }
    champion_ladder_metrics = ladder_to_metrics(champion_ladder, config={**config, "model_role": "champion"})
    challenger_ladder_metrics = ladder_to_metrics(challenger_ladder, config={**config, "model_role": "challenger"})
    champion_benchmark_metrics = benchmark_to_metrics(champion_benchmark, config={**config, "model_role": "champion"})
    challenger_benchmark_metrics = benchmark_to_metrics(challenger_benchmark, config={**config, "model_role": "challenger"})

    decision = decide_promotion(
        champion_ladder_metrics=champion_ladder_metrics,
        challenger_ladder_metrics=challenger_ladder_metrics,
        champion_benchmark_metrics=champion_benchmark_metrics,
        challenger_benchmark_metrics=challenger_benchmark_metrics,
        min_rating_delta=min_rating_delta,
        min_win_rate_delta=min_win_rate_delta,
        max_speed_ratio=max_speed_ratio,
    )
    promoted_to = None
    if decision.passed and promote_to is not None:
        destination = Path(promote_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(challenger_model, destination)
        promoted_to = str(destination)

    payload = {
        "config": config,
        "decision": decision_to_dict(decision),
        "promoted_to": promoted_to,
        "artifacts": {
            "champion_ladder": "champion-ladder.json",
            "challenger_ladder": "challenger-ladder.json",
            "champion_benchmark": "champion-benchmark.json",
            "challenger_benchmark": "challenger-benchmark.json",
        },
    }
    _write_json(output / "champion-ladder.json", champion_ladder_metrics)
    _write_json(output / "challenger-ladder.json", challenger_ladder_metrics)
    _write_json(output / "champion-benchmark.json", champion_benchmark_metrics)
    _write_json(output / "challenger-benchmark.json", challenger_benchmark_metrics)
    _write_json(output / "promotion.json", payload)
    return payload


def decide_promotion(
    *,
    champion_ladder_metrics: dict,
    challenger_ladder_metrics: dict,
    champion_benchmark_metrics: dict,
    challenger_benchmark_metrics: dict,
    min_rating_delta: float,
    min_win_rate_delta: float,
    max_speed_ratio: float,
) -> PromotionDecision:
    champion_entry = _policy_rollout_entry(champion_ladder_metrics)
    challenger_entry = _policy_rollout_entry(challenger_ladder_metrics)
    champion_seconds = _policy_rollout_seconds(champion_benchmark_metrics)
    challenger_seconds = _policy_rollout_seconds(challenger_benchmark_metrics)
    rating_delta = float(challenger_entry["rating"]) - float(champion_entry["rating"])
    win_rate_delta = float(challenger_entry["hand_win_rate"]) - float(champion_entry["hand_win_rate"])
    speed_ratio = challenger_seconds / champion_seconds if champion_seconds else float("inf")

    reasons: list[str] = []
    if rating_delta < min_rating_delta:
        reasons.append(f"rating delta {rating_delta:.3f} below minimum {min_rating_delta:.3f}")
    if win_rate_delta < min_win_rate_delta:
        reasons.append(f"win-rate delta {win_rate_delta:.3f} below minimum {min_win_rate_delta:.3f}")
    if speed_ratio > max_speed_ratio:
        reasons.append(f"speed ratio {speed_ratio:.3f} above maximum {max_speed_ratio:.3f}")

    return PromotionDecision(
        passed=not reasons,
        champion_rating=float(champion_entry["rating"]),
        challenger_rating=float(challenger_entry["rating"]),
        rating_delta=rating_delta,
        champion_win_rate=float(champion_entry["hand_win_rate"]),
        challenger_win_rate=float(challenger_entry["hand_win_rate"]),
        win_rate_delta=win_rate_delta,
        champion_seconds_per_decision=champion_seconds,
        challenger_seconds_per_decision=challenger_seconds,
        speed_ratio=speed_ratio,
        reasons=tuple(reasons),
    )


def decision_to_dict(decision: PromotionDecision) -> dict:
    return {
        "passed": decision.passed,
        "champion_rating": decision.champion_rating,
        "challenger_rating": decision.challenger_rating,
        "rating_delta": decision.rating_delta,
        "champion_win_rate": decision.champion_win_rate,
        "challenger_win_rate": decision.challenger_win_rate,
        "win_rate_delta": decision.win_rate_delta,
        "champion_seconds_per_decision": decision.champion_seconds_per_decision,
        "challenger_seconds_per_decision": decision.challenger_seconds_per_decision,
        "speed_ratio": decision.speed_ratio,
        "reasons": list(decision.reasons),
    }


def _policy_rollout_entry(metrics: dict) -> dict:
    for entry in metrics["standings"]:
        if entry["bot"] == "policy-rollout":
            return entry
    raise ValueError("metrics do not contain policy-rollout standing")


def _policy_rollout_seconds(metrics: dict) -> float:
    for entry in metrics["bots"]:
        if entry["bot"] == "policy-rollout":
            return float(entry["average_seconds"])
    raise ValueError("metrics do not contain policy-rollout benchmark")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate a challenger Haggis model against the current champion")
    parser.add_argument("--champion", default="models/linear_policy.json", help="Champion model path")
    parser.add_argument("--challenger", required=True, help="Challenger model path")
    parser.add_argument("--output-dir", required=True, help="Directory for promotion artifacts")
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES), help="Comma-separated baseline bots")
    parser.add_argument("--search-simulations", type=int, default=8)
    parser.add_argument("--search-root-moves", type=int, default=4)
    parser.add_argument("--search-rollout-turns", type=int, default=40)
    parser.add_argument("--benchmark-states", type=int, default=25)
    parser.add_argument("--min-rating-delta", type=float, default=0.0)
    parser.add_argument("--min-win-rate-delta", type=float, default=0.0)
    parser.add_argument("--max-speed-ratio", type=float, default=1.25)
    parser.add_argument("--promote-to", help="Copy challenger here when gate passes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_promotion_gate(
        champion_model=args.champion,
        challenger_model=args.challenger,
        output_dir=args.output_dir,
        hands=args.hands,
        seed=args.seed,
        baselines=tuple(bot.strip() for bot in args.baselines.split(",") if bot.strip()),
        search_simulations=args.search_simulations,
        search_root_moves=args.search_root_moves,
        search_rollout_turns=args.search_rollout_turns,
        min_rating_delta=args.min_rating_delta,
        min_win_rate_delta=args.min_win_rate_delta,
        max_speed_ratio=args.max_speed_ratio,
        benchmark_states=args.benchmark_states,
        promote_to=args.promote_to,
    )
    decision = result["decision"]
    status = "PASS" if decision["passed"] else "FAIL"
    print(
        f"Promotion gate {status}: rating_delta={decision['rating_delta']:.2f}, "
        f"win_rate_delta={decision['win_rate_delta']:.3f}, speed_ratio={decision['speed_ratio']:.3f}; "
        f"artifacts={args.output_dir}"
    )
    for reason in decision["reasons"]:
        print(f"- {reason}")
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
