from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ladder import run_ladder, write_ladder_metrics
from .policy import train_policy_from_jsonl
from .self_play import export_self_play_jsonl
from .tournament import BOT_TYPES, run_match

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ExperimentArtifacts:
    output_dir: Path
    data_path: Path
    model_path: Path
    metrics_path: Path
    manifest_path: Path
    ladder_path: Path


@dataclass(frozen=True)
class ExperimentResult:
    artifacts: ExperimentArtifacts
    records: int
    training_examples: int
    training_updates: int
    training_accuracy: float
    training_validation_accuracy: float | None
    train_examples: int
    validation_examples: int
    evaluation: dict[str, JsonObject]
    rollout_evaluation: dict[str, JsonObject]
    ladder_ran: bool = False


def run_policy_experiment(
    output_dir: str | Path,
    *,
    teacher_a: str = "point-aware",
    teacher_b: str = "bomb-control",
    data_hands: int = 20,
    epochs: int = 5,
    seed: int = 1,
    eval_hands: int = 10,
    eval_opponents: tuple[str, ...] | list[str] = ("greedy", "point-aware", "bomb-control"),
    max_turns: int = 500,
    only_winners: bool = False,
    observation_mode: str = "perfect",
    averaged: bool = False,
    validation_fraction: float = 0.0,
    ladder_hands: int = 0,
    evaluate_policy_rollout: bool = False,
    rollout_simulations: int = 2,
    rollout_root_moves: int = 8,
    rollout_turns: int = 120,
) -> ExperimentResult:
    if data_hands < 1:
        raise ValueError("data_hands must be at least 1")
    if eval_hands < 1:
        raise ValueError("eval_hands must be at least 1")
    if ladder_hands < 0:
        raise ValueError("ladder_hands must be >= 0")
    unknown = sorted({teacher_a, teacher_b, *eval_opponents} - set(BOT_TYPES))
    if unknown:
        raise ValueError(f"unknown bot(s): {', '.join(unknown)}")

    artifacts = _artifacts_for(output_dir)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    records = export_self_play_jsonl(
        artifacts.data_path,
        bot_a=teacher_a,
        bot_b=teacher_b,
        hands=data_hands,
        seed=seed,
        max_turns=max_turns,
        observation_mode=observation_mode,
    )
    policy, training = train_policy_from_jsonl(
        artifacts.data_path,
        epochs=epochs,
        only_winners=only_winners,
        averaged=averaged,
        validation_fraction=validation_fraction,
    )
    policy.save(artifacts.model_path)

    evaluation = _evaluate_bot(
        bot_name="policy",
        eval_opponents=tuple(eval_opponents),
        eval_hands=eval_hands,
        seed=seed,
        max_turns=max_turns,
        model_path=artifacts.model_path,
    )
    rollout_evaluation = {}
    if evaluate_policy_rollout:
        rollout_evaluation = _evaluate_bot(
            bot_name="policy-rollout",
            eval_opponents=tuple(eval_opponents),
            eval_hands=eval_hands,
            seed=seed + 30_000,
            max_turns=max_turns,
            model_path=artifacts.model_path,
            search_simulations=rollout_simulations,
            search_root_moves=rollout_root_moves,
            search_rollout_turns=rollout_turns,
        )

    ladder_bots = ("policy", *tuple(eval_opponents))
    if evaluate_policy_rollout:
        ladder_bots = ("policy", "policy-rollout", *tuple(eval_opponents))

    ladder_ran = False
    if ladder_hands:
        ladder = run_ladder(
            ladder_bots,
            hands_per_match=ladder_hands,
            seed=seed + 20_000,
            max_turns=max_turns,
            policy_model=str(artifacts.model_path),
        )
        write_ladder_metrics(
            ladder,
            artifacts.ladder_path,
            config={
                "bots": list(ladder_bots),
                "hands_per_match": ladder_hands,
                "seed": seed + 20_000,
                "max_turns": max_turns,
                "policy_model": str(artifacts.model_path),
            },
        )
        ladder_ran = True

    result = ExperimentResult(
        artifacts=artifacts,
        records=records,
        training_examples=training.examples,
        training_updates=training.updates,
        training_accuracy=training.accuracy,
        training_validation_accuracy=training.validation_accuracy,
        train_examples=training.train_examples,
        validation_examples=training.validation_examples,
        evaluation=evaluation,
        rollout_evaluation=rollout_evaluation,
        ladder_ran=ladder_ran,
    )
    _write_metrics(result)
    _write_manifest(
        result,
        teacher_a=teacher_a,
        teacher_b=teacher_b,
        data_hands=data_hands,
        epochs=epochs,
        seed=seed,
        eval_hands=eval_hands,
        eval_opponents=tuple(eval_opponents),
        max_turns=max_turns,
        only_winners=only_winners,
        observation_mode=observation_mode,
        averaged=averaged,
        validation_fraction=validation_fraction,
        evaluate_policy_rollout=evaluate_policy_rollout,
        rollout_simulations=rollout_simulations,
        rollout_root_moves=rollout_root_moves,
        rollout_turns=rollout_turns,
    )
    return result


def format_experiment_summary(result: ExperimentResult) -> str:
    lines = [
        "Haggis policy experiment",
        f"Output: {result.artifacts.output_dir}",
        f"Records: {result.records}",
        f"Training: examples={result.training_examples} train_examples={result.train_examples} "
        f"validation_examples={result.validation_examples} updates={result.training_updates} "
        f"accuracy={result.training_accuracy:.3f} "
        f"validation_accuracy={result.training_validation_accuracy if result.training_validation_accuracy is not None else 'n/a'}",
        "Evaluation:",
    ]
    for opponent, metrics in sorted(result.evaluation.items()):
        lines.append(
            f"  policy vs {opponent}: score {metrics['score'][0]}-{metrics['score'][1]} "
            f"hand_wins {metrics['hand_wins'][0]}-{metrics['hand_wins'][1]} "
            f"margin {metrics['score_margin']:+d}"
        )
    if result.rollout_evaluation:
        lines.append("Policy-rollout evaluation:")
        for opponent, metrics in sorted(result.rollout_evaluation.items()):
            lines.append(
                f"  policy-rollout vs {opponent}: score {metrics['score'][0]}-{metrics['score'][1]} "
                f"hand_wins {metrics['hand_wins'][0]}-{metrics['hand_wins'][1]} "
                f"margin {metrics['score_margin']:+d}"
            )
    lines.extend(
        [
            f"Data: {result.artifacts.data_path}",
            f"Model: {result.artifacts.model_path}",
            f"Metrics: {result.artifacts.metrics_path}",
            f"Manifest: {result.artifacts.manifest_path}",
            f"Ladder: {result.artifacts.ladder_path if result.ladder_ran else 'not run'}",
        ]
    )
    return "\n".join(lines)


def _evaluate_bot(
    *,
    bot_name: str,
    eval_opponents: tuple[str, ...],
    eval_hands: int,
    seed: int,
    max_turns: int,
    model_path: Path,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
) -> dict[str, JsonObject]:
    evaluation: dict[str, JsonObject] = {}
    for index, opponent in enumerate(eval_opponents):
        match = run_match(
            bot_name,
            opponent,
            hands=eval_hands,
            seed=seed + 10_000 + index * 101,
            max_turns=max_turns,
            policy_model=str(model_path),
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        )
        evaluation[opponent] = {
            "hands": len(match.hands),
            "hand_wins": list(match.hand_wins),
            "score": list(match.total_score),
            "score_margin": match.score_margin,
            "average_turns": match.average_turns,
            "bets_placed": list(match.total_bets_placed),
            "bets_succeeded": list(match.total_bets_succeeded),
            "bets_failed": list(match.total_bets_failed),
        }
    return evaluation


def _write_metrics(result: ExperimentResult) -> None:
    payload = {
        "records": result.records,
        "training": {
            "examples": result.training_examples,
            "train_examples": result.train_examples,
            "validation_examples": result.validation_examples,
            "updates": result.training_updates,
            "accuracy": result.training_accuracy,
            "validation_accuracy": result.training_validation_accuracy,
            "averaged": json.loads(result.artifacts.model_path.read_text(encoding="utf-8")).get("averaged", False),
        },
        "evaluation": result.evaluation,
        "policy_rollout_evaluation": result.rollout_evaluation,
        "ladder": str(result.artifacts.ladder_path) if result.ladder_ran else None,
    }
    result.artifacts.metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(result: ExperimentResult, **config: Any) -> None:
    payload = {
        "experiment": "linear_policy_imitation",
        "config": _jsonable(config),
        "artifacts": {
            "data": str(result.artifacts.data_path),
            "model": str(result.artifacts.model_path),
            "metrics": str(result.artifacts.metrics_path),
            "ladder": str(result.artifacts.ladder_path) if result.ladder_ran else None,
        },
        "summary": {
            "records": result.records,
            "training_accuracy": result.training_accuracy,
            "training_validation_accuracy": result.training_validation_accuracy,
            "ladder_ran": result.ladder_ran,
        },
    }
    result.artifacts.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifacts_for(output_dir: str | Path) -> ExperimentArtifacts:
    output = Path(output_dir)
    return ExperimentArtifacts(
        output_dir=output,
        data_path=output / "self_play.jsonl",
        model_path=output / "linear_policy.json",
        metrics_path=output / "metrics.json",
        manifest_path=output / "manifest.json",
        ladder_path=output / "ladder.json",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end Haggis policy experiment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-a", default="point-aware")
    parser.add_argument("--teacher-b", default="bomb-control")
    parser.add_argument("--data-hands", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-hands", type=int, default=10)
    parser.add_argument("--eval-opponents", default="greedy,point-aware,bomb-control")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--only-winners", action="store_true")
    parser.add_argument("--averaged", action="store_true", help="Use averaged perceptron training")
    parser.add_argument("--validation-fraction", type=float, default=0.0, help="Held-out fraction from the end of generated self-play records")
    parser.add_argument("--observation-mode", choices=("perfect", "player"), default="perfect")
    parser.add_argument("--ladder-hands", type=int, default=0, help="If >0, run a policy-vs-opponents ladder with this many hands per ordered matchup")
    parser.add_argument("--evaluate-policy-rollout", action="store_true", help="Also evaluate policy-rollout using the trained model")
    parser.add_argument("--rollout-simulations", type=int, default=2, help="Policy-rollout simulations per root move during experiment evaluation")
    parser.add_argument("--rollout-root-moves", type=int, default=8, help="Policy-rollout root move cap during experiment evaluation")
    parser.add_argument("--rollout-turns", type=int, default=120, help="Policy-rollout rollout turn cap during experiment evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opponents = tuple(bot.strip() for bot in args.eval_opponents.split(",") if bot.strip())
    result = run_policy_experiment(
        args.output_dir,
        teacher_a=args.teacher_a,
        teacher_b=args.teacher_b,
        data_hands=args.data_hands,
        epochs=args.epochs,
        seed=args.seed,
        eval_hands=args.eval_hands,
        eval_opponents=opponents,
        max_turns=args.max_turns,
        only_winners=args.only_winners,
        observation_mode=args.observation_mode,
        averaged=args.averaged,
        validation_fraction=args.validation_fraction,
        ladder_hands=args.ladder_hands,
        evaluate_policy_rollout=args.evaluate_policy_rollout,
        rollout_simulations=args.rollout_simulations,
        rollout_root_moves=args.rollout_root_moves,
        rollout_turns=args.rollout_turns,
    )
    print(format_experiment_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
