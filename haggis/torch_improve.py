from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import run_benchmark, benchmark_to_metrics
from .ladder import run_ladder, ladder_to_metrics
from .self_play import export_self_play_jsonl
from .torch_gate import run_torch_gate
from .torch_policy import train_torch_policy_from_jsonl


@dataclass(frozen=True)
class ImproveIteration:
    iteration: int
    seed_model: str
    candidate_model: str
    training_records: int
    train_accuracy: float
    validation_accuracy: float | None
    accepted: bool
    previous_rating: float | None
    candidate_rating: float | None
    previous_win_rate: float | None
    candidate_win_rate: float | None
    gate_passed: bool
    gate_challenger_wins: int
    gate_champion_wins: int
    gate_average_margin: float


def run_improvement_loop(
    *,
    seed_model: str | Path,
    output_dir: str | Path,
    iterations: int = 1,
    hands_per_iteration: int = 50,
    teacher_a: str = "policy-rollout",
    teacher_b: str = "information-set",
    champion_baseline: str = "policy-rollout",
    policy_model: str = "models/linear_policy.json",
    epochs: int = 6,
    hidden_size: int = 128,
    batch_size: int = 16,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    margin_weight: float = 0.0,
    validation_fraction: float = 0.2,
    seed: int = 1,
    ladder_hands: int = 4,
    gate_seeds: tuple[int, ...] = (200, 201, 202, 203, 204),
    gate_target_score: int = 350,
    gate_max_hands: int = 30,
    gate_require_wins: int | None = None,
    search_root_moves: int | None = 4,
    search_rollout_turns: int | None = 40,
) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if hands_per_iteration < 1:
        raise ValueError("hands_per_iteration must be at least 1")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    current_model = output_root / "champion.pt"
    shutil.copyfile(seed_model, current_model)
    iterations_report: list[ImproveIteration] = []

    for iteration in range(1, iterations + 1):
        iteration_dir = output_root / f"iteration-{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        previous_model = iteration_dir / "previous.pt"
        shutil.copyfile(current_model, previous_model)
        data_a = iteration_dir / "teacher_a.jsonl"
        data_b = iteration_dir / "teacher_b.jsonl"
        training_data = iteration_dir / "training.jsonl"
        records_a = export_self_play_jsonl(
            data_a,
            bot_a=teacher_a,
            bot_b=teacher_b,
            hands=hands_per_iteration,
            seed=seed + iteration * 100,
            observation_mode="player",
            bot_a_policy_model=_model_for_bot(teacher_a, policy_model=policy_model, torch_policy_model=str(current_model)),
            bot_b_policy_model=_model_for_bot(teacher_b, policy_model=policy_model, torch_policy_model=str(current_model)),
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        )
        records_b = export_self_play_jsonl(
            data_b,
            bot_a=teacher_b,
            bot_b=teacher_a,
            hands=hands_per_iteration,
            seed=seed + iteration * 100 + 10_000,
            observation_mode="player",
            bot_a_policy_model=_model_for_bot(teacher_b, policy_model=policy_model, torch_policy_model=str(current_model)),
            bot_b_policy_model=_model_for_bot(teacher_a, policy_model=policy_model, torch_policy_model=str(current_model)),
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        )
        training_data.write_text(data_a.read_text(encoding="utf-8") + data_b.read_text(encoding="utf-8"), encoding="utf-8")
        candidate_model = iteration_dir / "candidate.pt"
        training_result = train_torch_policy_from_jsonl(
            training_data,
            output_path=candidate_model,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            batch_size=batch_size,
            dropout=dropout,
            validation_fraction=validation_fraction,
            seed=seed + iteration,
            weight_decay=weight_decay,
            margin_weight=margin_weight,
        )
        ladder = run_ladder(
            (f"candidate:torch-policy@{candidate_model}", f"previous:torch-policy@{current_model}", champion_baseline),
            hands_per_match=ladder_hands,
            seed=seed + iteration * 1_000,
            policy_model=policy_model,
            torch_policy_model=str(candidate_model),
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        )
        ladder_metrics = ladder_to_metrics(
            ladder,
            config={
                "candidate_model": str(candidate_model),
                "previous_model": str(current_model),
                "champion_baseline": champion_baseline,
                "policy_model": policy_model,
                "torch_policy_model": str(candidate_model),
                "hands_per_match": ladder_hands,
            },
        )
        (iteration_dir / "ladder.json").write_text(json.dumps(ladder_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate = _standing(ladder_metrics, "candidate")
        previous = _standing(ladder_metrics, "previous")
        baseline = _standing(ladder_metrics, champion_baseline)
        gate = run_torch_gate(
            champion=current_model,
            challenger=candidate_model,
            output_dir=iteration_dir / "gate",
            seeds=tuple(seed + iteration * 10_000 + gate_seed for gate_seed in gate_seeds),
            policy_model=policy_model,
            target_score=gate_target_score,
            max_hands=gate_max_hands,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
            require_wins=gate_require_wins,
        )
        accepted = gate.passed and _candidate_improves_ladder(candidate, previous)
        if accepted:
            shutil.copyfile(candidate_model, current_model)
        iterations_report.append(
            ImproveIteration(
                iteration=iteration,
                seed_model=str(previous_model),
                candidate_model=str(candidate_model),
                training_records=records_a + records_b,
                train_accuracy=training_result.train_accuracy,
                validation_accuracy=training_result.validation_accuracy,
                accepted=accepted,
                previous_rating=previous["rating"] if previous else None,
                candidate_rating=candidate["rating"] if candidate else None,
                previous_win_rate=previous["hand_win_rate"] if previous else None,
                candidate_win_rate=candidate["hand_win_rate"] if candidate else None,
                gate_passed=gate.passed,
                gate_challenger_wins=gate.challenger_wins,
                gate_champion_wins=gate.champion_wins,
                gate_average_margin=gate.average_margin,
            )
        )

    benchmark = run_benchmark(
        bots=("torch-policy", "policy"),
        states=3,
        seed=seed + 50_000,
        policy_model=policy_model,
        torch_policy_model=str(current_model),
    )
    benchmark_metrics = benchmark_to_metrics(
        benchmark,
        config={"policy_model": policy_model, "torch_policy_model": str(current_model), "states": 3, "seed": seed + 50_000},
    )
    payload = {
        "seed_model": str(seed_model),
        "champion_model": str(current_model),
        "iterations": [asdict(item) for item in iterations_report],
        "benchmark": benchmark_metrics,
    }
    (output_root / "improve_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _model_for_bot(bot: str, *, policy_model: str, torch_policy_model: str) -> str | None:
    if bot == "torch-policy":
        return torch_policy_model
    if bot in {"policy", "policy-rollout"}:
        return policy_model
    return None


def _standing(metrics: dict, bot: str) -> dict | None:
    return next((standing for standing in metrics["standings"] if standing["bot"] == bot), None)


def _candidate_improves_ladder(candidate: dict | None, previous: dict | None) -> bool:
    if candidate is None or previous is None:
        return False
    return candidate["rating"] >= previous["rating"] and candidate["hand_win_rate"] >= previous["hand_win_rate"]

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
    parser = argparse.ArgumentParser(description="Run offline PyTorch Haggis policy improvement iterations")
    parser.add_argument("--seed-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--hands-per-iteration", type=int, default=50)
    parser.add_argument("--teacher-a", default="policy-rollout")
    parser.add_argument("--teacher-b", default="information-set")
    parser.add_argument("--champion-baseline", default="policy-rollout")
    parser.add_argument("--policy-model", default="models/linear_policy.json")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ladder-hands", type=int, default=4)
    parser.add_argument("--gate-seeds", default="200,201,202,203,204", help="Comma-separated gate seeds; inclusive ranges like 7600:7659 are accepted")
    parser.add_argument("--gate-target-score", type=int, default=350)
    parser.add_argument("--gate-max-hands", type=int, default=30)
    parser.add_argument("--gate-require-wins", type=int)
    parser.add_argument("--search-root-moves", type=int, default=4)
    parser.add_argument("--search-rollout-turns", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_improvement_loop(
        seed_model=args.seed_model,
        output_dir=args.output_dir,
        iterations=args.iterations,
        hands_per_iteration=args.hands_per_iteration,
        teacher_a=args.teacher_a,
        teacher_b=args.teacher_b,
        champion_baseline=args.champion_baseline,
        policy_model=args.policy_model,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin_weight=args.margin_weight,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        ladder_hands=args.ladder_hands,
        gate_seeds=_parse_seeds(args.gate_seeds),
        gate_target_score=args.gate_target_score,
        gate_max_hands=args.gate_max_hands,
        gate_require_wins=args.gate_require_wins,
        search_root_moves=args.search_root_moves,
        search_rollout_turns=args.search_rollout_turns,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
