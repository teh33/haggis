from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .torch_policy import train_torch_policy_from_jsonl


@dataclass(frozen=True)
class SweepTrial:
    rank: int
    output: str
    hidden_size: int
    batch_size: int
    dropout: float
    learning_rate: float
    weight_decay: float
    margin_weight: float
    train_accuracy: float
    validation_accuracy: float | None


def run_sweep(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    epochs: int,
    validation_fraction: float,
    seed: int,
    hidden_sizes: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    dropouts: tuple[float, ...],
    learning_rates: tuple[float, ...],
    weight_decays: tuple[float, ...],
    margin_weights: tuple[float, ...] = (0.0,),
) -> list[SweepTrial]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trials: list[SweepTrial] = []
    for index, (hidden_size, batch_size, dropout, learning_rate, weight_decay, margin_weight) in enumerate(
        itertools.product(hidden_sizes, batch_sizes, dropouts, learning_rates, weight_decays, margin_weights), start=1
    ):
        model_path = output_root / f"trial-{index:03d}.pt"
        result = train_torch_policy_from_jsonl(
            input_path,
            output_path=model_path,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            batch_size=batch_size,
            dropout=dropout,
            validation_fraction=validation_fraction,
            seed=seed,
            weight_decay=weight_decay,
            margin_weight=margin_weight,
        )
        trials.append(
            SweepTrial(
                rank=0,
                output=str(model_path),
                hidden_size=hidden_size,
                batch_size=batch_size,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                margin_weight=margin_weight,
                train_accuracy=result.train_accuracy,
                validation_accuracy=result.validation_accuracy,
            )
        )
    ranked = sorted(trials, key=lambda trial: trial.validation_accuracy if trial.validation_accuracy is not None else trial.train_accuracy, reverse=True)
    return [SweepTrial(rank=index, **{key: value for key, value in asdict(trial).items() if key != "rank"}) for index, trial in enumerate(ranked, start=1)]


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep experimental PyTorch Haggis policy hyperparameters")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-sizes", default="64,128")
    parser.add_argument("--batch-sizes", default="1,16,32")
    parser.add_argument("--dropouts", default="0.0,0.1")
    parser.add_argument("--learning-rates", default="0.001")
    parser.add_argument("--weight-decays", default="0.0001")
    parser.add_argument("--margin-weights", default="0.0")
    parser.add_argument("--summary", help="Optional JSON summary path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trials = run_sweep(
        args.input,
        output_dir=args.output_dir,
        epochs=args.epochs,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        hidden_sizes=_parse_ints(args.hidden_sizes),
        batch_sizes=_parse_ints(args.batch_sizes),
        dropouts=_parse_floats(args.dropouts),
        learning_rates=_parse_floats(args.learning_rates),
        weight_decays=_parse_floats(args.weight_decays),
        margin_weights=_parse_floats(args.margin_weights),
    )
    payload = {"trials": [asdict(trial) for trial in trials], "best": asdict(trials[0]) if trials else None}
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
