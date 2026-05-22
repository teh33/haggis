from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .policy import FeatureVector, features_from_record_action, features_from_state_action, load_records_jsonl, split_train_validation
from .engine import HaggisState, Move

JsonObject = dict[str, Any]

try:
    import torch as _torch_import
    _TorchModuleBase = _torch_import.nn.Module
except ModuleNotFoundError:
    _torch_import = None
    _TorchModuleBase = object


@dataclass(frozen=True)
class TorchTrainingResult:
    records: int
    train_records: int
    validation_records: int
    examples: int
    epochs: int
    train_accuracy: float
    validation_accuracy: float | None
    feature_count: int


def train_torch_policy_from_jsonl(
    input_path: str | Path,
    *,
    output_path: str | Path,
    epochs: int = 5,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
    batch_size: int = 32,
    dropout: float = 0.1,
    validation_fraction: float = 0.2,
    seed: int = 1,
    weight_decay: float = 0.0001,
    margin_weight: float = 0.0,
) -> TorchTrainingResult:
    torch = _torch()
    records = tuple(load_records_jsonl(input_path))
    if not records:
        raise ValueError("training data is empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    train_records, validation_records = split_train_validation(records, validation_fraction=validation_fraction)
    feature_names = _feature_names(train_records)
    if not feature_names:
        raise ValueError("training data produced no features")

    torch.manual_seed(seed)
    random.seed(seed)
    model = TorchActionRanker(len(feature_names), hidden_size, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    examples = 0
    train_batches = tuple(_record_action_batch(record, feature_names, margin_weight=margin_weight) for record in train_records)
    for epoch in range(epochs):
        shuffled = list(train_batches)
        random.Random(seed + epoch).shuffle(shuffled)
        model.train()
        for batch_start in range(0, len(shuffled), batch_size):
            batch = shuffled[batch_start : batch_start + batch_size]
            if not batch:
                continue
            logits = [model(features) for features, _target, _sample_weight in batch]
            targets = torch.tensor([target for _features, target, _sample_weight in batch], dtype=torch.long)
            sample_weights = torch.tensor([sample_weight for _features, _target, sample_weight in batch], dtype=torch.float32)
            per_sample_loss = loss_fn(torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=-1e9), targets)
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            examples += len(batch)

    train_accuracy = evaluate_torch_policy_accuracy(model, train_records, feature_names)
    validation_accuracy = evaluate_torch_policy_accuracy(model, validation_records, feature_names) if validation_records else None
    _save_torch_policy(
        output_path,
        model=model,
        feature_names=feature_names,
        hidden_size=hidden_size,
        config={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "dropout": dropout,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "weight_decay": weight_decay,
            "margin_weight": margin_weight,
        },
        metrics={
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "examples": examples,
            "records": len(records),
            "train_records": len(train_records),
            "validation_records": len(validation_records),
        },
    )
    return TorchTrainingResult(
        records=len(records),
        train_records=len(train_records),
        validation_records=len(validation_records),
        examples=examples,
        epochs=epochs,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        feature_count=len(feature_names),
    )


class TorchActionRanker(_TorchModuleBase):
    def __init__(self, feature_count: int, hidden_size: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        torch = _torch()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(feature_count, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def load_torch_policy(path: str | Path) -> "TorchPolicy":
    torch = _torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("model_type") != "torch_action_ranker":
        raise ValueError("not a Haggis torch action-ranker model")
    feature_names = tuple(str(name) for name in payload["feature_names"])
    hidden_size = int(payload.get("hidden_size", payload.get("config", {}).get("hidden_size", 128)))
    dropout = float(payload.get("config", {}).get("dropout", 0.0))
    model = TorchActionRanker(len(feature_names), hidden_size, dropout=dropout)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return TorchPolicy(model=model, feature_names=feature_names, model_path=Path(path))


@dataclass
class TorchPolicy:
    model: Any
    feature_names: tuple[str, ...]
    model_path: Path | None = None

    def score_move(self, state: HaggisState, move: Move) -> float:
        torch = _torch()
        features = features_from_state_action(state, move)
        row = torch.tensor([[float(features.get(name, 0.0)) for name in self.feature_names]], dtype=torch.float32)
        with torch.no_grad():
            self.model.eval()
            return float(self.model(row).item())

    def choose_move(self, state: HaggisState, legal_moves: tuple[Move, ...]) -> Move:
        torch = _torch()
        if not legal_moves:
            raise ValueError("no legal moves available")
        rows = []
        for move in legal_moves:
            features = features_from_state_action(state, move)
            rows.append([float(features.get(name, 0.0)) for name in self.feature_names])
        matrix = torch.tensor(rows, dtype=torch.float32)
        with torch.no_grad():
            self.model.eval()
            predicted = int(torch.argmax(self.model(matrix)).item())
        return legal_moves[predicted]


def evaluate_torch_policy_accuracy(model: Any, records: tuple[JsonObject, ...], feature_names: tuple[str, ...]) -> float:
    torch = _torch()
    if not records:
        return 0.0
    correct = 0
    with torch.no_grad():
        model.eval()
        for record in records:
            features = _record_action_feature_matrix(record, feature_names)
            predicted = int(torch.argmax(model(features)).item())
            if predicted == int(record["selected_action_index"]):
                correct += 1
    return correct / len(records)


def _record_action_batch(record: JsonObject, feature_names: tuple[str, ...], *, margin_weight: float = 0.0):
    return _record_action_feature_matrix(record, feature_names), int(record["selected_action_index"]), _record_sample_weight(record, margin_weight=margin_weight)


def _record_sample_weight(record: JsonObject, *, margin_weight: float) -> float:
    if margin_weight <= 0.0:
        return 1.0
    outcome = record.get("outcome", {})
    won = bool(outcome.get("actor_won", False))
    margin = abs(float(outcome.get("actor_score_margin", outcome.get("score_margin_for_actor", 0.0)))) / 350.0
    direction = 1.0 if won else -0.5
    return max(0.05, 1.0 + margin_weight * direction * margin)


def _record_action_feature_matrix(record: JsonObject, feature_names: tuple[str, ...]):
    torch = _torch()
    rows = []
    for action in record["legal_actions"]:
        features = features_from_record_action(record, action)
        rows.append([float(features.get(name, 0.0)) for name in feature_names])
    return torch.tensor(rows, dtype=torch.float32)


def _feature_names(records: tuple[JsonObject, ...]) -> tuple[str, ...]:
    names: set[str] = set()
    for record in records:
        for action in record["legal_actions"]:
            names.update(features_from_record_action(record, action))
    return tuple(sorted(names))


def _save_torch_policy(path: str | Path, *, model: Any, feature_names: tuple[str, ...], hidden_size: int, config: JsonObject, metrics: JsonObject) -> None:
    torch = _torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "torch_action_ranker",
            "feature_names": list(feature_names),
            "hidden_size": hidden_size,
            "state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        output,
    )


def _torch():
    if _torch_import is None:
        raise ModuleNotFoundError("PyTorch is required for haggis.torch_policy; install with `python -m pip install torch`")
    return _torch_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train experimental PyTorch Haggis policy models")
    parser.add_argument("--input", required=True, help="Self-play JSONL input path")
    parser.add_argument("--output", required=True, help="Torch model output path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--margin-weight", type=float, default=0.0, help="Weight winning high-margin records more and losing records less")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_torch_policy_from_jsonl(
        args.input,
        output_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        dropout=args.dropout,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        weight_decay=args.weight_decay,
        margin_weight=args.margin_weight,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
