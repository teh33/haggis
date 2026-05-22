from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cards import Card, Rank, Suit
from .policy import load_records_jsonl
from .search import bet_amount_for_hand

JsonObject = dict[str, Any]


try:
    import torch as _torch_import
    _TorchModuleBase = _torch_import.nn.Module
except ModuleNotFoundError:
    _torch_import = None
    _TorchModuleBase = object


@dataclass(frozen=True)
class BetTrainingResult:
    records: int
    train_records: int
    validation_records: int
    epochs: int
    train_accuracy: float
    validation_accuracy: float | None
    feature_count: int


class TorchBetClassifier(_TorchModuleBase):
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
            torch.nn.Linear(hidden_size, 3),
        )

    def forward(self, features):
        return self.layers(features)


def train_torch_bet_model_from_jsonl(
    input_path: str | Path,
    *,
    output_path: str | Path,
    epochs: int = 5,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    batch_size: int = 32,
    dropout: float = 0.0,
    validation_fraction: float = 0.2,
    seed: int = 1,
    weight_decay: float = 0.0001,
) -> BetTrainingResult:
    torch = _torch()
    records = _bet_records(load_records_jsonl(input_path))
    if not records:
        raise ValueError("training data contains no first-play bet records")
    train_records, validation_records = _split(records, validation_fraction=validation_fraction)
    feature_names = _feature_names(train_records)
    model = TorchBetClassifier(len(feature_names), hidden_size, dropout=dropout)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_batches = [(_feature_tensor(record, feature_names), torch.tensor(_target_index(record["target"]), dtype=torch.long)) for record in train_records]
    for epoch in range(epochs):
        shuffled = list(train_batches)
        random.Random(seed + epoch).shuffle(shuffled)
        model.train()
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            features = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            loss = loss_fn(model(features), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_accuracy = _accuracy(model, train_records, feature_names)
    validation_accuracy = _accuracy(model, validation_records, feature_names) if validation_records else None
    _save(output_path, model=model, feature_names=feature_names, hidden_size=hidden_size, config={
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "dropout": dropout,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "weight_decay": weight_decay,
    }, metrics={
        "records": len(records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
    })
    return BetTrainingResult(len(records), len(train_records), len(validation_records), epochs, train_accuracy, validation_accuracy, len(feature_names))


def _bet_records(records) -> list[JsonObject]:
    output = []
    for record in records:
        player = int(record["acting_player"])
        state = record["state"]
        if state.get("has_played", [False, False])[player]:
            continue
        hand_ids = state.get("hands", [None, None])[player]
        if not hand_ids:
            continue
        hand = tuple(_parse_card_id(card_id) for card_id in hand_ids)
        target = bet_amount_for_hand(hand, aggression=1)
        output.append({"features": bet_features_from_record(record), "target": target})
    return output


def bet_features_from_record(record: JsonObject) -> dict[str, float]:
    player = int(record["acting_player"])
    state = record["state"]
    hand = tuple(_parse_card_id(card_id) for card_id in (state.get("hands", [(), ()])[player] or ()))
    opponent = 1 - player
    return bet_features_from_hand(
        hand,
        own_score=float(state.get("captured_points", [0, 0])[player]),
        opponent_score=float(state.get("captured_points", [0, 0])[opponent]),
        opponent_has_bet=float(state.get("bets", [0, 0])[opponent] > 0),
        opponent_bet=float(state.get("bets", [0, 0])[opponent]),
    )


def bet_features_from_hand(
    hand: tuple[Card, ...],
    *,
    own_score: float = 0.0,
    opponent_score: float = 0.0,
    opponent_has_bet: float = 0.0,
    opponent_bet: float = 0.0,
) -> dict[str, float]:
    ranks = [int(card.rank) for card in hand if not card.is_wild]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    suits = [str(card.suit) for card in hand if not card.is_wild]
    return {
        "bias": 1.0,
        "cards": len(hand) / 17.0,
        "points": sum(card.points for card in hand) / 100.0,
        "wilds": sum(card.is_wild for card in hand) / 3.0,
        "high_cards": sum(rank >= 9 for rank in ranks) / 14.0,
        "pairs": sum(count >= 2 for count in rank_counts.values()) / 7.0,
        "triples": sum(count >= 3 for count in rank_counts.values()) / 4.0,
        "max_same_rank": max(rank_counts.values(), default=0) / 4.0,
        "suit_concentration": max((suits.count(suit) for suit in set(suits)), default=0) / 14.0,
        "score_delta": max(-1.0, min(1.0, (own_score - opponent_score) / 350.0)),
        "opponent_has_bet": opponent_has_bet,
        "opponent_bet": opponent_bet / 30.0,
    }


def load_torch_bet_model(path: str | Path) -> "TorchBetPolicy":
    torch = _torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("model_type") != "torch_bet_classifier":
        raise ValueError("not a Haggis torch bet model")
    feature_names = tuple(str(name) for name in payload["feature_names"])
    hidden_size = int(payload.get("hidden_size", payload.get("config", {}).get("hidden_size", 64)))
    dropout = float(payload.get("config", {}).get("dropout", 0.0))
    model = TorchBetClassifier(len(feature_names), hidden_size, dropout=dropout)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return TorchBetPolicy(model=model, feature_names=feature_names)


@dataclass
class TorchBetPolicy:
    model: Any
    feature_names: tuple[str, ...]

    def choose_bet_from_hand(self, hand: tuple[Card, ...], *, own_score: float = 0.0, opponent_score: float = 0.0, opponent_has_bet: float = 0.0, opponent_bet: float = 0.0) -> int:
        torch = _torch()
        features = bet_features_from_hand(hand, own_score=own_score, opponent_score=opponent_score, opponent_has_bet=opponent_has_bet, opponent_bet=opponent_bet)
        row = torch.tensor([[float(features.get(name, 0.0)) for name in self.feature_names]], dtype=torch.float32)
        with torch.no_grad():
            index = int(torch.argmax(self.model(row)).item())
        return (0, 15, 30)[index]


def _parse_card_id(card_id: str) -> Card:
    is_wild = card_id.endswith("*")
    core = card_id[:-1] if is_wild else card_id
    rank_label = core[:-1]
    suit_symbol = core[-1]
    rank = {"J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING}[rank_label] if rank_label in {"J", "Q", "K"} else Rank(int(rank_label))
    suit = {"♣": Suit.CLUBS, "♦": Suit.DIAMONDS, "♥": Suit.HEARTS, "♠": Suit.SPADES}[suit_symbol]
    return Card(rank, suit, is_wild)


def _feature_names(records: list[JsonObject]) -> tuple[str, ...]:
    names = set()
    for record in records:
        names.update(record["features"])
    return tuple(sorted(names))


def _feature_tensor(record: JsonObject, feature_names: tuple[str, ...]):
    torch = _torch()
    return torch.tensor([float(record["features"].get(name, 0.0)) for name in feature_names], dtype=torch.float32)


def _target_index(target: int) -> int:
    return {0: 0, 15: 1, 30: 2}[int(target)]


def _accuracy(model, records: list[JsonObject], feature_names: tuple[str, ...]) -> float:
    torch = _torch()
    if not records:
        return 0.0
    correct = 0
    with torch.no_grad():
        model.eval()
        for record in records:
            if int(torch.argmax(model(_feature_tensor(record, feature_names).unsqueeze(0))).item()) == _target_index(record["target"]):
                correct += 1
    return correct / len(records)


def _split(records: list[JsonObject], *, validation_fraction: float) -> tuple[list[JsonObject], list[JsonObject]]:
    if validation_fraction <= 0.0:
        return records, []
    split = max(1, min(len(records) - 1, int(len(records) * (1.0 - validation_fraction))))
    return records[:split], records[split:]


def _save(path: str | Path, *, model, feature_names: tuple[str, ...], hidden_size: int, config: JsonObject, metrics: JsonObject) -> None:
    torch = _torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "torch_bet_classifier", "feature_names": list(feature_names), "hidden_size": hidden_size, "state_dict": model.state_dict(), "config": config, "metrics": metrics}, output)


def _torch():
    if _torch_import is None:
        raise ModuleNotFoundError("PyTorch is required for haggis.torch_bet; install with `python -m pip install torch`")
    return _torch_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train experimental PyTorch Haggis bet models")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_torch_bet_model_from_jsonl(
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
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
