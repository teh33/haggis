from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .combinations import Combination, CombinationType
from .engine import HaggisState, Move

JsonObject = dict[str, Any]
FeatureVector = dict[str, float]

FEATURE_VERSION = 2


@dataclass(frozen=True)
class TrainingResult:
    examples: int
    updates: int
    epochs: int
    accuracy: float
    averaged: bool = False
    train_examples: int = 0
    validation_examples: int = 0
    train_accuracy: float = 0.0
    validation_accuracy: float | None = None


@dataclass(frozen=True)
class ValueTrainingResult:
    examples: int
    epochs: int
    updates: int
    mean_absolute_error: float
    train_examples: int = 0
    validation_examples: int = 0
    train_mean_absolute_error: float = 0.0
    validation_mean_absolute_error: float | None = None


@dataclass
class LinearValueModel:
    weights: dict[str, float] = field(default_factory=dict)
    feature_version: int = FEATURE_VERSION
    target: str = "actor_score_margin_normalized"
    feature_scale: str = "bounded_v1"

    def predict(self, features: FeatureVector) -> float:
        bounded = normalize_value_features(features)
        return sum(self.weights.get(name, 0.0) * value for name, value in bounded.items())

    def update(
        self,
        features: FeatureVector,
        target: float,
        learning_rate: float = 0.0001,
        l2: float = 0.0001,
    ) -> float:
        bounded = normalize_value_features(features)
        bounded_target = clamp_value_target(target)
        prediction = self.predict(bounded)
        error = bounded_target - prediction
        shrink = max(0.0, 1.0 - learning_rate * l2)
        for name in list(self.weights):
            self.weights[name] *= shrink
        for name, value in bounded.items():
            self.weights[name] = self.weights.get(name, 0.0) + learning_rate * error * value
        self._drop_zero_weights()
        return abs(error)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "linear_value_model",
            "feature_version": self.feature_version,
            "feature_scale": self.feature_scale,
            "target": self.target,
            "weights": dict(sorted(self.weights.items())),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearValueModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "linear_value_model":
            raise ValueError("not a Haggis linear value model")
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"unsupported feature version: {payload.get('feature_version')!r}")
        weights = {str(name): float(value) for name, value in payload.get("weights", {}).items()}
        return cls(
            weights=weights,
            feature_version=FEATURE_VERSION,
            target=str(payload.get("target", "actor_score_margin_normalized")),
            feature_scale=str(payload.get("feature_scale", "bounded_v1")),
        )

    def _drop_zero_weights(self) -> None:
        self.weights = {name: value for name, value in self.weights.items() if abs(value) > 1e-12}


@dataclass
class LinearPolicy:
    weights: dict[str, float] = field(default_factory=dict)
    feature_version: int = FEATURE_VERSION
    averaged: bool = False

    def score(self, features: FeatureVector) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in features.items())

    def choose_action_index(self, record: JsonObject) -> int:
        legal_actions = record["legal_actions"]
        scored = [
            (self.score(features_from_record_action(record, action)), -int(action["index"]), int(action["index"]))
            for action in legal_actions
        ]
        return max(scored)[2]

    def choose_move(self, state: HaggisState, moves: tuple[Move, ...]) -> Move:
        scored = [
            (self.score(features_from_state_action(state, move, action_index=index)), -index, move)
            for index, move in enumerate(moves)
        ]
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def update(self, positive: FeatureVector, negative: FeatureVector, learning_rate: float = 1.0) -> None:
        for name, value in positive.items():
            self.weights[name] = self.weights.get(name, 0.0) + learning_rate * value
        for name, value in negative.items():
            self.weights[name] = self.weights.get(name, 0.0) - learning_rate * value
        self._drop_zero_weights()

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "linear_action_ranker",
            "feature_version": self.feature_version,
            "averaged": self.averaged,
            "weights": dict(sorted(self.weights.items())),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "linear_action_ranker":
            raise ValueError("not a Haggis linear action-ranker model")
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"unsupported feature version: {payload.get('feature_version')!r}")
        weights = {str(name): float(value) for name, value in payload.get("weights", {}).items()}
        return cls(weights=weights, feature_version=FEATURE_VERSION, averaged=bool(payload.get("averaged", False)))

    def _drop_zero_weights(self) -> None:
        self.weights = {name: value for name, value in self.weights.items() if abs(value) > 1e-12}


def train_policy_from_jsonl(
    input_path: str | Path,
    *,
    epochs: int = 5,
    learning_rate: float = 1.0,
    only_winners: bool = False,
    averaged: bool = False,
    validation_fraction: float = 0.0,
) -> tuple[LinearPolicy, TrainingResult]:
    records = tuple(load_records_jsonl(input_path))
    if not records:
        raise ValueError("training data is empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    train_records, validation_records = split_train_validation(records, validation_fraction=validation_fraction)

    policy = LinearPolicy(averaged=averaged)
    best_weights = dict(policy.weights)
    best_accuracy = evaluate_policy_accuracy(policy, train_records, only_winners=only_winners)
    examples = 0
    updates = 0
    averaged_totals: dict[str, float] = {}
    averaged_steps = 0

    for _epoch in range(epochs):
        for record in train_records:
            if only_winners and not record.get("outcome", {}).get("actor_won", False):
                continue
            selected_index = int(record["selected_action_index"])
            predicted_index = policy.choose_action_index(record)
            examples += 1
            if predicted_index != selected_index:
                selected_action = record["legal_actions"][selected_index]
                predicted_action = record["legal_actions"][predicted_index]
                policy.update(
                    features_from_record_action(record, selected_action),
                    features_from_record_action(record, predicted_action),
                    learning_rate=learning_rate,
                )
                updates += 1

            if averaged:
                _accumulate_weights(averaged_totals, policy.weights)
                averaged_steps += 1

        if not averaged:
            epoch_accuracy = evaluate_policy_accuracy(policy, train_records, only_winners=only_winners)
            if epoch_accuracy >= best_accuracy:
                best_accuracy = epoch_accuracy
                best_weights = dict(policy.weights)

    if averaged:
        policy.weights = _average_weights(averaged_totals, averaged_steps)
        accuracy = evaluate_policy_accuracy(policy, train_records, only_winners=only_winners)
        validation_accuracy = evaluate_policy_accuracy(policy, validation_records, only_winners=only_winners) if validation_records else None
        return policy, TrainingResult(
            examples=examples,
            updates=updates,
            epochs=epochs,
            accuracy=accuracy,
            averaged=True,
            train_examples=_filtered_count(train_records, only_winners=only_winners),
            validation_examples=_filtered_count(validation_records, only_winners=only_winners),
            train_accuracy=accuracy,
            validation_accuracy=validation_accuracy,
        )

    policy.weights = best_weights
    validation_accuracy = evaluate_policy_accuracy(policy, validation_records, only_winners=only_winners) if validation_records else None
    return policy, TrainingResult(
        examples=examples,
        updates=updates,
        epochs=epochs,
        accuracy=best_accuracy,
        averaged=False,
        train_examples=_filtered_count(train_records, only_winners=only_winners),
        validation_examples=_filtered_count(validation_records, only_winners=only_winners),
        train_accuracy=best_accuracy,
        validation_accuracy=validation_accuracy,
    )


def train_value_model_from_jsonl(
    input_path: str | Path,
    *,
    epochs: int = 5,
    learning_rate: float = 0.0001,
    validation_fraction: float = 0.0,
    target: str = "actor_score_margin_normalized",
    l2: float = 0.0001,
) -> tuple[LinearValueModel, ValueTrainingResult]:
    records = tuple(load_records_jsonl(input_path))
    if not records:
        raise ValueError("training data is empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    train_records, validation_records = split_train_validation(records, validation_fraction=validation_fraction)

    model = LinearValueModel(target=target)
    examples = 0
    updates = 0
    for _epoch in range(epochs):
        for record in train_records:
            features = value_features_from_record(record)
            model.update(
                features,
                value_target_from_record(record, target=target),
                learning_rate=learning_rate,
                l2=l2,
            )
            examples += 1
            updates += 1

    train_mae = evaluate_value_mae(model, train_records, target=target)
    validation_mae = evaluate_value_mae(model, validation_records, target=target) if validation_records else None
    return model, ValueTrainingResult(
        examples=examples,
        epochs=epochs,
        updates=updates,
        mean_absolute_error=train_mae,
        train_examples=len(train_records),
        validation_examples=len(validation_records),
        train_mean_absolute_error=train_mae,
        validation_mean_absolute_error=validation_mae,
    )


def evaluate_value_mae(model: LinearValueModel, records: Iterable[JsonObject], *, target: str = "actor_score_margin_normalized") -> float:
    total_error = 0.0
    count = 0
    for record in records:
        total_error += abs(clamp_value_target(value_target_from_record(record, target=target)) - model.predict(value_features_from_record(record)))
        count += 1
    return total_error / count if count else 0.0


def value_target_from_record(record: JsonObject, *, target: str = "actor_score_margin_normalized") -> float:
    outcome = record.get("outcome", {})
    if target == "actor_win":
        return 1.0 if outcome.get("actor_won", False) else -1.0
    if target == "actor_score_margin_normalized":
        score_margin = outcome.get("actor_score_margin", outcome.get("score_margin_for_actor", 0.0))
        return float(score_margin) / 100.0
    raise ValueError(f"unsupported value target: {target}")


def value_features_from_record(record: JsonObject) -> FeatureVector:
    selected_index = int(record["selected_action_index"])
    action = record["legal_actions"][selected_index]
    return features_from_record_action(record, action)


def clamp_value_target(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def normalize_value_features(features: FeatureVector) -> FeatureVector:
    return {name: _bounded_feature_value(name, value) for name, value in features.items()}


def _bounded_feature_value(name: str, value: float) -> float:
    value = float(value)
    if name == "bias" or name.endswith(".pass") or name.endswith(".none") or ".type." in name:
        return value
    if name.endswith(".sheds_hand_fraction") or name.endswith(".point_fraction"):
        return max(0.0, min(1.0, value))
    if name.endswith(".rank"):
        return max(0.0, min(1.0, value / 13.0))
    if name.endswith(".bomb_rank"):
        return max(0.0, min(1.0, value / 6.0))
    if name.endswith(".sequence_width"):
        return max(0.0, min(1.0, value / 4.0))
    if name.endswith(".sequence_length"):
        return max(0.0, min(1.0, value / 9.0))
    if name.endswith(".index") or name.endswith(".neg_index"):
        return max(-1.0, min(1.0, value / 100.0))
    if name.endswith(".card_count") or name.endswith(".remaining_cards") or name.endswith(".actor_cards") or name.endswith(".opponent_cards") or name.endswith(".card_delta"):
        return max(-1.0, min(1.0, value / 17.0))
    if name.endswith(".bet") or name.endswith(".bet_delta"):
        return max(-1.0, min(1.0, value / 30.0))
    if "points" in name or "point_" in name:
        return max(-1.0, min(1.0, value / 100.0))
    return max(-1.0, min(1.0, value))

def split_train_validation(
    records: tuple[JsonObject, ...],
    *,
    validation_fraction: float,
) -> tuple[tuple[JsonObject, ...], tuple[JsonObject, ...]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    if validation_fraction == 0.0 or len(records) < 2:
        return records, ()

    validation_count = max(1, int(round(len(records) * validation_fraction)))
    validation_count = min(validation_count, len(records) - 1)
    split_at = len(records) - validation_count
    return records[:split_at], records[split_at:]


def _filtered_count(records: Iterable[JsonObject], *, only_winners: bool = False) -> int:
    return sum(1 for record in records if not only_winners or record.get("outcome", {}).get("actor_won", False))


def _accumulate_weights(totals: dict[str, float], weights: dict[str, float]) -> None:
    for name, value in weights.items():
        totals[name] = totals.get(name, 0.0) + value


def _average_weights(totals: dict[str, float], steps: int) -> dict[str, float]:
    if steps <= 0:
        return {}
    return {name: value / steps for name, value in totals.items() if abs(value / steps) > 1e-12}


def evaluate_policy_accuracy(
    policy: LinearPolicy,
    records: Iterable[JsonObject],
    *,
    only_winners: bool = False,
) -> float:
    correct = 0
    total = 0
    for record in records:
        if only_winners and not record.get("outcome", {}).get("actor_won", False):
            continue
        total += 1
        if policy.choose_action_index(record) == int(record["selected_action_index"]):
            correct += 1
    return correct / total if total else 0.0


def load_records_jsonl(path: str | Path) -> tuple[JsonObject, ...]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        _validate_record(record, line_number)
        records.append(record)
    return tuple(records)


def features_from_record_action(record: JsonObject, action: JsonObject) -> FeatureVector:
    state = record["state"]
    combination = action.get("combination")
    acting_player = int(record["acting_player"])
    opponent = 1 - acting_player
    hand_sizes = state["hand_sizes"]
    captured_points = state["captured_points"]
    last_combination = state.get("last_combination")

    actor_cards = float(hand_sizes[acting_player])
    opponent_cards = float(hand_sizes[opponent])
    action_cards = float(len(action.get("cards", [])))
    action_points = float(action.get("point_risk", 0))
    actor_hand_points = float(_visible_hand_points(state, acting_player))
    opponent_hand_points = float(_visible_hand_points(state, opponent))
    actor_captured_points = float(captured_points[acting_player])
    opponent_captured_points = float(captured_points[opponent])
    trick_points = float(state.get("trick_points", 0))
    bets = state.get("bets", [0, 0])
    actor_bet = float(bets[acting_player]) if len(bets) > acting_player else 0.0
    opponent_bet = float(bets[opponent]) if len(bets) > opponent else 0.0

    features: FeatureVector = {
        "bias": 1.0,
        "actor_is_player_0": _bool(acting_player == 0),
        "action.index": float(action.get("index", 0)),
        "action.neg_index": -float(action.get("index", 0)),
        "action.pass": _bool(action.get("is_pass", False)),
        "action.card_count": action_cards,
        "action.point_risk": action_points,
        "action.wild_count": float(_card_ids_wild_count(action.get("cards", []))),
        "action.empties_hand": _bool(action_cards == actor_cards),
        "action.remaining_cards": actor_cards - action_cards,
        "action.remaining_points": actor_hand_points - action_points,
        "action.sheds_hand_fraction": _safe_div(action_cards, actor_cards),
        "action.point_fraction": _safe_div(action_points, actor_hand_points),
        "state.actor_hand_points": actor_hand_points,
        "state.opponent_hand_points": opponent_hand_points,
        "state.hand_point_delta": opponent_hand_points - actor_hand_points,
        "state.actor_wild_count": float(_visible_hand_wild_count(state, acting_player)),
        "state.actor_cards": actor_cards,
        "state.opponent_cards": opponent_cards,
        "state.card_delta": opponent_cards - actor_cards,
        "state.actor_captured_points": actor_captured_points,
        "state.opponent_captured_points": opponent_captured_points,
        "state.captured_point_delta": actor_captured_points - opponent_captured_points,
        "state.trick_points": trick_points,
        "state.haggis_points": float(state.get("haggis_points") or 0),
        "state.responding": _bool(last_combination is not None),
        "state.leading": _bool(last_combination is None),
        "state.actor_bet": actor_bet,
        "state.opponent_bet": opponent_bet,
        "state.bet_delta": actor_bet - opponent_bet,
        "state.actor_has_played": _bool(_indexed_bool(state.get("has_played", []), acting_player)),
        "state.opponent_has_played": _bool(_indexed_bool(state.get("has_played", []), opponent)),
    }
    _add_combination_features(features, "action", combination)
    _add_combination_features(features, "last", last_combination)
    return features


def features_from_state_action(state: HaggisState, move: Move, *, action_index: int = 0) -> FeatureVector:
    player = state.current_player
    opponent = 1 - player
    actor_cards = float(len(state.hands[player]))
    opponent_cards = float(len(state.hands[opponent]))
    action_cards = float(len(move.cards))
    action_points = float(sum(card.points for card in move.cards))
    actor_hand_points = float(sum(card.points for card in state.hands[player]))
    opponent_hand_points = float(sum(card.points for card in state.hands[opponent]))
    actor_captured_points = float(sum(card.points for card in state.captured[player]))
    opponent_captured_points = float(sum(card.points for card in state.captured[opponent]))
    trick_points = float(sum(card.points for card in state.trick_cards))

    features: FeatureVector = {
        "bias": 1.0,
        "actor_is_player_0": _bool(player == 0),
        "action.index": float(action_index),
        "action.neg_index": -float(action_index),
        "action.pass": _bool(move.is_pass),
        "action.card_count": action_cards,
        "action.point_risk": action_points,
        "action.wild_count": float(sum(1 for card in move.cards if card.is_wild)),
        "action.empties_hand": _bool(action_cards == actor_cards),
        "action.remaining_cards": actor_cards - action_cards,
        "action.remaining_points": actor_hand_points - action_points,
        "action.sheds_hand_fraction": _safe_div(action_cards, actor_cards),
        "action.point_fraction": _safe_div(action_points, actor_hand_points),
        "state.actor_hand_points": actor_hand_points,
        "state.opponent_hand_points": opponent_hand_points,
        "state.hand_point_delta": opponent_hand_points - actor_hand_points,
        "state.actor_wild_count": float(sum(1 for card in state.hands[player] if card.is_wild)),
        "state.actor_cards": actor_cards,
        "state.opponent_cards": opponent_cards,
        "state.card_delta": opponent_cards - actor_cards,
        "state.actor_captured_points": actor_captured_points,
        "state.opponent_captured_points": opponent_captured_points,
        "state.captured_point_delta": actor_captured_points - opponent_captured_points,
        "state.trick_points": trick_points,
        "state.haggis_points": float(sum(card.points for card in state.haggis)),
        "state.responding": _bool(state.last_combination is not None),
        "state.leading": _bool(state.last_combination is None),
        "state.actor_bet": float(state.bets[player]),
        "state.opponent_bet": float(state.bets[opponent]),
        "state.bet_delta": float(state.bets[player] - state.bets[opponent]),
        "state.actor_has_played": _bool(state.has_played[player]),
        "state.opponent_has_played": _bool(state.has_played[opponent]),
    }
    _add_combination_features(features, "action", _combination_payload(move.combination))
    _add_combination_features(features, "last", _combination_payload(state.last_combination))
    return features


def _add_combination_features(features: FeatureVector, prefix: str, combination: JsonObject | None) -> None:
    if combination is None:
        features[f"{prefix}.none"] = 1.0
        return

    combo_type = str(combination.get("type"))
    features[f"{prefix}.type.{combo_type}"] = 1.0
    features[f"{prefix}.rank"] = float(combination.get("rank", 0))
    features[f"{prefix}.card_count"] = float(combination.get("card_count", 0))
    features[f"{prefix}.bomb_rank"] = float(combination.get("bomb_rank", 0))
    features[f"{prefix}.sequence_width"] = float(combination.get("sequence_width", 0))
    features[f"{prefix}.sequence_length"] = float(combination.get("sequence_length", 0))
    features[f"{prefix}.is_bomb"] = _bool(combination.get("is_bomb", False))


def _combination_payload(combination: Combination | None) -> JsonObject | None:
    if combination is None:
        return None
    return {
        "type": str(combination.type),
        "rank": combination.rank,
        "card_count": combination.card_count,
        "bomb_rank": combination.bomb_rank,
        "sequence_width": combination.sequence_width,
        "sequence_length": combination.sequence_length,
        "is_bomb": combination.type == CombinationType.BOMB,
    }


def _visible_hand_points(state: JsonObject, player: int) -> float:
    hand = state.get("hands", [None, None])[player]
    if not hand:
        return 0.0
    return float(sum(_card_id_points(card_id) for card_id in hand))


def _visible_hand_wild_count(state: JsonObject, player: int) -> int:
    hand = state.get("hands", [None, None])[player]
    if not hand:
        return 0
    return _card_ids_wild_count(hand)


def _card_ids_wild_count(card_ids: list[str]) -> int:
    return sum(1 for card_id in card_ids if str(card_id).endswith("*"))


def _card_id_points(card_id: str) -> int:
    core = card_id[:-1] if card_id.endswith("*") else card_id
    rank_label = core[:-1]
    if rank_label == "J":
        return 2
    if rank_label == "Q":
        return 3
    if rank_label == "K":
        return 5
    if rank_label in {"3", "5", "7", "9"}:
        return 1
    return 0


def _validate_record(record: JsonObject, line_number: int) -> None:
    required = ("state", "legal_actions", "selected_action_index", "acting_player", "outcome")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"line {line_number}: missing required keys: {', '.join(missing)}")
    selected = int(record["selected_action_index"])
    legal_count = len(record["legal_actions"])
    if not 0 <= selected < legal_count:
        raise ValueError(f"line {line_number}: selected_action_index out of range")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _indexed_bool(values: list[Any], index: int) -> bool:
    return bool(values[index]) if len(values) > index else False


def _bool(value: bool) -> float:
    return 1.0 if value else 0.0


def inspect_policy(policy: LinearPolicy, *, top: int = 10) -> dict:
    if top < 1:
        raise ValueError("top must be at least 1")
    positive = sorted(policy.weights.items(), key=lambda item: (item[1], item[0]), reverse=True)[:top]
    negative = sorted(policy.weights.items(), key=lambda item: (item[1], item[0]))[:top]
    return {
        "model_type": "linear_action_ranker",
        "feature_version": policy.feature_version,
        "averaged": policy.averaged,
        "weight_count": len(policy.weights),
        "top_positive": [{"feature": name, "weight": value} for name, value in positive],
        "top_negative": [{"feature": name, "weight": value} for name, value in negative],
        "groups": _weight_groups(policy.weights),
    }


def _weight_groups(weights: dict[str, float]) -> list[dict]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for feature, weight in weights.items():
        family = feature.split(".", 1)[0]
        grouped.setdefault(family, []).append((feature, weight))

    summaries = []
    for family, items in sorted(grouped.items()):
        positives = [(feature, weight) for feature, weight in items if weight > 0]
        negatives = [(feature, weight) for feature, weight in items if weight < 0]
        strongest_positive = max(positives, key=lambda item: (item[1], item[0]), default=None)
        strongest_negative = min(negatives, key=lambda item: (item[1], item[0]), default=None)
        summaries.append(
            {
                "family": family,
                "count": len(items),
                "positive_total": sum(weight for _feature, weight in positives),
                "negative_total": sum(weight for _feature, weight in negatives),
                "strongest_positive": _feature_weight(strongest_positive),
                "strongest_negative": _feature_weight(strongest_negative),
            }
        )
    return summaries


def _feature_weight(item: tuple[str, float] | None) -> dict | None:
    if item is None:
        return None
    feature, weight = item
    return {"feature": feature, "weight": weight}


def format_policy_inspection(inspection: dict) -> str:
    lines = [
        "Haggis linear policy inspection",
        f"Feature version: {inspection['feature_version']}",
        f"Averaged: {inspection['averaged']}",
        f"Weights: {inspection['weight_count']}",
        "Top positive weights:",
    ]
    for item in inspection["top_positive"]:
        lines.append(f"  {item['feature']}: {item['weight']:.6g}")
    lines.append("Top negative weights:")
    for item in inspection["top_negative"]:
        lines.append(f"  {item['feature']}: {item['weight']:.6g}")
    lines.append("Feature groups:")
    for group in inspection.get("groups", []):
        lines.append(
            f"  {group['family']}: count={group['count']} "
            f"positive_total={group['positive_total']:.6g} "
            f"negative_total={group['negative_total']:.6g}"
        )
        if group["strongest_positive"] is not None:
            item = group["strongest_positive"]
            lines.append(f"    strongest_positive={item['feature']} ({item['weight']:.6g})")
        if group["strongest_negative"] is not None:
            item = group["strongest_negative"]
            lines.append(f"    strongest_negative={item['feature']} ({item['weight']:.6g})")
    return "\n".join(lines)


def write_policy_inspection(inspection: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and inspect Haggis linear policy models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a linear imitation policy from self-play JSONL")
    train.add_argument("--input", required=True, help="Self-play JSONL input path")
    train.add_argument("--output", required=True, help="Model JSON output path")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=1.0)
    train.add_argument("--only-winners", action="store_true", help="Train only from decisions made by the hand winner")
    train.add_argument("--averaged", action="store_true", help="Use averaged perceptron weights")
    train.add_argument("--validation-fraction", type=float, default=0.0, help="Held-out fraction from the end of the JSONL records")
    train_value = subparsers.add_parser("train-value", help="Train a linear value model from self-play JSONL")
    train_value.add_argument("--input", required=True, help="Self-play JSONL input path")
    train_value.add_argument("--output", required=True, help="Value model JSON output path")
    train_value.add_argument("--epochs", type=int, default=5)
    train_value.add_argument("--learning-rate", type=float, default=0.0001)
    train_value.add_argument("--l2", type=float, default=0.0001, help="L2 regularization strength")
    train_value.add_argument("--validation-fraction", type=float, default=0.0, help="Held-out fraction from the end of the JSONL records")
    train_value.add_argument("--target", choices=("actor_score_margin_normalized", "actor_win"), default="actor_score_margin_normalized")
    inspect = subparsers.add_parser("inspect", help="Inspect top linear policy weights")
    inspect.add_argument("--model", required=True, help="Model JSON path")
    inspect.add_argument("--top", type=int, default=10, help="Number of positive/negative weights to show")
    inspect.add_argument("--output-json", help="Optional path to write inspection JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        policy, result = train_policy_from_jsonl(
            args.input,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            only_winners=args.only_winners,
            averaged=args.averaged,
            validation_fraction=args.validation_fraction,
        )
        policy.save(args.output)
        print(
            f"Trained linear policy on {result.examples} examples "
            f"for {result.epochs} epochs with {result.updates} updates; "
            f"averaged={result.averaged}; "
            f"accuracy={result.accuracy:.3f}; "
            f"validation_accuracy={result.validation_accuracy if result.validation_accuracy is not None else 'n/a'}; "
            f"wrote {args.output}"
        )
        return 0
    if args.command == "train-value":
        value_model, result = train_value_model_from_jsonl(
            args.input,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            target=args.target,
            l2=args.l2,
        )
        value_model.save(args.output)
        print(
            f"Trained linear value model on {result.examples} examples "
            f"for {result.epochs} epochs with {result.updates} updates; "
            f"target={args.target}; "
            f"mae={result.mean_absolute_error:.3f}; "
            f"validation_mae={result.validation_mean_absolute_error if result.validation_mean_absolute_error is not None else 'n/a'}; "
            f"wrote {args.output}"
        )
        return 0
    if args.command == "inspect":
        policy = LinearPolicy.load(args.model)
        inspection = inspect_policy(policy, top=args.top)
        print(format_policy_inspection(inspection))
        if args.output_json:
            write_policy_inspection(inspection, args.output_json)
        return 0
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
