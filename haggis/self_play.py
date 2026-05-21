from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .combinations import Combination, CombinationType
from .engine import HaggisState, Move
from .tournament import Bot, _place_initial_bets, make_bot

JsonObject = dict[str, Any]


def export_self_play_jsonl(
    output_path: str | Path,
    *,
    bot_a: str,
    bot_b: str,
    hands: int,
    seed: int = 1,
    max_turns: int = 500,
    enable_betting: bool = True,
    observation_mode: str = "perfect",
    bot_a_policy_model: str | None = None,
    bot_b_policy_model: str | None = None,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
) -> int:
    """Write JSONL decision records from deterministic bot-vs-bot self-play.

    Returns the number of decision records written.
    """
    records = generate_self_play_records(
        bot_a=bot_a,
        bot_b=bot_b,
        hands=hands,
        seed=seed,
        max_turns=max_turns,
        enable_betting=enable_betting,
        observation_mode=observation_mode,
        bot_a_policy_model=bot_a_policy_model,
        bot_b_policy_model=bot_b_policy_model,
        search_simulations=search_simulations,
        search_root_moves=search_root_moves,
        search_rollout_turns=search_rollout_turns,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def generate_self_play_records(
    *,
    bot_a: str,
    bot_b: str,
    hands: int,
    seed: int = 1,
    max_turns: int = 500,
    enable_betting: bool = True,
    observation_mode: str = "perfect",
    bot_a_policy_model: str | None = None,
    bot_b_policy_model: str | None = None,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
) -> tuple[JsonObject, ...]:
    if hands < 1:
        raise ValueError("hands must be at least 1")
    _validate_observation_mode(observation_mode)

    search_config = {
        "search_simulations": search_simulations,
        "search_root_moves": search_root_moves,
        "search_rollout_turns": search_rollout_turns,
    }
    bots = (
        make_bot(
            bot_a,
            seed=seed * 2 + 1,
            policy_model=bot_a_policy_model,
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        ),
        make_bot(
            bot_b,
            seed=seed * 2 + 2,
            policy_model=bot_b_policy_model,
            search_simulations=search_simulations,
            search_root_moves=search_root_moves,
            search_rollout_turns=search_rollout_turns,
        ),
    )
    all_records: list[JsonObject] = []

    for hand_index in range(hands):
        hand_seed = seed + hand_index
        dealer = hand_index % 2
        all_records.extend(
            _play_hand_records(
                bots,
                bot_names=(bot_a, bot_b),
                hand_index=hand_index,
                seed=hand_seed,
                dealer=dealer,
                max_turns=max_turns,
                enable_betting=enable_betting,
                observation_mode=observation_mode,
                model_paths=(bot_a_policy_model, bot_b_policy_model),
                search_config=search_config,
            )
        )

    return tuple(all_records)


def _play_hand_records(
    bots: tuple[Bot, Bot],
    *,
    bot_names: tuple[str, str],
    hand_index: int,
    seed: int,
    dealer: int,
    max_turns: int,
    enable_betting: bool,
    observation_mode: str,
    model_paths: tuple[str | None, str | None],
    search_config: JsonObject,
) -> tuple[JsonObject, ...]:
    state = HaggisState.new_deal(seed=seed, dealer=dealer).assert_invariants(full_deck=True)
    if enable_betting:
        state = _place_initial_bets(state, bots).assert_invariants(full_deck=True)
    pending: list[JsonObject] = []
    turn_index = 0

    while state.hand_winner is None:
        if turn_index >= max_turns:
            raise RuntimeError(f"hand exceeded {max_turns} turns")

        player = state.current_player
        legal_moves = state.legal_moves()
        legal_set = set(legal_moves)
        move = bots[player].choose_move(state)
        if move not in legal_set:
            raise ValueError(f"bot {player} chose an illegal move: {move}")

        selected_index = legal_moves.index(move)
        pending.append(
            {
                "schema_version": 1,
                "bot_names": list(bot_names),
                "hand_index": hand_index,
                "hand_seed": seed,
                "dealer": dealer,
                "turn_index": turn_index,
                "acting_player": player,
                "observation_mode": observation_mode,
                "teacher": {
                    "bot": bot_names[player],
                    "model_path": model_paths[player],
                    "search": dict(search_config),
                },
                "dataset_source": "search_improved" if _is_search_teacher(bot_names[player], search_config) else "bot_policy",
                "state": _state_summary(state, perspective=player, observation_mode=observation_mode),
                "legal_actions": [_action_summary(index, legal_move) for index, legal_move in enumerate(legal_moves)],
                "selected_action_index": selected_index,
                "selected_action": _action_summary(selected_index, move),
            }
        )

        state = state.apply_move(move).assert_invariants(full_deck=True)
        turn_index += 1

    score = state.score_hand()
    outcome = {
        "winner": score.winner,
        "score": list(score.points),
        "turns": turn_index,
    }

    finalized = []
    for record in pending:
        actor = record["acting_player"]
        record = dict(record)
        record["outcome"] = {
            **outcome,
            "actor_won": actor == score.winner,
            "score_margin_for_actor": score.points[actor] - score.points[1 - actor],
        }
        finalized.append(record)

    return tuple(finalized)


def _is_search_teacher(bot_name: str, search_config: JsonObject) -> bool:
    if bot_name in {"policy-rollout", "tree-information-set", "information-set", "ucb-information-set", "monte-carlo"}:
        return True
    return any(value is not None for value in search_config.values())


def _state_summary(state: HaggisState, *, perspective: int, observation_mode: str) -> JsonObject:
    if observation_mode == "perfect":
        hands: list[list[str] | None] = [[_card_id(card) for card in hand] for hand in state.hands]
        haggis_points: int | None = sum(card.points for card in state.haggis)
    elif observation_mode == "player":
        opponent = 1 - perspective
        hands = [None, None]
        hands[perspective] = [_card_id(card) for card in state.hands[perspective]]
        hands[opponent] = None
        haggis_points = None
    else:
        raise ValueError(f"unknown observation_mode: {observation_mode!r}")

    return {
        "current_player": state.current_player,
        "hand_sizes": [len(state.hands[0]), len(state.hands[1])],
        "hands": hands,
        "haggis_count": len(state.haggis),
        "haggis_points": haggis_points,
        "captured_counts": [len(state.captured[0]), len(state.captured[1])],
        "captured_points": [sum(card.points for card in state.captured[0]), sum(card.points for card in state.captured[1])],
        "trick_cards": [_card_id(card) for card in state.trick_cards],
        "trick_points": sum(card.points for card in state.trick_cards),
        "last_player": state.last_player,
        "last_combination": _combination_summary(state.last_combination),
        "bets": list(state.bets),
        "has_played": list(state.has_played),
    }


def _action_summary(index: int, move: Move) -> JsonObject:
    return {
        "index": index,
        "is_pass": move.is_pass,
        "cards": [_card_id(card) for card in move.cards],
        "combination": _combination_summary(move.combination),
        "point_risk": sum(card.points for card in move.cards),
    }


def _combination_summary(combination: Combination | None) -> JsonObject | None:
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


def _card_id(card: object) -> str:
    return card.short_name()  # type: ignore[attr-defined]


def _validate_observation_mode(observation_mode: str) -> None:
    if observation_mode not in {"perfect", "player"}:
        raise ValueError("observation_mode must be 'perfect' or 'player'")


def summarize_self_play_jsonl(path: str | Path) -> JsonObject:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    observation_modes = Counter(record.get("observation_mode", "unknown") for record in records)
    bot_names = Counter(tuple(record.get("bot_names", ())) for record in records)
    dataset_sources = Counter(record.get("dataset_source", "unknown") for record in records)
    teachers = Counter(str(record.get("teacher", {}).get("bot", "unknown")) for record in records)
    actor_counts = Counter(str(record.get("acting_player")) for record in records)
    action_types = Counter(_selected_action_type(record) for record in records)
    winners = Counter(str(record.get("outcome", {}).get("winner")) for record in records)
    actor_wins = sum(1 for record in records if record.get("outcome", {}).get("actor_won"))
    bet_records = sum(1 for record in records if any(record.get("state", {}).get("bets", (0, 0))))

    return {
        "records": len(records),
        "observation_modes": dict(sorted(observation_modes.items())),
        "bot_names": {" vs ".join(names): count for names, count in sorted(bot_names.items())},
        "dataset_sources": dict(sorted(dataset_sources.items())),
        "teachers": dict(sorted(teachers.items())),
        "actor_counts": dict(sorted(actor_counts.items())),
        "selected_action_types": dict(sorted(action_types.items())),
        "outcomes": {
            "winner_counts": dict(sorted(winners.items())),
            "actor_win_rate": actor_wins / len(records) if records else 0.0,
        },
        "bets": {
            "records_with_bets": bet_records,
            "records_without_bets": len(records) - bet_records,
            "bet_record_rate": bet_records / len(records) if records else 0.0,
        },
    }


def format_summary(summary: JsonObject) -> str:
    return "\n".join(
        (
            "Haggis self-play dataset summary",
            f"Records: {summary['records']}",
            f"Observation modes: {summary['observation_modes']}",
            f"Bot names: {summary['bot_names']}",
            f"Dataset sources: {summary.get('dataset_sources', {})}",
            f"Teachers: {summary.get('teachers', {})}",
            f"Actor counts: {summary['actor_counts']}",
            f"Selected action types: {summary['selected_action_types']}",
            f"Winner counts: {summary['outcomes']['winner_counts']}",
            f"Actor win rate: {summary['outcomes']['actor_win_rate']:.3f}",
            f"Records with bets: {summary['bets']['records_with_bets']}",
        )
    )


def write_summary(summary: JsonObject, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_action_type(record: JsonObject) -> str:
    action = record.get("selected_action", {})
    if action.get("is_pass"):
        return "pass"
    combination = action.get("combination") or {}
    return str(combination.get("type", "unknown"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export or summarize Haggis self-play decision records")
    subparsers = parser.add_subparsers(dest="command")

    export = subparsers.add_parser("export", help="Export self-play decision records as JSONL")
    export.add_argument("--bot-a", default="point-aware")
    export.add_argument("--bot-b", default="bomb-control")
    export.add_argument("--hands", type=int, default=100)
    export.add_argument("--seed", type=int, default=1)
    export.add_argument("--max-turns", type=int, default=500)
    export.add_argument("--no-betting", action="store_true", help="Disable bot pre-play betting in exported records")
    export.add_argument("--observation-mode", choices=("perfect", "player"), default="perfect")
    export.add_argument("--bot-a-policy-model", help="Policy model path for bot A when using policy-based teachers")
    export.add_argument("--bot-b-policy-model", help="Policy model path for bot B when using policy-based teachers")
    export.add_argument("--policy-model", help="Convenience model path applied to both bots")
    export.add_argument("--search-simulations", type=int)
    export.add_argument("--search-root-moves", type=int)
    export.add_argument("--search-rollout-turns", type=int)
    export.add_argument("--output", required=True, help="Path to write JSONL records")

    summary = subparsers.add_parser("summary", help="Summarize a self-play JSONL dataset")
    summary.add_argument("--input", required=True, help="Self-play JSONL input path")
    summary.add_argument("--output-json", help="Optional path to write summary JSON")

    # Backward-compatible export flags on the root command.
    parser.add_argument("--bot-a", default="point-aware")
    parser.add_argument("--bot-b", default="bomb-control")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--no-betting", action="store_true", help="Disable bot pre-play betting in exported records")
    parser.add_argument("--observation-mode", choices=("perfect", "player"), default="perfect")
    parser.add_argument("--bot-a-policy-model", help="Policy model path for bot A when using policy-based teachers")
    parser.add_argument("--bot-b-policy-model", help="Policy model path for bot B when using policy-based teachers")
    parser.add_argument("--policy-model", help="Convenience model path applied to both bots")
    parser.add_argument("--search-simulations", type=int)
    parser.add_argument("--search-root-moves", type=int)
    parser.add_argument("--search-rollout-turns", type=int)
    parser.add_argument("--output", help="Path to write JSONL records")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summary":
        summary = summarize_self_play_jsonl(args.input)
        print(format_summary(summary))
        if args.output_json:
            write_summary(summary, args.output_json)
        return 0

    output = args.output
    if output is None:
        raise SystemExit("--output is required when exporting self-play records")
    model_a = args.bot_a_policy_model or args.policy_model
    model_b = args.bot_b_policy_model or args.policy_model
    count = export_self_play_jsonl(
        output,
        bot_a=args.bot_a,
        bot_b=args.bot_b,
        hands=args.hands,
        seed=args.seed,
        max_turns=args.max_turns,
        enable_betting=not args.no_betting,
        observation_mode=args.observation_mode,
        bot_a_policy_model=model_a,
        bot_b_policy_model=model_b,
        search_simulations=args.search_simulations,
        search_root_moves=args.search_root_moves,
        search_rollout_turns=args.search_rollout_turns,
    )
    print(f"Wrote {count} self-play decision records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
