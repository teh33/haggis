from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .tournament import BOT_TYPES, MatchResult, run_match


@dataclass(frozen=True)
class LadderBotSpec:
    label: str
    bot_name: str
    policy_model: str | None = None


@dataclass(frozen=True)
class LadderEntry:
    bot: str
    rating: float
    matches: int
    hands: int
    hand_wins: int
    hand_losses: int
    score_for: int
    score_against: int

    @property
    def score_margin(self) -> int:
        return self.score_for - self.score_against

    @property
    def hand_win_rate(self) -> float:
        return self.hand_wins / self.hands if self.hands else 0.0


@dataclass(frozen=True)
class LadderMatch:
    bot_a: str
    bot_b: str
    result: MatchResult
    rating_before: tuple[float, float]
    rating_after: tuple[float, float]


@dataclass(frozen=True)
class LadderResult:
    entries: tuple[LadderEntry, ...]
    matches: tuple[LadderMatch, ...]

    @property
    def standings(self) -> tuple[LadderEntry, ...]:
        return tuple(
            sorted(
                self.entries,
                key=lambda entry: (entry.rating, entry.score_margin, entry.hand_win_rate, entry.bot),
                reverse=True,
            )
        )


def run_ladder(
    bot_names: tuple[str, ...] | list[str],
    *,
    hands_per_match: int,
    seed: int = 1,
    max_turns: int = 500,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    policy_model: str | None = None,
    torch_policy_model: str | None = None,
    torch_bet_model: str | None = None,
    search_simulations: int | None = None,
    search_root_moves: int | None = None,
    search_rollout_turns: int | None = None,
) -> LadderResult:
    specs = tuple(_parse_bot_spec(bot, policy_model=policy_model, torch_policy_model=torch_policy_model) for bot in bot_names)
    labels = tuple(spec.label for spec in specs)
    unknown = sorted({spec.bot_name for spec in specs} - set(BOT_TYPES))
    if unknown:
        choices = ", ".join(sorted(BOT_TYPES))
        raise ValueError(f"unknown bot(s): {', '.join(unknown)}; expected one of: {choices}")
    if len(set(labels)) != len(labels):
        raise ValueError("bot labels must be unique")
    if hands_per_match < 1:
        raise ValueError("hands_per_match must be at least 1")

    ratings = {label: initial_rating for label in labels}
    stats = {
        label: {
            "matches": 0,
            "hands": 0,
            "hand_wins": 0,
            "hand_losses": 0,
            "score_for": 0,
            "score_against": 0,
        }
        for label in labels
    }
    matches: list[LadderMatch] = []
    match_index = 0

    for left_index, spec_left in enumerate(specs):
        for spec_right in specs[left_index + 1 :]:
            for spec_a, spec_b in ((spec_left, spec_right), (spec_right, spec_left)):
                bot_a = spec_a.label
                bot_b = spec_b.label
                match_seed = seed + match_index * 1009
                before = (ratings[bot_a], ratings[bot_b])
                result = run_match(
                    spec_a.bot_name,
                    spec_b.bot_name,
                    hands=hands_per_match,
                    seed=match_seed,
                    max_turns=max_turns,
                    policy_model=policy_model,
                    bot_a_policy_model=spec_a.policy_model,
                    bot_b_policy_model=spec_b.policy_model,
                    torch_bet_model=torch_bet_model,
                    search_simulations=search_simulations,
                    search_root_moves=search_root_moves,
                    search_rollout_turns=search_rollout_turns,
                )
                _update_stats(stats, bot_a, bot_b, result)
                ratings[bot_a], ratings[bot_b] = _update_elo(
                    ratings[bot_a],
                    ratings[bot_b],
                    _match_score(result.hand_wins),
                    k_factor=k_factor,
                )
                after = (ratings[bot_a], ratings[bot_b])
                matches.append(LadderMatch(bot_a=bot_a, bot_b=bot_b, result=result, rating_before=before, rating_after=after))
                match_index += 1

    entries = tuple(
        LadderEntry(
            bot=label,
            rating=ratings[label],
            matches=stats[label]["matches"],
            hands=stats[label]["hands"],
            hand_wins=stats[label]["hand_wins"],
            hand_losses=stats[label]["hand_losses"],
            score_for=stats[label]["score_for"],
            score_against=stats[label]["score_against"],
        )
        for label in labels
    )
    return LadderResult(entries=entries, matches=tuple(matches))


def format_ladder(result: LadderResult) -> str:
    lines = [
        "Haggis bot ladder",
        "rank bot              rating  matches hands win%   score margin",
    ]
    for rank, entry in enumerate(result.standings, start=1):
        lines.append(
            f"{rank:>4} {entry.bot:<16} {entry.rating:>7.1f} {entry.matches:>8} "
            f"{entry.hands:>5} {entry.hand_win_rate:>5.1%} {entry.score_for:>6}-{entry.score_against:<6} "
            f"{entry.score_margin:+d}"
        )
    lines.append(f"Matches: {len(result.matches)}")
    return "\n".join(lines)


def ladder_to_metrics(result: LadderResult, *, config: dict | None = None) -> dict:
    return {
        "config": config or {},
        "rating_system": {
            "name": "elo",
            "initial_rating": (config or {}).get("initial_rating", 1500.0),
            "k_factor": (config or {}).get("k_factor", 32.0),
            "score_basis": "hand_win_rate_per_ordered_match",
        },
        "standings": [_entry_to_dict(entry) for entry in result.standings],
        "entries": [_entry_to_dict(entry) for entry in result.entries],
        "matches": [_match_to_dict(match) for match in result.matches],
    }


def write_ladder_metrics(result: LadderResult, path: str | Path, *, config: dict | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ladder_to_metrics(result, config=config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _entry_to_dict(entry: LadderEntry) -> dict:
    return {
        "bot": entry.bot,
        "rating": entry.rating,
        "matches": entry.matches,
        "hands": entry.hands,
        "hand_wins": entry.hand_wins,
        "hand_losses": entry.hand_losses,
        "hand_win_rate": entry.hand_win_rate,
        "score_for": entry.score_for,
        "score_against": entry.score_against,
        "score_margin": entry.score_margin,
    }


def _match_to_dict(match: LadderMatch) -> dict:
    return {
        "bot_a": match.bot_a,
        "bot_b": match.bot_b,
        "rating_before": list(match.rating_before),
        "rating_after": list(match.rating_after),
        "hands": len(match.result.hands),
        "hand_wins": list(match.result.hand_wins),
        "score": list(match.result.total_score),
        "score_margin": match.result.score_margin,
        "average_turns": match.result.average_turns,
        "bets_placed": list(match.result.total_bets_placed),
        "bets_succeeded": list(match.result.total_bets_succeeded),
        "bets_failed": list(match.result.total_bets_failed),
    }


def _update_stats(stats: dict[str, dict[str, int]], bot_a: str, bot_b: str, result: MatchResult) -> None:
    wins = result.hand_wins
    scores = result.total_score
    hands = len(result.hands)

    stats[bot_a]["matches"] += 1
    stats[bot_a]["hands"] += hands
    stats[bot_a]["hand_wins"] += wins[0]
    stats[bot_a]["hand_losses"] += wins[1]
    stats[bot_a]["score_for"] += scores[0]
    stats[bot_a]["score_against"] += scores[1]

    stats[bot_b]["matches"] += 1
    stats[bot_b]["hands"] += hands
    stats[bot_b]["hand_wins"] += wins[1]
    stats[bot_b]["hand_losses"] += wins[0]
    stats[bot_b]["score_for"] += scores[1]
    stats[bot_b]["score_against"] += scores[0]


def _match_score(hand_wins: tuple[int, int]) -> float:
    total = hand_wins[0] + hand_wins[1]
    if total == 0:
        return 0.5
    return hand_wins[0] / total


def _update_elo(rating_a: float, rating_b: float, score_a: float, *, k_factor: float) -> tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    delta = k_factor * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


def _parse_bot_spec(spec: str, *, policy_model: str | None, torch_policy_model: str | None) -> LadderBotSpec:
    label_model_split = spec.split("@", 1)
    label_and_bot = label_model_split[0]
    explicit_model = label_model_split[1] if len(label_model_split) == 2 else None
    if ":" in label_and_bot:
        label, bot_name = label_and_bot.split(":", 1)
    else:
        label = bot_name = label_and_bot
    if not label or not bot_name:
        raise ValueError(f"invalid bot spec: {spec!r}")
    model = explicit_model
    if model is None:
        model = _model_for_named_bot(bot_name, policy_model=policy_model, torch_policy_model=torch_policy_model)
    return LadderBotSpec(label=label, bot_name=bot_name, policy_model=model)


def _model_for_named_bot(bot_name: str, *, policy_model: str | None, torch_policy_model: str | None) -> str | None:
    if bot_name == "torch-policy":
        return torch_policy_model or policy_model
    return policy_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic round-robin Haggis bot ladder")
    parser.add_argument(
        "--bots",
        default="random,greedy,point-aware,bomb-control",
        help="Comma-separated bot names/specs. Specs may be name, label:name, or label:name@model_path. Available: " + ", ".join(sorted(BOT_TYPES)),
    )
    parser.add_argument("--hands", type=int, default=20, help="Hands per ordered matchup")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--k-factor", type=float, default=32.0)
    parser.add_argument("--policy-model", default="models/linear_policy.json", help="Model path when using policy or policy-rollout bots")
    parser.add_argument("--torch-policy-model", help="Model path when using torch-policy bots")
    parser.add_argument("--torch-bet-model", help="Optional bet model path when using torch-policy bots")
    parser.add_argument("--search-simulations", type=int, help="Simulation budget for rollout/search bots")
    parser.add_argument("--search-root-moves", type=int, help="Maximum root moves considered by rollout/search bots")
    parser.add_argument("--search-rollout-turns", type=int, help="Maximum turns per rollout for rollout/search bots")
    parser.add_argument("--output-json", help="Optional path to write machine-readable ladder metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bot_names = tuple(bot.strip() for bot in args.bots.split(",") if bot.strip())
    result = run_ladder(
        bot_names,
        hands_per_match=args.hands,
        seed=args.seed,
        max_turns=args.max_turns,
        k_factor=args.k_factor,
        policy_model=args.policy_model,
        torch_policy_model=args.torch_policy_model,
        torch_bet_model=args.torch_bet_model,
        search_simulations=args.search_simulations,
        search_root_moves=args.search_root_moves,
        search_rollout_turns=args.search_rollout_turns,
    )
    print(format_ladder(result))
    if args.output_json:
        write_ladder_metrics(
            result,
            args.output_json,
            config={
                "bots": list(bot_names),
                "hands_per_match": args.hands,
                "seed": args.seed,
                "max_turns": args.max_turns,
                "initial_rating": 1500.0,
                "k_factor": args.k_factor,
                "policy_model": args.policy_model,
                "torch_policy_model": args.torch_policy_model,
                "torch_bet_model": args.torch_bet_model,
                "search_simulations": args.search_simulations,
                "search_root_moves": args.search_root_moves,
                "search_rollout_turns": args.search_rollout_turns,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
