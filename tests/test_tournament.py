import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout

from haggis.tournament import (
    BOT_TYPES,
    GameHandRecord,
    GameResult,
    HandResult,
    MatchResult,
    _next_dealer,
    format_game_summary,
    format_summary,
    game_to_metrics,
    main,
    make_bot,
    play_hand,
    run_game,
    run_match,
    tournament_to_metrics,
    write_metrics,
)
from haggis.bots import BombControlBot, EndgameSearchBot, GreedySheddingBot, InformationSetRolloutBot, MonteCarloRolloutBot, PointAwareBot, UCBInformationSetBot
from haggis import HaggisState
from haggis import tournament


class TournamentTests(unittest.TestCase):
    def test_play_hand_returns_score_and_activity_stats(self):
        result = play_hand((GreedySheddingBot(), GreedySheddingBot()), seed=1)

        self.assertIn(result.winner, (0, 1))
        self.assertEqual(len(result.score), 2)
        self.assertGreater(sum(result.score), 0)
        self.assertGreater(result.turns, 0)
        self.assertGreaterEqual(result.passes, 0)
        self.assertGreaterEqual(result.bombs, 0)
        self.assertIn(0, result.cards_remaining)

    def test_run_match_is_deterministic_for_seeded_bots_and_deals(self):
        first = run_match("random", "greedy", hands=5, seed=11)
        second = run_match("random", "greedy", hands=5, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(len(first.hands), 5)

    def test_match_result_aggregates_scores_and_counts(self):
        result = MatchResult(
            bot_names=("a", "b"),
            hands=(
                HandResult(winner=0, score=(10, 2), turns=3, passes=1, bombs=0, cards_remaining=(0, 2), bets=(15, 0)),
                HandResult(winner=1, score=(1, 12), turns=5, passes=2, bombs=1, cards_remaining=(3, 0), bets=(30, 15)),
            ),
        )

        self.assertEqual(result.hand_wins, (1, 1))
        self.assertEqual(result.total_score, (11, 14))
        self.assertEqual(result.score_margin, -3)
        self.assertEqual(result.total_turns, 8)
        self.assertEqual(result.total_passes, 3)
        self.assertEqual(result.total_bombs, 1)
        self.assertEqual(result.total_bets_placed, (2, 1))
        self.assertEqual(result.total_bets_succeeded, (1, 1))
        self.assertEqual(result.total_bets_failed, (1, 0))
        self.assertEqual(result.average_turns, 4.0)

    def test_format_summary_includes_core_tournament_stats(self):
        result = run_match("greedy", "greedy", hands=2, seed=3)
        summary = format_summary(result)

        self.assertIn("Haggis tournament: greedy vs greedy", summary)
        self.assertIn("Hands: 2", summary)
        self.assertIn("Hand wins:", summary)
        self.assertIn("Score:", summary)
        self.assertIn("Bombs played:", summary)
        self.assertIn("Bets placed:", summary)
        self.assertIn("Bets succeeded:", summary)

    def test_cli_prints_summary(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--bot-a", "random", "--bot-b", "greedy", "--hands", "3", "--seed", "1"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis tournament: random vs greedy", output.getvalue())
        self.assertIn("Hands: 3", output.getvalue())

    def test_tournament_registers_heuristic_baseline_bots(self):
        self.assertIs(BOT_TYPES["point-aware"], PointAwareBot)
        self.assertIs(BOT_TYPES["bomb-control"], BombControlBot)
        self.assertIs(BOT_TYPES["endgame-search"], EndgameSearchBot)
        self.assertIs(BOT_TYPES["information-set"], InformationSetRolloutBot)
        self.assertIs(BOT_TYPES["monte-carlo"], MonteCarloRolloutBot)
        self.assertIs(BOT_TYPES["ucb-information-set"], UCBInformationSetBot)

    def test_heuristic_baselines_can_play_a_match(self):
        result = run_match("point-aware", "bomb-control", hands=3, seed=5)

        self.assertEqual(result.bot_names, ("point-aware", "bomb-control"))
        self.assertEqual(len(result.hands), 3)
        self.assertGreater(sum(result.total_score), 0)

    def test_play_hand_places_initial_bot_bets_before_cards_are_played(self):
        class FixedBetBot(GreedySheddingBot):
            def __init__(self, amount):
                self.amount = amount

            def choose_bet(self, state, player):
                self.seen_has_played = state.has_played
                return self.amount

        bot_a = FixedBetBot(15)
        bot_b = FixedBetBot(30)

        result = play_hand((bot_a, bot_b), seed=1)

        self.assertEqual(result.bets, (15, 30))
        self.assertEqual(bot_a.seen_has_played, (False, False))
        self.assertEqual(bot_b.seen_has_played, (False, False))
        self.assertEqual(result.bet_stats.placed, (1, 1))
        self.assertEqual(sum(result.bet_stats.succeeded), 1)
        self.assertEqual(sum(result.bet_stats.failed), 1)

    def test_betting_can_be_disabled_for_compatibility(self):
        class FixedBetBot(GreedySheddingBot):
            def choose_bet(self, state, player):
                return 30

        result = play_hand((FixedBetBot(), FixedBetBot()), seed=1, enable_betting=False)

        self.assertEqual(result.bets, (0, 0))
        self.assertEqual(result.bet_stats.placed, (0, 0))

    def test_baseline_bot_betting_is_deterministic_and_valid(self):
        state = HaggisState.new_deal(seed=9)
        bots = [GreedySheddingBot(), PointAwareBot(), BombControlBot(), EndgameSearchBot()]

        for bot in bots:
            with self.subTest(bot=bot.__class__.__name__):
                first = bot.choose_bet(state, 0)
                second = bot.choose_bet(state, 0)
                self.assertEqual(first, second)
                self.assertIn(first, (0, 15, 30))

    def test_cli_can_disable_betting(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--bot-a", "greedy", "--bot-b", "bomb-control", "--hands", "1", "--seed", "1", "--no-betting"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Bets placed: 0 - 0", output.getvalue())

    def test_next_dealer_is_leader_or_last_winner_when_tied(self):
        self.assertEqual(_next_dealer((40, 10), last_hand_winner=1), 0)
        self.assertEqual(_next_dealer((10, 40), last_hand_winner=0), 1)
        self.assertEqual(_next_dealer((20, 20), last_hand_winner=1), 1)

    def test_run_game_accumulates_scores_and_dealer_progression(self):
        result = run_game("random", "greedy", target_score=80, seed=2, max_hands=10)

        self.assertGreaterEqual(max(result.total_score), 80)
        self.assertNotEqual(result.total_score[0], result.total_score[1])
        self.assertIn(result.winner, (0, 1))
        for index, record in enumerate(result.hand_records):
            expected = (
                sum(previous.hand.score[0] for previous in result.hand_records[: index + 1]),
                sum(previous.hand.score[1] for previous in result.hand_records[: index + 1]),
            )
            self.assertEqual(record.cumulative_score, expected)
        for previous, current in zip(result.hand_records, result.hand_records[1:]):
            self.assertEqual(current.dealer, _next_dealer(previous.cumulative_score, last_hand_winner=previous.hand.winner))

    def test_run_game_does_not_stop_on_tied_target_score_threshold(self):
        class ScriptedHandPlayer:
            def __init__(self, hands):
                self.hands = iter(hands)

            def __call__(self, *_args, **_kwargs):
                return next(self.hands)

        original_play_hand = tournament.play_hand
        tournament.play_hand = ScriptedHandPlayer(
            (
                HandResult(winner=0, score=(50, 50), turns=1, passes=0, bombs=0, cards_remaining=(0, 1)),
                HandResult(winner=1, score=(0, 10), turns=1, passes=0, bombs=0, cards_remaining=(1, 0)),
            )
        )
        try:
            result = run_game("random", "greedy", target_score=50, seed=1, max_hands=2)
        finally:
            tournament.play_hand = original_play_hand

        self.assertEqual(len(result.hands), 2)
        self.assertEqual(result.total_score, (50, 60))
        self.assertEqual(result.winner, 1)

    def test_format_game_summary_includes_target_score_and_winner(self):
        result = run_game("random", "greedy", target_score=60, seed=4, max_hands=10)
        summary = format_game_summary(result)

        self.assertIn("Haggis game: random vs greedy", summary)
        self.assertIn("Target score: 60", summary)
        self.assertIn("Winner:", summary)
        self.assertIn("Final score:", summary)

    def test_cli_can_run_target_score_game(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--bot-a", "random", "--bot-b", "greedy", "--target-score", "60", "--max-hands", "10", "--seed", "2"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis game: random vs greedy", output.getvalue())
        self.assertIn("Target score: 60", output.getvalue())

    def test_make_bot_passes_search_budget_to_rollout_bots(self):
        monte_carlo = make_bot(
            "monte-carlo",
            seed=1,
            search_simulations=5,
            search_root_moves=6,
            search_rollout_turns=70,
        )
        ucb = make_bot(
            "ucb-information-set",
            seed=1,
            search_simulations=7,
            search_root_moves=4,
            search_rollout_turns=80,
        )

        self.assertEqual(monte_carlo.simulations_per_move, 5)
        self.assertEqual(monte_carlo.max_root_moves, 6)
        self.assertEqual(monte_carlo.max_rollout_turns, 70)
        self.assertEqual(ucb.simulations, 7)
        self.assertEqual(ucb.max_root_moves, 4)
        self.assertEqual(ucb.max_rollout_turns, 80)

    def test_tournament_cli_accepts_search_budget_flags(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--bot-a",
                    "random",
                    "--bot-b",
                    "greedy",
                    "--hands",
                    "1",
                    "--seed",
                    "1",
                    "--search-simulations",
                    "1",
                    "--search-root-moves",
                    "2",
                    "--search-rollout-turns",
                    "20",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis tournament: random vs greedy", output.getvalue())

    def test_tournament_metrics_schema_contains_config_scores_bets_and_hands(self):
        result = run_match("random", "greedy", hands=2, seed=6)
        metrics = tournament_to_metrics(result, config={"seed": 6, "mode": "tournament"})

        self.assertEqual(metrics["type"], "tournament")
        self.assertEqual(metrics["config"]["seed"], 6)
        self.assertEqual(metrics["bot_names"], ["random", "greedy"])
        self.assertEqual(metrics["hands"], 2)
        self.assertIn("hand_wins", metrics)
        self.assertIn("score", metrics)
        self.assertIn("score_margin", metrics)
        self.assertIn("bets_placed", metrics)
        self.assertEqual(len(metrics["hand_records"]), 2)
        self.assertIn("bets", metrics["hand_records"][0])
        self.assertIn("cards_remaining", metrics["hand_records"][0])

    def test_game_metrics_schema_contains_target_score_and_per_hand_records(self):
        result = run_game("random", "greedy", target_score=60, max_hands=10, seed=6)
        metrics = game_to_metrics(result, config={"seed": 6, "mode": "game"})

        self.assertEqual(metrics["type"], "game")
        self.assertEqual(metrics["config"]["mode"], "game")
        self.assertEqual(metrics["bot_names"], ["random", "greedy"])
        self.assertEqual(metrics["target_score"], 60)
        self.assertIn(metrics["winner"], (0, 1))
        self.assertEqual(len(metrics["hand_records"]), len(result.hand_records))
        first = metrics["hand_records"][0]
        self.assertIn("dealer", first)
        self.assertIn("seed", first)
        self.assertIn("cumulative_score", first)
        self.assertIn("hand", first)
        self.assertIn("score", first["hand"])

    def test_write_metrics_writes_json_file(self):
        result = run_match("random", "greedy", hands=1, seed=7)
        metrics = tournament_to_metrics(result, config={"seed": 7})

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/tournament.json"
            write_metrics(metrics, output_path)
            with open(output_path, encoding="utf-8") as file:
                written = json.load(file)

        self.assertEqual(written["config"]["seed"], 7)
        self.assertEqual(written["type"], "tournament")
        self.assertEqual(len(written["hand_records"]), 1)

    def test_tournament_cli_writes_json_metrics(self):
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/tournament.json"
            with redirect_stdout(output):
                exit_code = main(["--bot-a", "random", "--bot-b", "greedy", "--hands", "1", "--seed", "8", "--output-json", output_path])
            with open(output_path, encoding="utf-8") as file:
                metrics = json.load(file)

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis tournament: random vs greedy", output.getvalue())
        self.assertEqual(metrics["type"], "tournament")
        self.assertEqual(metrics["config"]["mode"], "tournament")
        self.assertEqual(metrics["config"]["seed"], 8)
        self.assertEqual(metrics["bot_names"], ["random", "greedy"])

    def test_game_cli_writes_json_metrics(self):
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/game.json"
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--bot-a",
                        "random",
                        "--bot-b",
                        "greedy",
                        "--target-score",
                        "60",
                        "--max-hands",
                        "10",
                        "--seed",
                        "9",
                        "--output-json",
                        output_path,
                    ]
                )
            with open(output_path, encoding="utf-8") as file:
                metrics = json.load(file)

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis game: random vs greedy", output.getvalue())
        self.assertEqual(metrics["type"], "game")
        self.assertEqual(metrics["config"]["mode"], "game")
        self.assertEqual(metrics["config"]["target_score"], 60)
        self.assertEqual(metrics["bot_names"], ["random", "greedy"])
        self.assertGreaterEqual(len(metrics["hand_records"]), 1)


if __name__ == "__main__":
    unittest.main()
