import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout

from haggis.ladder import LadderEntry, format_ladder, ladder_to_metrics, main, run_ladder, write_ladder_metrics


class LadderTests(unittest.TestCase):
    def test_run_ladder_aggregates_round_robin_in_both_seat_orders(self):
        result = run_ladder(("random", "greedy"), hands_per_match=2, seed=3)

        self.assertEqual(len(result.matches), 2)
        self.assertEqual({match.bot_a for match in result.matches}, {"random", "greedy"})
        entries = {entry.bot: entry for entry in result.entries}
        self.assertEqual(entries["random"].matches, 2)
        self.assertEqual(entries["greedy"].matches, 2)
        self.assertEqual(entries["random"].hands, 4)
        self.assertEqual(entries["greedy"].hands, 4)
        self.assertEqual(entries["random"].hand_wins + entries["greedy"].hand_wins, 4)
        self.assertEqual(entries["random"].score_for, entries["greedy"].score_against)
        self.assertEqual(entries["greedy"].score_for, entries["random"].score_against)

    def test_ladder_is_deterministic_for_same_seed(self):
        first = run_ladder(("random", "greedy", "point-aware"), hands_per_match=1, seed=9)
        second = run_ladder(("random", "greedy", "point-aware"), hands_per_match=1, seed=9)

        self.assertEqual(first, second)

    def test_standings_sort_by_rating(self):
        result = run_ladder(("greedy", "point-aware"), hands_per_match=2, seed=4)
        ratings = [entry.rating for entry in result.standings]

        self.assertEqual(ratings, sorted(ratings, reverse=True))

    def test_ladder_entry_computed_properties(self):
        entry = LadderEntry(
            bot="x",
            rating=1510.0,
            matches=2,
            hands=4,
            hand_wins=3,
            hand_losses=1,
            score_for=100,
            score_against=70,
        )

        self.assertEqual(entry.score_margin, 30)
        self.assertEqual(entry.hand_win_rate, 0.75)

    def test_format_ladder_includes_ratings_and_scores(self):
        result = run_ladder(("random", "greedy"), hands_per_match=1, seed=5)
        output = format_ladder(result)

        self.assertIn("Haggis bot ladder", output)
        self.assertIn("rating", output)
        self.assertIn("random", output)
        self.assertIn("greedy", output)
        self.assertIn("Matches: 2", output)

    def test_ladder_cli_prints_table(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["--bots", "random,greedy", "--hands", "1", "--seed", "2"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis bot ladder", stdout.getvalue())
        self.assertIn("random", stdout.getvalue())
        self.assertIn("greedy", stdout.getvalue())

    def test_ladder_cli_accepts_search_budget_flags(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--bots",
                    "random,greedy",
                    "--hands",
                    "1",
                    "--seed",
                    "2",
                    "--search-simulations",
                    "1",
                    "--search-root-moves",
                    "2",
                    "--search-rollout-turns",
                    "20",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis bot ladder", stdout.getvalue())
        self.assertIn("random", stdout.getvalue())

    def test_ladder_metrics_schema_contains_standings_entries_matches_and_config(self):
        result = run_ladder(("random", "greedy"), hands_per_match=1, seed=7)
        metrics = ladder_to_metrics(result, config={"bots": ["random", "greedy"], "seed": 7})

        self.assertEqual(metrics["config"]["bots"], ["random", "greedy"])
        self.assertEqual(metrics["config"]["seed"], 7)
        self.assertEqual(len(metrics["standings"]), 2)
        self.assertEqual(len(metrics["entries"]), 2)
        self.assertEqual(len(metrics["matches"]), 2)
        self.assertIn("bot", metrics["standings"][0])
        self.assertIn("rating", metrics["standings"][0])
        self.assertIn("score_margin", metrics["standings"][0])
        self.assertIn("bot_a", metrics["matches"][0])
        self.assertIn("bot_b", metrics["matches"][0])
        self.assertIn("hand_wins", metrics["matches"][0])
        self.assertIn("score", metrics["matches"][0])

    def test_write_ladder_metrics_writes_json_file(self):
        result = run_ladder(("random", "greedy"), hands_per_match=1, seed=8)

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/ladder.json"
            write_ladder_metrics(result, output_path, config={"seed": 8})
            with open(output_path, encoding="utf-8") as file:
                metrics = json.load(file)

        self.assertEqual(metrics["config"]["seed"], 8)
        self.assertEqual(len(metrics["entries"]), 2)
        self.assertEqual(len(metrics["matches"]), 2)

    def test_ladder_cli_writes_json_metrics_when_requested(self):
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/ladder.json"
            with redirect_stdout(stdout):
                exit_code = main(["--bots", "random,greedy", "--hands", "1", "--seed", "9", "--output-json", output_path])
            with open(output_path, encoding="utf-8") as file:
                metrics = json.load(file)

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis bot ladder", stdout.getvalue())
        self.assertEqual(metrics["config"]["bots"], ["random", "greedy"])
        self.assertEqual(metrics["config"]["hands_per_match"], 1)
        self.assertEqual(metrics["config"]["seed"], 9)
        self.assertEqual(len(metrics["standings"]), 2)
        self.assertEqual(len(metrics["matches"]), 2)


if __name__ == "__main__":
    unittest.main()
