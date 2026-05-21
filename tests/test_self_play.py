import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from haggis.self_play import (
    export_self_play_jsonl,
    format_summary,
    generate_self_play_records,
    main,
    summarize_self_play_jsonl,
)


class SelfPlayExportTests(unittest.TestCase):
    def test_generate_records_includes_schema_state_actions_and_outcome(self):
        records = generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=1, seed=7)

        self.assertGreater(len(records), 0)
        record = records[0]
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["observation_mode"], "perfect")
        self.assertEqual(record["dataset_source"], "bot_policy")
        self.assertEqual(record["teacher"]["bot"], record["bot_names"][record["acting_player"]])
        self.assertEqual(record["teacher"]["search"]["search_root_moves"], None)
        self.assertEqual(record["bot_names"], ["point-aware", "bomb-control"])
        self.assertEqual(record["hand_index"], 0)
        self.assertEqual(record["hand_seed"], 7)
        self.assertIn(record["acting_player"], (0, 1))
        self.assertIn("state", record)
        self.assertIn("hands", record["state"])
        self.assertIn("legal_actions", record)
        self.assertGreater(len(record["legal_actions"]), 0)
        self.assertEqual(record["selected_action"], record["legal_actions"][record["selected_action_index"]])
        self.assertIn("outcome", record)
        self.assertIn(record["outcome"]["winner"], (0, 1))
        self.assertIn("score_margin_for_actor", record["outcome"])

    def test_generate_records_supports_search_improved_policy_rollout_teachers(self):
        records = generate_self_play_records(
            bot_a="policy-rollout",
            bot_b="policy-rollout",
            hands=1,
            seed=7,
            observation_mode="player",
            bot_a_policy_model="models/linear_policy.json",
            bot_b_policy_model="models/linear_policy.json",
            search_root_moves=3,
            search_rollout_turns=8,
        )

        self.assertGreater(len(records), 0)
        record = records[0]
        self.assertEqual(record["dataset_source"], "search_improved")
        self.assertEqual(record["teacher"]["bot"], "policy-rollout")
        self.assertEqual(record["teacher"]["model_path"], "models/linear_policy.json")
        self.assertEqual(record["teacher"]["search"]["search_root_moves"], 3)
        self.assertEqual(record["teacher"]["search"]["search_rollout_turns"], 8)

    def test_generate_records_is_deterministic(self):
        first = generate_self_play_records(bot_a="random", bot_b="greedy", hands=2, seed=5)
        second = generate_self_play_records(bot_a="random", bot_b="greedy", hands=2, seed=5)

        self.assertEqual(first, second)

    def test_generate_records_includes_bot_bets_by_default(self):
        records = generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=1, seed=7)

        self.assertGreater(len(records), 0)
        self.assertTrue(any(record["state"]["bets"] != [0, 0] for record in records))

    def test_generate_records_can_disable_betting(self):
        records = generate_self_play_records(
            bot_a="point-aware",
            bot_b="bomb-control",
            hands=1,
            seed=7,
            enable_betting=False,
        )

        self.assertGreater(len(records), 0)
        self.assertTrue(all(record["state"]["bets"] == [0, 0] for record in records))

    def test_player_observation_mode_hides_opponent_hand_and_haggis_points(self):
        records = generate_self_play_records(
            bot_a="point-aware",
            bot_b="bomb-control",
            hands=1,
            seed=7,
            observation_mode="player",
        )

        self.assertGreater(len(records), 0)
        for record in records:
            actor = record["acting_player"]
            opponent = 1 - actor
            state = record["state"]
            self.assertEqual(record["observation_mode"], "player")
            self.assertIsInstance(state["hands"][actor], list)
            self.assertIsNone(state["hands"][opponent])
            self.assertEqual(len(state["hands"][actor]), state["hand_sizes"][actor])
            self.assertGreaterEqual(state["hand_sizes"][opponent], 0)
            self.assertIsNone(state["haggis_points"])
            self.assertIn("legal_actions", record)
            self.assertIn("outcome", record)

    def test_perfect_observation_mode_preserves_default_full_hands_and_haggis_points(self):
        records = generate_self_play_records(
            bot_a="point-aware",
            bot_b="bomb-control",
            hands=1,
            seed=7,
            observation_mode="perfect",
        )

        record = records[0]
        self.assertEqual(record["observation_mode"], "perfect")
        self.assertIsInstance(record["state"]["hands"][0], list)
        self.assertIsInstance(record["state"]["hands"][1], list)
        self.assertIsInstance(record["state"]["haggis_points"], int)

    def test_export_writes_jsonl_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "self_play.jsonl"

            count = export_self_play_jsonl(output, bot_a="point-aware", bot_b="bomb-control", hands=1, seed=3)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, len(lines))
        self.assertGreater(count, 0)
        parsed = [json.loads(line) for line in lines]
        self.assertEqual(parsed[0]["bot_names"], ["point-aware", "bomb-control"])
        self.assertIn("selected_action", parsed[0])
        self.assertIn("outcome", parsed[-1])

    def test_cli_writes_output_and_prints_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "export",
                        "--bot-a",
                        "point-aware",
                        "--bot-b",
                        "bomb-control",
                        "--hands",
                        "1",
                        "--seed",
                        "2",
                        "--output",
                        str(output),
                    ]
                )

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(exit_code, 0)
        self.assertGreater(len(lines), 0)
        self.assertIn("Wrote", stdout.getvalue())
        self.assertIn("self-play decision records", stdout.getvalue())

    def test_cli_still_supports_legacy_root_export_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"

            exit_code = main(["--bot-a", "point-aware", "--bot-b", "bomb-control", "--hands", "1", "--seed", "2", "--output", str(output)])

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(exit_code, 0)
        self.assertGreater(len(lines), 0)

    def test_cli_can_disable_betting_in_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"

            exit_code = main(
                [
                    "export",
                    "--bot-a",
                    "point-aware",
                    "--bot-b",
                    "bomb-control",
                    "--hands",
                    "1",
                    "--seed",
                    "2",
                    "--no-betting",
                    "--output",
                    str(output),
                ]
            )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(exit_code, 0)
        self.assertGreater(len(records), 0)
        self.assertTrue(all(record["state"]["bets"] == [0, 0] for record in records))

    def test_cli_can_export_player_observation_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"

            exit_code = main(
                [
                    "export",
                    "--bot-a",
                    "point-aware",
                    "--bot-b",
                    "bomb-control",
                    "--hands",
                    "1",
                    "--seed",
                    "2",
                    "--observation-mode",
                    "player",
                    "--output",
                    str(output),
                ]
            )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(exit_code, 0)
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["observation_mode"], "player")
        self.assertIsNone(records[0]["state"]["hands"][1 - records[0]["acting_player"]])
        self.assertIsNone(records[0]["state"]["haggis_points"])

    def test_cli_can_export_search_improved_player_observation_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"

            exit_code = main(
                [
                    "export",
                    "--bot-a",
                    "policy-rollout",
                    "--bot-b",
                    "policy-rollout",
                    "--hands",
                    "1",
                    "--seed",
                    "2",
                    "--observation-mode",
                    "player",
                    "--policy-model",
                    "models/linear_policy.json",
                    "--search-root-moves",
                    "3",
                    "--search-rollout-turns",
                    "8",
                    "--output",
                    str(output),
                ]
            )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(exit_code, 0)
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["dataset_source"], "search_improved")
        self.assertEqual(records[0]["teacher"]["model_path"], "models/linear_policy.json")
        self.assertEqual(records[0]["teacher"]["search"]["search_root_moves"], 3)
        self.assertEqual(records[0]["observation_mode"], "player")

    def test_summarize_self_play_jsonl_reports_dataset_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"
            export_self_play_jsonl(output, bot_a="point-aware", bot_b="bomb-control", hands=1, seed=3)

            summary = summarize_self_play_jsonl(output)

        self.assertGreater(summary["records"], 0)
        self.assertEqual(summary["observation_modes"], {"perfect": summary["records"]})
        self.assertEqual(summary["bot_names"], {"point-aware vs bomb-control": summary["records"]})
        self.assertEqual(summary["dataset_sources"], {"bot_policy": summary["records"]})
        self.assertEqual(sum(summary["teachers"].values()), summary["records"])
        self.assertIn("point-aware", summary["teachers"])
        self.assertIn("bomb-control", summary["teachers"])
        self.assertEqual(sum(summary["actor_counts"].values()), summary["records"])
        self.assertGreater(sum(summary["selected_action_types"].values()), 0)
        self.assertIn("winner_counts", summary["outcomes"])
        self.assertGreaterEqual(summary["outcomes"]["actor_win_rate"], 0.0)
        self.assertLessEqual(summary["outcomes"]["actor_win_rate"], 1.0)
        self.assertGreater(summary["bets"]["records_with_bets"], 0)

    def test_format_dataset_summary_includes_core_sections(self):
        summary = {
            "records": 2,
            "observation_modes": {"player": 2},
            "bot_names": {"a vs b": 2},
            "dataset_sources": {"search_improved": 2},
            "teachers": {"policy-rollout": 2},
            "actor_counts": {"0": 1, "1": 1},
            "selected_action_types": {"set": 2},
            "outcomes": {"winner_counts": {"0": 2}, "actor_win_rate": 0.5},
            "bets": {"records_with_bets": 2},
        }

        output = format_summary(summary)

        self.assertIn("Haggis self-play dataset summary", output)
        self.assertIn("Records: 2", output)
        self.assertIn("Observation modes:", output)
        self.assertIn("Dataset sources:", output)
        self.assertIn("Teachers:", output)
        self.assertIn("Selected action types:", output)
        self.assertIn("Actor win rate:", output)

    def test_summary_cli_prints_and_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.jsonl"
            summary_path = Path(directory) / "summary.json"
            export_self_play_jsonl(records_path, bot_a="point-aware", bot_b="bomb-control", hands=1, seed=4)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["summary", "--input", str(records_path), "--output-json", str(summary_path)])

            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis self-play dataset summary", stdout.getvalue())
        self.assertGreater(summary["records"], 0)
        self.assertIn("observation_modes", summary)
        self.assertIn("selected_action_types", summary)
        self.assertIn("bets", summary)


if __name__ == "__main__":
    unittest.main()
