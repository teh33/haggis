import json
import tempfile
import unittest
from pathlib import Path

from haggis.benchmark import main as benchmark_main
from haggis.engine import HaggisState
from haggis.ladder import run_ladder
from haggis.self_play import export_self_play_jsonl
from haggis.torch_policy import _record_sample_weight, train_torch_policy_from_jsonl
from haggis.tournament import make_bot

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchPolicyTests(unittest.TestCase):
    def test_train_torch_policy_writes_model_and_metrics(self):
        from haggis.torch_policy import load_torch_policy, main

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "torch_policy.pt"
            cli_model_path = Path(directory) / "torch_policy_cli.pt"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=31, observation_mode="player")

            result = train_torch_policy_from_jsonl(data_path, output_path=model_path, epochs=1, hidden_size=16, batch_size=4, dropout=0.0, validation_fraction=0.25, seed=3)
            exit_code = main([
                "--input", str(data_path),
                "--output", str(cli_model_path),
                "--epochs", "1",
                "--hidden-size", "16",
                "--batch-size", "4",
                "--dropout", "0.0",
                "--validation-fraction", "0.25",
                "--seed", "3",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(model_path.exists())
            self.assertTrue(cli_model_path.exists())

            state = HaggisState.new_deal(seed=44)
            loaded = load_torch_policy(model_path)
            self.assertIn(loaded.choose_move(state, state.legal_moves()), state.legal_moves())
            bot = make_bot("torch-policy", policy_model=str(model_path))
            self.assertIn(bot.choose_move(state), state.legal_moves())

            output_path = Path(directory) / "benchmark.json"
            with tempfile.TemporaryFile(mode="w+"):
                benchmark_exit_code = benchmark_main([
                    "--bots", "policy,torch-policy",
                    "--states", "1",
                    "--seed", "45",
                    "--policy-model", "models/linear_policy.json",
                    "--torch-policy-model", str(model_path),
                    "--output-json", str(output_path),
                ])
            self.assertEqual(benchmark_exit_code, 0)
            metrics = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["config"]["policy_model"], "models/linear_policy.json")
            self.assertEqual(metrics["config"]["torch_policy_model"], str(model_path))
            self.assertEqual([bot["bot"] for bot in metrics["bots"]], ["policy", "torch-policy"])

        self.assertGreater(result.records, 0)
        self.assertGreater(result.feature_count, 0)
        self.assertGreaterEqual(result.train_accuracy, 0.0)
        self.assertLessEqual(result.train_accuracy, 1.0)
        self.assertIsNotNone(result.validation_accuracy)

    def test_margin_weight_changes_record_sample_weight(self):
        winning = {"outcome": {"actor_won": True, "actor_score_margin": 350}}
        losing = {"outcome": {"actor_won": False, "actor_score_margin": -350}}

        self.assertEqual(_record_sample_weight(winning, margin_weight=0.0), 1.0)
        self.assertEqual(_record_sample_weight(winning, margin_weight=1.0), 2.0)
        self.assertEqual(_record_sample_weight(losing, margin_weight=1.0), 0.5)

    def test_ladder_runs_torch_policy_bot(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "torch_policy.pt"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=35, observation_mode="player")
            train_torch_policy_from_jsonl(data_path, output_path=model_path, epochs=1, hidden_size=8, batch_size=1, validation_fraction=0.25, seed=4)

            result = run_ladder(("torch-policy", "point-aware"), hands_per_match=1, seed=7, policy_model=str(model_path))

        self.assertEqual({entry.bot for entry in result.entries}, {"torch-policy", "point-aware"})
        self.assertEqual({match.bot_a for match in result.matches} | {match.bot_b for match in result.matches}, {"torch-policy", "point-aware"})


if __name__ == "__main__":
    unittest.main()
