import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout

from haggis.benchmark import benchmark_to_metrics, format_benchmark_summary, main, run_benchmark, write_benchmark_metrics


class BenchmarkTests(unittest.TestCase):
    def test_run_benchmark_returns_deterministic_schema(self):
        first = run_benchmark(bots=("random", "greedy"), states=2, seed=3)
        second = run_benchmark(bots=("random", "greedy"), states=2, seed=3)

        self.assertEqual(first.seeds, second.seeds)
        self.assertEqual([state.legal_moves for state in first.state_benchmarks], [state.legal_moves for state in second.state_benchmarks])
        self.assertEqual([bot.moves for bot in first.bot_benchmarks], [bot.moves for bot in second.bot_benchmarks])
        self.assertEqual(len(first.state_benchmarks), 2)
        self.assertEqual(len(first.bot_benchmarks), 2)
        self.assertTrue(all(state.legal_moves > 0 for state in first.state_benchmarks))
        self.assertTrue(all(bot.decisions == 2 for bot in first.bot_benchmarks))
        self.assertTrue(all(bot.average_seconds >= 0 for bot in first.bot_benchmarks))

    def test_benchmark_metrics_schema_includes_states_bots_and_config(self):
        result = run_benchmark(bots=("random", "greedy"), states=1, seed=4)
        metrics = benchmark_to_metrics(result, config={"seed": 4, "bots": ["random", "greedy"]})

        self.assertEqual(metrics["config"]["seed"], 4)
        self.assertEqual(metrics["config"]["bots"], ["random", "greedy"])
        self.assertEqual(metrics["seeds"], [4])
        self.assertEqual(len(metrics["states"]), 1)
        self.assertEqual(len(metrics["bots"]), 2)
        self.assertIn("legal_moves", metrics["states"][0])
        self.assertIn("legal_move_seconds", metrics["states"][0])
        self.assertIn("average_seconds", metrics["bots"][0])
        self.assertIn("moves", metrics["bots"][0])

    def test_write_benchmark_metrics_writes_json_file(self):
        result = run_benchmark(bots=("random",), states=1, seed=5)

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/benchmark.json"
            write_benchmark_metrics(result, output_path, config={"seed": 5})
            with open(output_path, encoding="utf-8") as file:
                metrics = json.load(file)

        self.assertEqual(metrics["config"]["seed"], 5)
        self.assertEqual(len(metrics["states"]), 1)
        self.assertEqual(len(metrics["bots"]), 1)

    def test_format_benchmark_summary_includes_core_stats(self):
        result = run_benchmark(bots=("random", "greedy"), states=1, seed=6)
        summary = format_benchmark_summary(result)

        self.assertIn("Haggis benchmark", summary)
        self.assertIn("States: 1", summary)
        self.assertIn("Legal moves:", summary)
        self.assertIn("random:", summary)
        self.assertIn("greedy:", summary)

    def test_benchmark_cli_prints_summary_and_writes_json(self):
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            output_path = f"{directory}/benchmark.json"
            with redirect_stdout(stdout):
                exit_code = main(["--bots", "random,greedy", "--states", "1", "--seed", "7", "--output-json", output_path])
            with open(output_path, encoding="utf-8") as file:
                metrics = json.load(file)

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis benchmark", stdout.getvalue())
        self.assertEqual(metrics["config"]["bots"], ["random", "greedy"])
        self.assertEqual(metrics["config"]["states"], 1)
        self.assertEqual(metrics["config"]["seed"], 7)
    def test_benchmark_cli_accepts_policy_model_for_policy_bots(self):
        from pathlib import Path

        from haggis.policy import LinearPolicy

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "policy.json"
            output_path = Path(directory) / "benchmark.json"
            LinearPolicy(weights={"action.card_count": 1.0}).save(model_path)

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--bots",
                        "policy,policy-rollout",
                        "--states",
                        "1",
                        "--seed",
                        "8",
                        "--policy-model",
                        str(model_path),
                        "--search-simulations",
                        "1",
                        "--search-root-moves",
                        "3",
                        "--search-rollout-turns",
                        "8",
                        "--output-json",
                        str(output_path),
                    ]
                )
            metrics = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(metrics["config"]["policy_model"], str(model_path))
        self.assertEqual([bot["bot"] for bot in metrics["bots"]], ["policy", "policy-rollout"])


if __name__ == "__main__":
    unittest.main()
