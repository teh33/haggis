import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from haggis.experiment import format_experiment_summary, main, run_policy_experiment


class ExperimentTests(unittest.TestCase):
    def test_run_policy_experiment_writes_artifacts_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_policy_experiment(
                directory,
                teacher_a="point-aware",
                teacher_b="bomb-control",
                data_hands=1,
                epochs=2,
                seed=3,
                eval_hands=1,
                eval_opponents=("greedy",),
                observation_mode="player",
                averaged=True,
                validation_fraction=0.25,
                ladder_hands=1,
                evaluate_policy_rollout=True,
                rollout_simulations=1,
                rollout_root_moves=3,
                rollout_turns=10,
            )

            data_lines = result.artifacts.data_path.read_text(encoding="utf-8").splitlines()
            model = json.loads(result.artifacts.model_path.read_text(encoding="utf-8"))
            metrics = json.loads(result.artifacts.metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))
            ladder = json.loads(result.artifacts.ladder_path.read_text(encoding="utf-8"))

        self.assertGreater(len(data_lines), 0)
        self.assertEqual(result.records, len(data_lines))
        self.assertEqual(model["model_type"], "linear_action_ranker")
        self.assertEqual(metrics["records"], result.records)
        self.assertIn("greedy", metrics["evaluation"])
        self.assertIn("greedy", metrics["policy_rollout_evaluation"])
        self.assertEqual(manifest["experiment"], "linear_policy_imitation")
        self.assertEqual(manifest["config"]["teacher_a"], "point-aware")
        self.assertEqual(manifest["config"]["observation_mode"], "player")
        self.assertTrue(manifest["config"]["averaged"])
        self.assertTrue(model["averaged"])
        self.assertTrue(metrics["training"]["averaged"])
        self.assertGreater(metrics["training"]["train_examples"], 0)
        self.assertGreater(metrics["training"]["validation_examples"], 0)
        self.assertIsNotNone(metrics["training"]["validation_accuracy"])
        self.assertEqual(metrics["ladder"], str(result.artifacts.ladder_path))
        self.assertEqual(manifest["config"]["validation_fraction"], 0.25)
        self.assertIn("training_validation_accuracy", manifest["summary"])
        self.assertEqual(manifest["artifacts"]["data"], str(result.artifacts.data_path))
        self.assertEqual(manifest["artifacts"]["ladder"], str(result.artifacts.ladder_path))
        self.assertTrue(manifest["summary"]["ladder_ran"])
        self.assertEqual(ladder["config"]["bots"], ["policy", "policy-rollout", "greedy"])
        self.assertEqual(ladder["config"]["hands_per_match"], 1)
        self.assertEqual(len(ladder["standings"]), 3)
        self.assertEqual(json.loads(data_lines[0])["observation_mode"], "player")

    def test_format_experiment_summary_lists_artifacts_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_policy_experiment(
                directory,
                data_hands=1,
                epochs=1,
                seed=4,
                eval_hands=1,
                eval_opponents=("greedy",),
            )

            summary = format_experiment_summary(result)

        self.assertIn("Haggis policy experiment", summary)
        self.assertIn("Records:", summary)
        self.assertIn("policy vs greedy", summary)
        self.assertNotIn("Policy-rollout evaluation:", summary)
        self.assertIn("Data:", summary)
        self.assertIn("Model:", summary)
        self.assertIn("Metrics:", summary)
        self.assertIn("Manifest:", summary)
        self.assertIn("Ladder: not run", summary)

    def test_experiment_cli_writes_outputs_and_prints_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output-dir",
                        directory,
                        "--data-hands",
                        "1",
                        "--epochs",
                        "1",
                        "--eval-hands",
                        "1",
                        "--eval-opponents",
                        "greedy",
                        "--observation-mode",
                        "player",
                        "--averaged",
                        "--validation-fraction",
                        "0.25",
                        "--ladder-hands",
                        "1",
                        "--evaluate-policy-rollout",
                        "--rollout-simulations",
                        "1",
                        "--rollout-root-moves",
                        "3",
                        "--rollout-turns",
                        "10",
                        "--seed",
                        "5",
                    ]
                )

            output = Path(directory)
            files = {path.name for path in output.iterdir()}
            records = [json.loads(line) for line in (output / "self_play.jsonl").read_text(encoding="utf-8").splitlines()]
            model = json.loads((output / "linear_policy.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            ladder = json.loads((output / "ladder.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("self_play.jsonl", files)
        self.assertIn("linear_policy.json", files)
        self.assertIn("metrics.json", files)
        self.assertIn("manifest.json", files)
        self.assertIn("ladder.json", files)
        self.assertIn("Haggis policy experiment", stdout.getvalue())
        self.assertEqual(records[0]["observation_mode"], "player")
        self.assertTrue(model["averaged"])
        self.assertGreater(metrics["training"]["validation_examples"], 0)
        self.assertIsNotNone(metrics["training"]["validation_accuracy"])
        self.assertEqual(metrics["ladder"], str(output / "ladder.json"))
        self.assertEqual(ladder["config"]["bots"], ["policy", "policy-rollout", "greedy"])
        self.assertEqual(ladder["config"]["hands_per_match"], 1)
        self.assertEqual(len(ladder["standings"]), 3)


if __name__ == "__main__":
    unittest.main()
