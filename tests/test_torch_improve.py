import json
import tempfile
import unittest
from pathlib import Path

from haggis.self_play import export_self_play_jsonl

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchImproveTests(unittest.TestCase):
    def test_improvement_loop_writes_candidate_and_report(self):
        from haggis.torch_gate import TorchGateResult, TorchGateGame
        from haggis.torch_improve import main, run_improvement_loop
        from haggis.torch_policy import train_torch_policy_from_jsonl

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "seed_records.jsonl"
            seed_model = Path(directory) / "seed.pt"
            output_dir = Path(directory) / "improve"
            cli_output_dir = Path(directory) / "improve_cli"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=33, observation_mode="player")
            train_torch_policy_from_jsonl(data_path, output_path=seed_model, epochs=1, hidden_size=8, batch_size=1, validation_fraction=0.25, seed=4)

            original_gate = __import__("haggis.torch_improve", fromlist=["run_torch_gate"]).run_torch_gate
            def fake_gate(**kwargs):
                return TorchGateResult(
                    passed=True,
                    champion=str(kwargs["champion"]),
                    challenger=str(kwargs["challenger"]),
                    target_score=kwargs["target_score"],
                    games=(TorchGateGame(seed=1, winner="challenger", score_margin_for_challenger=1, score=(51, 50), hands=1, path="fake"),),
                    challenger_wins=1,
                    champion_wins=0,
                    average_margin=1.0,
                    benchmark={},
                    promoted_to=None,
                )
            import haggis.torch_improve as torch_improve
            torch_improve.run_torch_gate = fake_gate
            try:
                payload = run_improvement_loop(
                    seed_model=seed_model,
                    output_dir=output_dir,
                    iterations=1,
                    hands_per_iteration=1,
                    teacher_a="point-aware",
                    teacher_b="bomb-control",
                    champion_baseline="point-aware",
                    epochs=1,
                    hidden_size=8,
                    batch_size=1,
                    validation_fraction=0.25,
                    seed=6,
                    ladder_hands=1,
                    gate_seeds=(1,),
                    gate_target_score=50,
                    gate_max_hands=3,
                    gate_require_wins=0,
                    search_root_moves=1,
                    search_rollout_turns=2,
                )
                exit_code = main([
                    "--seed-model", str(seed_model),
                    "--output-dir", str(cli_output_dir),
                    "--iterations", "1",
                    "--hands-per-iteration", "1",
                    "--teacher-a", "point-aware",
                    "--teacher-b", "bomb-control",
                    "--champion-baseline", "point-aware",
                    "--epochs", "1",
                    "--hidden-size", "8",
                    "--batch-size", "1",
                    "--validation-fraction", "0.25",
                    "--seed", "6",
                    "--ladder-hands", "1",
                    "--gate-seeds", "1:1",
                    "--gate-target-score", "50",
                    "--gate-max-hands", "3",
                    "--gate-require-wins", "0",
                    "--search-root-moves", "1",
                    "--search-rollout-turns", "2",
                ])
            finally:
                torch_improve.run_torch_gate = original_gate
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "champion.pt").exists())
            self.assertTrue((output_dir / "iteration-001" / "candidate.pt").exists())
            self.assertTrue((output_dir / "improve_report.json").exists())
            self.assertEqual(len(payload["iterations"]), 1)
            report = json.loads((output_dir / "improve_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["iterations"][0]["iteration"], 1)
            self.assertTrue(report["iterations"][0]["gate_passed"])


if __name__ == "__main__":
    unittest.main()
