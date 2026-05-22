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
class TorchGateTests(unittest.TestCase):
    def test_gate_writes_report_and_promotes_when_passing(self):
        from haggis.torch_gate import main, run_torch_gate
        from haggis.torch_policy import train_torch_policy_from_jsonl

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            champion = Path(directory) / "champion.pt"
            challenger = Path(directory) / "challenger.pt"
            output_dir = Path(directory) / "gate"
            promoted = Path(directory) / "promoted" / "champion.pt"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=36, observation_mode="player")
            train_torch_policy_from_jsonl(data_path, output_path=champion, epochs=1, hidden_size=8, batch_size=1, validation_fraction=0.25, seed=4)
            train_torch_policy_from_jsonl(data_path, output_path=challenger, epochs=1, hidden_size=8, batch_size=1, validation_fraction=0.25, seed=4)

            from haggis.torch_gate import _parse_seeds

            self.assertEqual(_parse_seeds("1,3:5,8"), (1, 3, 4, 5, 8))
            self.assertEqual(_parse_seeds("5:3"), (5, 4, 3))

            result = run_torch_gate(
                champion=champion,
                challenger=challenger,
                output_dir=output_dir,
                seeds=(3,),
                target_score=50,
                max_hands=3,
                search_root_moves=1,
                search_rollout_turns=2,
                promote_to=promoted,
                min_promotion_games=1,
            )
            exit_code = main([
                "--champion", str(champion),
                "--challenger", str(challenger),
                "--output-dir", str(output_dir / "cli"),
                "--seeds", "3",
                "--target-score", "50",
                "--max-hands", "3",
                "--search-root-moves", "1",
                "--search-rollout-turns", "2",
                "--require-wins", "0",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(result.passed)
            self.assertTrue(promoted.exists())
            self.assertTrue((output_dir / "gate_report.json").exists())
            report = json.loads((output_dir / "gate_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["challenger"], str(challenger))
            self.assertEqual(len(report["games"]), 1)
    def test_promote_to_requires_stronger_gate(self):
        from haggis.torch_gate import run_torch_gate

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.pt"
            model.write_bytes(b"placeholder")
            with self.assertRaisesRegex(ValueError, "at least 60 gate games"):
                run_torch_gate(
                    champion=model,
                    challenger=model,
                    output_dir=Path(directory) / "gate",
                    seeds=(1,),
                    promote_to=Path(directory) / "promoted.pt",
                )

    def test_gate_requires_both_bet_models_for_bet_model_gate(self):
        from haggis.torch_gate import run_torch_gate

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.pt"
            model.write_bytes(b"placeholder")
            with self.assertRaisesRegex(ValueError, "must be provided together"):
                run_torch_gate(
                    champion=model,
                    challenger=model,
                    output_dir=Path(directory) / "gate",
                    seeds=(1,),
                    challenger_bet_model=Path(directory) / "bet.pt",
                )


if __name__ == "__main__":
    unittest.main()
