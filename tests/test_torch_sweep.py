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
class TorchSweepTests(unittest.TestCase):
    def test_sweep_ranks_trials_and_writes_summary(self):
        from haggis.torch_sweep import main, run_sweep

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            output_dir = Path(directory) / "models"
            summary_path = Path(directory) / "summary.json"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=32, observation_mode="player")

            trials = run_sweep(
                data_path,
                output_dir=output_dir,
                epochs=1,
                validation_fraction=0.25,
                seed=5,
                hidden_sizes=(8, 16),
                batch_sizes=(1,),
                dropouts=(0.0,),
                learning_rates=(0.001,),
                weight_decays=(0.0001,),
            )
            exit_code = main([
                "--input", str(data_path),
                "--output-dir", str(output_dir / "cli"),
                "--epochs", "1",
                "--validation-fraction", "0.25",
                "--seed", "5",
                "--hidden-sizes", "8,16",
                "--batch-sizes", "1",
                "--dropouts", "0.0",
                "--learning-rates", "0.001",
                "--weight-decays", "0.0001",
                "--summary", str(summary_path),
            ])
            self.assertEqual(exit_code, 0)
            self.assertEqual([trial.rank for trial in trials], [1, 2])
            self.assertTrue(all(Path(trial.output).exists() for trial in trials))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["best"]["rank"], 1)
            self.assertEqual(len(summary["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
