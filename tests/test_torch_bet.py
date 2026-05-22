import tempfile
import unittest
from pathlib import Path

from haggis.self_play import export_self_play_jsonl

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchBetTests(unittest.TestCase):
    def test_train_load_and_choose_bet(self):
        from haggis.engine import HaggisState
        from haggis.torch_bet import load_torch_bet_model, train_torch_bet_model_from_jsonl

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "bet.pt"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=3, seed=37, observation_mode="player")

            result = train_torch_bet_model_from_jsonl(data_path, output_path=model_path, epochs=1, hidden_size=8, batch_size=2, validation_fraction=0.25, seed=4)
            policy = load_torch_bet_model(model_path)
            bet = policy.choose_bet_from_hand(HaggisState.new_deal(seed=1).hands[0])

            self.assertTrue(model_path.exists())
            self.assertGreater(result.records, 0)
            self.assertIn(bet, (0, 15, 30))


if __name__ == "__main__":
    unittest.main()
