from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from haggis.promotion import decide_promotion, main, run_promotion_gate


class PromotionTests(unittest.TestCase):
    def test_decide_promotion_passes_when_challenger_improves_and_speed_ok(self):
        decision = decide_promotion(
            champion_ladder_metrics={"standings": [{"bot": "policy-rollout", "rating": 1500.0, "hand_win_rate": 0.5}]},
            challenger_ladder_metrics={"standings": [{"bot": "policy-rollout", "rating": 1510.0, "hand_win_rate": 0.55}]},
            champion_benchmark_metrics={"bots": [{"bot": "policy-rollout", "average_seconds": 0.1}]},
            challenger_benchmark_metrics={"bots": [{"bot": "policy-rollout", "average_seconds": 0.11}]},
            min_rating_delta=0.0,
            min_win_rate_delta=0.0,
            max_speed_ratio=1.25,
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.rating_delta, 10.0)
        self.assertAlmostEqual(decision.win_rate_delta, 0.05)

    def test_decide_promotion_fails_when_rating_or_speed_gate_fails(self):
        decision = decide_promotion(
            champion_ladder_metrics={"standings": [{"bot": "policy-rollout", "rating": 1510.0, "hand_win_rate": 0.55}]},
            challenger_ladder_metrics={"standings": [{"bot": "policy-rollout", "rating": 1500.0, "hand_win_rate": 0.50}]},
            champion_benchmark_metrics={"bots": [{"bot": "policy-rollout", "average_seconds": 0.1}]},
            challenger_benchmark_metrics={"bots": [{"bot": "policy-rollout", "average_seconds": 0.2}]},
            min_rating_delta=0.0,
            min_win_rate_delta=0.0,
            max_speed_ratio=1.25,
        )

        self.assertFalse(decision.passed)
        self.assertGreaterEqual(len(decision.reasons), 2)

    def test_run_promotion_gate_writes_artifacts_and_can_promote_on_equal_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champion = root / "champion.json"
            challenger = root / "challenger.json"
            promoted = root / "promoted.json"
            shutil.copyfile("models/linear_policy.json", champion)
            shutil.copyfile("models/linear_policy.json", challenger)

            payload = run_promotion_gate(
                champion_model=champion,
                challenger_model=challenger,
                output_dir=root / "gate",
                hands=1,
                seed=22,
                baselines=("greedy",),
                search_simulations=1,
                search_root_moves=2,
                search_rollout_turns=8,
                benchmark_states=1,
                max_speed_ratio=100.0,
                promote_to=promoted,
            )
            written = json.loads((root / "gate" / "promotion.json").read_text(encoding="utf-8"))

            self.assertTrue(payload["decision"]["passed"])
            self.assertEqual(written["decision"]["passed"], payload["decision"]["passed"])
            self.assertTrue(promoted.exists())
            self.assertTrue((root / "gate" / "champion-ladder.json").exists())
            self.assertTrue((root / "gate" / "challenger-benchmark.json").exists())

    def test_promotion_cli_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champion = root / "champion.json"
            challenger = root / "challenger.json"
            shutil.copyfile("models/linear_policy.json", champion)
            shutil.copyfile("models/linear_policy.json", challenger)

            exit_code = main(
                [
                    "--champion",
                    str(champion),
                    "--challenger",
                    str(challenger),
                    "--output-dir",
                    str(root / "gate"),
                    "--hands",
                    "1",
                    "--seed",
                    "23",
                    "--baselines",
                    "greedy",
                    "--search-simulations",
                    "1",
                    "--search-root-moves",
                    "2",
                    "--search-rollout-turns",
                    "8",
                    "--benchmark-states",
                    "1",
                    "--max-speed-ratio",
                    "100",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "gate" / "promotion.json").exists())


if __name__ == "__main__":
    unittest.main()
