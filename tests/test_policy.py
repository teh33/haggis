import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from haggis import Card, HaggisState, Move, PolicyBot, Rank, Suit, validate_combination
from haggis.bots import PointAwareBot
from haggis.policy import (
    LinearPolicy,
    LinearValueModel,
    clamp_value_target,
    evaluate_policy_accuracy,
    evaluate_value_mae,
    features_from_record_action,
    features_from_state_action,
    split_train_validation,
    format_policy_inspection,
    inspect_policy,
    main,
    normalize_value_features,
    train_policy_from_jsonl,
    train_value_model_from_jsonl,
    value_features_from_record,
    value_target_from_record,
)
from haggis.self_play import export_self_play_jsonl, generate_self_play_records
from haggis.tournament import BOT_TYPES, run_match


SUIT_BY_SYMBOL = {
    "♣": Suit.CLUBS,
    "♦": Suit.DIAMONDS,
    "♥": Suit.HEARTS,
    "♠": Suit.SPADES,
}
RANK_BY_LABEL = {"J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING}


def c(rank, suit=Suit.CLUBS, wild=False):
    return Card(Rank(rank), suit, wild)


def _parse_card(card_id: str) -> Card:
    is_wild = card_id.endswith("*")
    core = card_id[:-1] if is_wild else card_id
    rank_label = core[:-1]
    suit_symbol = core[-1]
    rank = RANK_BY_LABEL[rank_label] if rank_label in RANK_BY_LABEL else Rank(int(rank_label))
    return Card(rank, SUIT_BY_SYMBOL[suit_symbol], is_wild)


def _card_points(card_id: str) -> int:
    core = card_id[:-1] if card_id.endswith("*") else card_id
    rank_label = core[:-1]
    if rank_label == "J":
        return 2
    if rank_label == "Q":
        return 3
    if rank_label == "K":
        return 5
    if rank_label in {"3", "5", "7", "9"}:
        return 1
    return 0


class PolicyTrainingTests(unittest.TestCase):
    def test_feature_extraction_matches_record_and_live_state(self):
        state = HaggisState(hands=((c(7), c(8)), (c(6),)), current_player=0)
        move = Move((c(7),), validate_combination((c(7),)))
        record = {
            "acting_player": 0,
            "state": {
                "hands": [["7♣", "8♣"], ["6♣"]],
                "hand_sizes": [2, 1],
                "captured_points": [0, 0],
                "trick_points": 0,
                "haggis_points": 0,
                "last_combination": None,
            },
            "legal_actions": [],
            "selected_action_index": 0,
            "outcome": {"actor_won": True},
        }
        action = {
            "index": 3,
            "is_pass": False,
            "cards": ["7♣"],
            "point_risk": 1,
            "combination": {
                "type": "set",
                "rank": 7,
                "card_count": 1,
                "bomb_rank": 0,
                "sequence_width": 0,
                "sequence_length": 0,
                "is_bomb": False,
            },
        }

        from_record = features_from_record_action(record, action)
        from_state = features_from_state_action(state, move, action_index=3)

        self.assertEqual(from_record["action.index"], from_state["action.index"])
        self.assertEqual(from_record["action.neg_index"], from_state["action.neg_index"])
        self.assertEqual(from_record["action.card_count"], from_state["action.card_count"])
        self.assertEqual(from_record["action.point_risk"], from_state["action.point_risk"])
        self.assertEqual(from_record["action.wild_count"], from_state["action.wild_count"])
        self.assertEqual(from_record["action.empties_hand"], from_state["action.empties_hand"])
        self.assertEqual(from_record["action.remaining_cards"], from_state["action.remaining_cards"])
        self.assertEqual(from_record["action.remaining_points"], from_state["action.remaining_points"])
        self.assertEqual(from_record["state.actor_hand_points"], from_state["state.actor_hand_points"])
        self.assertEqual(from_record["state.actor_wild_count"], from_state["state.actor_wild_count"])
        self.assertEqual(from_record["action.rank"], from_state["action.rank"])
        self.assertEqual(from_record["state.actor_cards"], 2.0)
        self.assertEqual(from_record["state.opponent_cards"], 1.0)
        self.assertEqual(from_record["action.sheds_hand_fraction"], from_state["action.sheds_hand_fraction"])
        self.assertEqual(from_record["action.point_fraction"], from_state["action.point_fraction"])
        self.assertEqual(from_record["state.opponent_hand_points"], from_state["state.opponent_hand_points"])
        self.assertEqual(from_record["state.hand_point_delta"], from_state["state.hand_point_delta"])
        self.assertEqual(from_record["state.captured_point_delta"], from_state["state.captured_point_delta"])
        self.assertEqual(from_record["state.leading"], from_state["state.leading"])
        self.assertEqual(from_record["state.actor_bet"], from_state["state.actor_bet"])
        self.assertEqual(from_record["state.opponent_bet"], from_state["state.opponent_bet"])
        self.assertEqual(from_record["state.bet_delta"], from_state["state.bet_delta"])

    def test_training_saves_loads_and_improves_accuracy_on_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "model.json"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=4)

            records = generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=2, seed=4)
            untrained = LinearPolicy()
            untrained_accuracy = evaluate_policy_accuracy(untrained, records)
            policy, result = train_policy_from_jsonl(data_path, epochs=3)
            policy.save(model_path)
            loaded = LinearPolicy.load(model_path)

        self.assertGreater(result.examples, 0)
        self.assertGreater(result.updates, 0)
        self.assertFalse(result.averaged)
        self.assertFalse(loaded.averaged)
        self.assertGreaterEqual(result.accuracy, untrained_accuracy)
        self.assertEqual(policy.weights, loaded.weights)

    def test_averaged_training_saves_loads_averaged_model(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "averaged_model.json"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=12)

            policy, result = train_policy_from_jsonl(data_path, epochs=2, averaged=True)
            policy.save(model_path)
            loaded = LinearPolicy.load(model_path)

        self.assertTrue(result.averaged)
        self.assertTrue(policy.averaged)
        self.assertTrue(loaded.averaged)
        self.assertGreater(result.examples, 0)
        self.assertGreater(len(loaded.weights), 0)
        self.assertEqual(policy.weights, loaded.weights)

    def test_validation_split_is_deterministic_and_reports_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=2, seed=13)
            records = tuple(range(len(generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=2, seed=13))))

            train_indexes, validation_indexes = split_train_validation(records, validation_fraction=0.25)
            _, result = train_policy_from_jsonl(data_path, epochs=2, validation_fraction=0.25)

        self.assertGreater(len(train_indexes), 0)
        self.assertGreater(len(validation_indexes), 0)
        self.assertEqual(tuple(train_indexes), records[: len(train_indexes)])
        self.assertEqual(tuple(validation_indexes), records[len(train_indexes) :])
        self.assertEqual(result.train_examples, len(train_indexes))
        self.assertEqual(result.examples, len(train_indexes) * result.epochs)
        self.assertEqual(result.validation_examples, len(validation_indexes))
        self.assertIsNotNone(result.validation_accuracy)
        self.assertGreaterEqual(result.train_accuracy, 0.0)
        self.assertLessEqual(result.train_accuracy, 1.0)
        self.assertGreaterEqual(result.validation_accuracy, 0.0)
        self.assertLessEqual(result.validation_accuracy, 1.0)

    def test_value_targets_are_extracted_from_outcome(self):
        record = {
            "outcome": {"actor_won": True, "actor_score_margin": 37},
            "selected_action_index": 0,
            "legal_actions": [],
        }

        self.assertEqual(value_target_from_record(record, target="actor_win"), 1.0)
        self.assertEqual(value_target_from_record(record, target="actor_score_margin_normalized"), 0.37)

    def test_score_margin_target_accepts_legacy_and_current_outcome_keys(self):
        self.assertEqual(value_target_from_record({"outcome": {"score_margin_for_actor": 37}}, target="actor_score_margin_normalized"), 0.37)
        self.assertEqual(value_target_from_record({"outcome": {"actor_score_margin": -42}}, target="actor_score_margin_normalized"), -0.42)

    def test_value_features_are_bounded_for_stable_training(self):
        features = {
            "bias": 1.0,
            "action.index": 250.0,
            "action.neg_index": -250.0,
            "action.card_count": 40.0,
            "state.bet_delta": -60.0,
            "state.trick_points": 500.0,
            "action.sheds_hand_fraction": 2.5,
        }

        normalized = normalize_value_features(features)

        self.assertEqual(normalized["bias"], 1.0)
        self.assertEqual(normalized["action.index"], 1.0)
        self.assertEqual(normalized["action.neg_index"], -1.0)
        self.assertEqual(normalized["action.card_count"], 1.0)
        self.assertEqual(normalized["state.bet_delta"], -1.0)
        self.assertEqual(normalized["state.trick_points"], 1.0)
        self.assertEqual(normalized["action.sheds_hand_fraction"], 1.0)
        self.assertEqual(clamp_value_target(5.0), 1.0)
        self.assertEqual(clamp_value_target(-5.0), -1.0)

    def test_train_value_model_saves_loads_and_reports_mae(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "value_model.json"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=4, seed=14)

            model, result = train_value_model_from_jsonl(data_path, epochs=2, validation_fraction=0.25, target="actor_win")
            model.save(model_path)
            loaded = LinearValueModel.load(model_path)
            records = generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=4, seed=14)
            mae = evaluate_value_mae(loaded, records, target="actor_win")

        self.assertEqual(loaded.target, "actor_win")
        self.assertEqual(loaded.feature_scale, "bounded_v1")
        self.assertGreater(result.examples, 0)
        self.assertEqual(result.updates, result.examples)
        self.assertGreater(len(loaded.weights), 0)
        self.assertGreaterEqual(mae, 0.0)
        self.assertIsNotNone(result.validation_mean_absolute_error)

    def test_value_features_use_selected_action_features(self):
        records = generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=1, seed=15)
        record = records[0]

        features = value_features_from_record(record)
        selected = features_from_record_action(record, record["selected_action"])

        self.assertEqual(features, selected)

    def test_value_training_stays_finite_on_search_improved_scale_features(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            rows = []
            for index in range(40):
                rows.append(
                    {
                        "acting_player": 0,
                        "state": {
                            "hands": [["K♣*", "Q♣*", "J♣*"], None],
                            "hand_sizes": [17, 17],
                            "captured_points": [300 + index, 0],
                            "trick_points": 250 + index,
                            "haggis_points": 80,
                            "last_combination": None,
                            "bets": [30, 0],
                            "has_played": [True, False],
                        },
                        "legal_actions": [
                            {
                                "index": 120 + index,
                                "is_pass": False,
                                "cards": ["K♣*", "Q♣*", "J♣*"],
                                "point_risk": 500 + index,
                                "combination": {
                                    "type": "set",
                                    "rank": 13,
                                    "card_count": 3,
                                    "bomb_rank": 0,
                                    "sequence_width": 0,
                                    "sequence_length": 0,
                                    "is_bomb": False,
                                },
                            }
                        ],
                        "selected_action_index": 0,
                        "outcome": {"actor_won": index % 2 == 0, "actor_score_margin": 1000 if index % 2 == 0 else -1000},
                    }
                )
            data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            model, result = train_value_model_from_jsonl(data_path, epochs=3, learning_rate=0.0001, target="actor_score_margin_normalized")

        self.assertLess(result.mean_absolute_error, 10.0)
        self.assertTrue(all(abs(weight) < 10.0 for weight in model.weights.values()))

    def test_self_play_records_include_current_and_legacy_actor_margin_keys(self):
        records = generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=1, seed=17)
        record = records[0]

        self.assertIn("actor_score_margin", record["outcome"])
        self.assertEqual(record["outcome"]["actor_score_margin"], record["outcome"]["score_margin_for_actor"])

    def test_policy_cli_trains_value_model(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "value_model.json"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=4, seed=16)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "train-value",
                        "--input",
                        str(data_path),
                        "--output",
                        str(model_path),
                        "--epochs",
                        "2",
                        "--validation-fraction",
                        "0.25",
                        "--target",
                        "actor_win",
                    ]
                )
            loaded = LinearValueModel.load(model_path)

        self.assertEqual(exit_code, 0)
        self.assertIn("Trained linear value model", stdout.getvalue())
        self.assertGreater(len(loaded.weights), 0)

    def test_live_policy_scoring_uses_ordered_action_index_features(self):
        state = HaggisState(hands=((c(7), c(8), c(9)), (c(3),)), current_player=0)
        moves = state.legal_moves()
        policy = LinearPolicy(weights={"action.index": 1.0})

        move = policy.choose_move(state, moves)

        self.assertEqual(move, moves[-1])

    def test_live_and_record_features_have_same_action_indexes_for_legal_actions(self):
        records = generate_self_play_records(bot_a="point-aware", bot_b="bomb-control", hands=1, seed=9)
        record = records[0]
        hand0 = tuple(_parse_card(card_id) for card_id in record["state"]["hands"][0])
        hand1 = tuple(_parse_card(card_id) for card_id in record["state"]["hands"][1])
        state = HaggisState(
            hands=(hand0, hand1),
            current_player=record["state"]["current_player"],
            bets=tuple(record["state"]["bets"]),
            has_played=tuple(record["state"]["has_played"]),
        )
        moves = state.legal_moves()
        self.assertEqual(len(moves), len(record["legal_actions"]))

        for action, move in zip(record["legal_actions"], moves):
            with self.subTest(index=action["index"]):
                from_record = features_from_record_action(record, action)
                from_state = features_from_state_action(state, move, action_index=action["index"])
                self.assertEqual(from_record["action.index"], from_state["action.index"])
                self.assertEqual(from_record["action.neg_index"], from_state["action.neg_index"])
                self.assertEqual(from_record["action.card_count"], from_state["action.card_count"])
                self.assertEqual(from_record["action.point_risk"], from_state["action.point_risk"])
                self.assertEqual(from_record["action.wild_count"], from_state["action.wild_count"])
                self.assertEqual(from_record["action.empties_hand"], from_state["action.empties_hand"])
                self.assertEqual(from_record["action.remaining_cards"], from_state["action.remaining_cards"])
                self.assertEqual(from_record["action.remaining_points"], from_state["action.remaining_points"])
                self.assertEqual(from_record["state.actor_hand_points"], from_state["state.actor_hand_points"])
                self.assertEqual(from_record["state.actor_wild_count"], from_state["state.actor_wild_count"])

    def test_hand_aware_features_work_with_player_observation_records(self):
        records = generate_self_play_records(
            bot_a="point-aware",
            bot_b="bomb-control",
            hands=1,
            seed=10,
            observation_mode="player",
        )
        record = records[0]
        action = record["selected_action"]

        features = features_from_record_action(record, action)

        actor_hand = record["state"]["hands"][record["acting_player"]]
        expected_hand_points = sum(_card_points(card_id) for card_id in actor_hand)
        self.assertEqual(features["state.actor_hand_points"], expected_hand_points)
        self.assertEqual(features["state.actor_wild_count"], sum(1 for card_id in actor_hand if card_id.endswith("*")))
        self.assertEqual(features["action.wild_count"], sum(1 for card_id in action["cards"] if card_id.endswith("*")))
        self.assertEqual(features["action.remaining_cards"], len(actor_hand) - len(action["cards"]))
        self.assertEqual(features["action.remaining_points"], expected_hand_points - action["point_risk"])

    def test_live_policy_can_rank_by_hand_aware_action_features(self):
        state = HaggisState(hands=((c(7), c(7, Suit.HEARTS)), (c(3),)), current_player=0)
        moves = state.legal_moves()
        policy = LinearPolicy(weights={"action.empties_hand": 10.0, "action.remaining_cards": -1.0})

        move = policy.choose_move(state, moves)

        self.assertEqual(set(move.cards), {c(7), c(7, Suit.HEARTS)})

    def test_policy_bot_chooses_legal_move_from_saved_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            LinearPolicy(weights={"action.card_count": 1.0}).save(model_path)
            state = HaggisState(hands=((c(7), c(7, Suit.HEARTS)), (c(3),)), current_player=0)

            move = PolicyBot(model_path).choose_move(state)

        self.assertIn(move, state.legal_moves())
        self.assertEqual(len(move.cards), 2)

    def test_policy_bot_is_registered_for_tournaments(self):
        self.assertIs(BOT_TYPES["policy"], PolicyBot)

    def test_policy_bot_uses_baseline_betting_heuristic(self):
        state = HaggisState.new_deal(seed=9)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            LinearPolicy(weights={"action.card_count": 1.0}).save(model_path)
            policy_bot = PolicyBot(model_path)

            first = policy_bot.choose_bet(state, 0)
            second = policy_bot.choose_bet(state, 0)

        self.assertEqual(first, second)
        self.assertIn(first, (0, 15, 30))
        self.assertEqual(first, PointAwareBot().choose_bet(state, 0))

    def test_policy_bot_bets_are_applied_in_tournaments(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            LinearPolicy(weights={"action.card_count": 1.0}).save(model_path)

            result = run_match("policy", "policy", hands=1, seed=9, policy_model=str(model_path))

        self.assertEqual(result.hands[0].bets[0], result.hands[0].bets[1])
        self.assertIn(result.hands[0].bets[0], (0, 15, 30))
        self.assertEqual(result.total_bets_placed[0], 1 if result.hands[0].bets[0] else 0)
        self.assertEqual(result.total_bets_placed[1], 1 if result.hands[0].bets[1] else 0)

    def test_policy_can_play_tournament_when_model_path_is_provided(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            LinearPolicy(weights={"action.card_count": 1.0}).save(model_path)

            result = run_match("policy", "greedy", hands=1, seed=2, policy_model=str(model_path))

        self.assertEqual(result.bot_names, ("policy", "greedy"))
        self.assertEqual(len(result.hands), 1)
        self.assertGreater(sum(result.total_score), 0)

    def test_train_cli_writes_model_and_prints_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.jsonl"
            model_path = Path(directory) / "model.json"
            export_self_play_jsonl(data_path, bot_a="point-aware", bot_b="bomb-control", hands=1, seed=8)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "train",
                        "--input",
                        str(data_path),
                        "--output",
                        str(model_path),
                        "--epochs",
                        "2",
                        "--averaged",
                        "--validation-fraction",
                        "0.25",
                    ]
                )

            loaded = LinearPolicy.load(model_path)

        self.assertEqual(exit_code, 0)
        self.assertTrue(loaded.averaged)
        self.assertGreater(len(loaded.weights), 0)
        self.assertIn("Trained linear policy", stdout.getvalue())
        self.assertIn("averaged=True", stdout.getvalue())
        self.assertIn("validation_accuracy=", stdout.getvalue())

    def test_inspect_policy_returns_top_positive_and_negative_weights(self):
        policy = LinearPolicy(weights={"bad": -2.0, "small": 0.5, "good": 3.0}, averaged=True)

        inspection = inspect_policy(policy, top=2)

        self.assertEqual(inspection["model_type"], "linear_action_ranker")
        self.assertTrue(inspection["averaged"])
        self.assertEqual(inspection["weight_count"], 3)
        self.assertEqual([item["feature"] for item in inspection["top_positive"]], ["good", "small"])
        self.assertEqual([item["feature"] for item in inspection["top_negative"]], ["bad", "small"])

    def test_inspect_policy_groups_weights_by_feature_family(self):
        policy = LinearPolicy(
            weights={
                "action.card_count": 2.0,
                "action.point_risk": -1.5,
                "state.actor_cards": 0.75,
                "state.opponent_cards": -0.25,
                "bias": 0.1,
            }
        )

        inspection = inspect_policy(policy, top=2)
        groups = {group["family"]: group for group in inspection["groups"]}

        self.assertEqual(groups["action"]["count"], 2)
        self.assertEqual(groups["action"]["positive_total"], 2.0)
        self.assertEqual(groups["action"]["negative_total"], -1.5)
        self.assertEqual(groups["action"]["strongest_positive"], {"feature": "action.card_count", "weight": 2.0})
        self.assertEqual(groups["action"]["strongest_negative"], {"feature": "action.point_risk", "weight": -1.5})
        self.assertEqual(groups["state"]["count"], 2)
        self.assertEqual(groups["state"]["positive_total"], 0.75)
        self.assertEqual(groups["state"]["negative_total"], -0.25)
        self.assertEqual(groups["bias"]["count"], 1)
        self.assertIsNone(groups["bias"]["strongest_negative"])

    def test_format_policy_inspection_prints_readable_sections(self):
        inspection = inspect_policy(LinearPolicy(weights={"bad": -1.0, "good": 2.0, "action.card_count": 3.0}), top=1)

        output = format_policy_inspection(inspection)

        self.assertIn("Haggis linear policy inspection", output)
        self.assertIn("Top positive weights:", output)
        self.assertIn("good", output)
        self.assertIn("Top negative weights:", output)
        self.assertIn("bad", output)
        self.assertIn("Feature groups:", output)
        self.assertIn("action: count=1", output)
        self.assertIn("strongest_positive=action.card_count", output)

    def test_inspect_cli_prints_and_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            output_path = Path(directory) / "inspection.json"
            LinearPolicy(weights={"bad": -2.0, "good": 3.0}, averaged=True).save(model_path)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "inspect",
                        "--model",
                        str(model_path),
                        "--top",
                        "1",
                        "--output-json",
                        str(output_path),
                    ]
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("Haggis linear policy inspection", stdout.getvalue())
        self.assertEqual(payload["top_positive"][0]["feature"], "good")
        self.assertEqual(payload["top_negative"][0]["feature"], "bad")
        self.assertIn("groups", payload)
        self.assertTrue(payload["averaged"])

    def test_inspect_policy_rejects_invalid_top_count(self):
        with self.assertRaises(ValueError):
            inspect_policy(LinearPolicy(weights={"good": 1.0}), top=0)


if __name__ == "__main__":
    unittest.main()
