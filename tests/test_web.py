from __future__ import annotations

import unittest

from haggis import Card, HaggisState, Rank, Suit
from haggis.web import HaggisWebApp, card_key


class WebSessionTests(unittest.TestCase):
    def test_web_session_can_bet_and_play_a_legal_move(self):
        app = HaggisWebApp()
        session_id, session = app.create_session(seed=1, cpu_name="greedy")

        initial = session.snapshot()
        self.assertFalse(initial["bettingComplete"])
        self.assertEqual(initial["targetScore"], 350)
        self.assertEqual(initial["cumulativeScore"], {"human": 0, "cpu": 0})
        self.assertIsNone(initial["gameWinner"])
        self.assertEqual(app.get_session(session_id), session)
        self.assertEqual([card["rank"] for card in initial["cpuWilds"]], ["J", "Q", "K"])
        self.assertEqual(initial["lastPlayedCards"], [])
        self.assertEqual(initial["trickCards"], [])
        self.assertFalse(initial["currentPlayCleared"])
        self.assertIsNone(initial["handScoreBreakdown"])

        session.place_human_bet(0)
        after_bet = session.snapshot()
        self.assertTrue(after_bet["bettingComplete"])
        self.assertIn(after_bet["currentPlayer"], {"human", "cpu"})

        if after_bet["currentPlayer"] == "human" and after_bet["legalMoves"]:
            first_move = after_bet["legalMoves"][0]
            session.play_human_cards(first_move["cards"])
            after_play = session.snapshot()
            self.assertLessEqual(len(after_play["humanHand"]), len(after_bet["humanHand"]))
            self.assertIn("lastPlayedCards", after_play)
            self.assertIn("trickCards", after_play)
            self.assertTrue(after_play["log"])

    def test_cpu_pass_clears_visible_stack(self):
        _session_id, session = HaggisWebApp().create_session(seed=1, cpu_name="greedy")
        first_move = session.state.legal_moves()[0]
        session._apply_move("You", first_move)
        pass_move = next(move for move in session.state.legal_moves() if move.is_pass)
        session._apply_move("CPU", pass_move)

        snapshot = session.snapshot()
        self.assertTrue(snapshot["currentPlayCleared"])
        self.assertEqual(snapshot["trickCards"], [])

    def test_snapshot_preserves_latest_played_cards_after_trick_capture(self):
        _session_id, session = HaggisWebApp().create_session(seed=1, cpu_name="greedy")
        first_move = session.state.legal_moves()[0]
        session._apply_move("You", first_move)
        pass_move = next(move for move in session.state.legal_moves() if move.is_pass)
        session._apply_move("CPU", pass_move)

        snapshot = session.snapshot()
        expected_keys = [card_key(card) for card in first_move.cards]
        self.assertEqual([card["key"] for card in snapshot["lastPlayedCards"]], expected_keys)
        self.assertEqual(snapshot["currentPlayCards"], [])
        self.assertEqual(snapshot["trickCards"], [])

    def test_web_session_reaches_350_point_game_winner(self):
        _session_id, session = HaggisWebApp().create_session(seed=1, cpu_name="greedy")
        session.cumulative_score = (349, 100)
        session.state = HaggisState(hands=((), ((Card(Rank.THREE, Suit.CLUBS)),)), hand_winner=0)

        session._record_completed_hand()

        snapshot = session.snapshot()
        self.assertEqual(snapshot["targetScore"], 350)
        self.assertEqual(snapshot["gameWinner"], "human")
        self.assertIsNotNone(snapshot["handScoreBreakdown"])
        self.assertEqual(snapshot["handScoreBreakdown"]["total"], snapshot["score"])
        self.assertGreaterEqual(snapshot["cumulativeScore"]["human"], 350)

    def test_web_session_next_hand_resets_deal_when_game_not_complete(self):
        _session_id, session = HaggisWebApp().create_session(seed=1, cpu_name="greedy")
        session.cumulative_score = (20, 10)
        session.state = session.state.apply_move(session.state.legal_moves()[0])
        while session.state.hand_winner is None:
            legal = session.state.legal_moves()
            move = next((candidate for candidate in legal if candidate.is_pass), legal[0])
            session.state = session.state.apply_move(move)
        previous_hand = session.hand_number

        session.start_next_hand()

        self.assertEqual(session.hand_number, previous_hand + 1)
        self.assertFalse(session.betting_complete)
        self.assertIsNone(session.state.hand_winner)

    def test_web_session_rejects_illegal_selection(self):
        _session_id, session = HaggisWebApp().create_session(seed=1, cpu_name="greedy")
        session.place_human_bet(0)
        if session.snapshot()["currentPlayer"] != "human":
            self.skipTest("CPU retained the turn for this deterministic hand")

        with self.assertRaises(ValueError):
            session.play_human_cards(["not-a-card"])


if __name__ == "__main__":
    unittest.main()
