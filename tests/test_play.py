from __future__ import annotations

import unittest

from haggis import HaggisState
from haggis.play import (
    format_cards,
    format_legal_moves,
    format_move,
    play_player_vs_cpu,
    prompt_bet,
    prompt_move,
)


def c(rank: int, suit: str = "C"):
    from haggis import Card, Rank, Suit

    return Card(Rank(rank), Suit(suit))


class PlayTests(unittest.TestCase):
    def test_prompt_bet_accepts_default_and_reprompts_invalid_amount(self):
        outputs: list[str] = []

        self.assertEqual(prompt_bet(lambda _prompt: "", outputs.append), 0)

        answers = iter(["7", "15"])
        self.assertEqual(prompt_bet(lambda _prompt: next(answers), outputs.append), 15)
        self.assertIn("Enter 0, 15, or 30.", outputs)

    def test_prompt_move_selects_by_number_and_pass_alias(self):
        state = HaggisState(hands=((c(3), c(4)), (c(5),)), haggis=(c(6),), current_player=0)
        selected = prompt_move(state, lambda _prompt: "1", lambda _line: None)
        self.assertIn(selected, state.legal_moves())

        played = state.apply_move(selected)
        pass_move = prompt_move(played, lambda _prompt: "pass", lambda _line: None)
        self.assertTrue(pass_move.is_pass)

    def test_prompt_move_selects_by_ascii_card_names(self):
        state = HaggisState(hands=((c(3), c(4)), (c(5),)), haggis=(c(6),), current_player=0)

        selected = prompt_move(state, lambda _prompt: "3C", lambda _line: None)

        self.assertEqual(format_cards(selected.cards), "3♣")

    def test_prompt_move_does_not_dump_all_legal_moves_by_default(self):
        state = HaggisState.new_deal(seed=1)
        answers = iter(["q"])
        outputs: list[str] = []

        with self.assertRaises(KeyboardInterrupt):
            prompt_move(state, lambda _prompt: next(answers), outputs.append)

        joined = "\n".join(outputs)
        self.assertIn("Legal options:", joined)
        self.assertIn("Type 'moves'", joined)
        self.assertNotIn("Legal moves:", joined)

    def test_prompt_move_can_browse_limited_legal_moves_on_request(self):
        state = HaggisState.new_deal(seed=1)
        answers = iter(["moves", "q"])
        outputs: list[str] = []

        with self.assertRaises(KeyboardInterrupt):
            prompt_move(state, lambda _prompt: next(answers), outputs.append)

        browsed = "\n".join(outputs)
        self.assertIn("Legal moves:", browsed)
        self.assertIn("more", browsed)

    def test_format_legal_moves_can_show_every_move(self):
        legal_moves = HaggisState.new_deal(seed=1).legal_moves()

        formatted = format_legal_moves(legal_moves, limit=None)

        self.assertEqual(formatted.count("\n  "), len(legal_moves))
        self.assertNotIn("more", formatted)

    def test_player_vs_cpu_can_quit_cleanly(self):
        answers = iter(["0", "q"])
        outputs: list[str] = []

        with self.assertRaises(KeyboardInterrupt):
            play_player_vs_cpu(
                cpu_bot="greedy",
                seed=3,
                max_turns=5,
                enable_betting=True,
                input_fn=lambda _prompt: next(answers),
                output_fn=outputs.append,
            )

        self.assertTrue(any("Your hand" in line for line in outputs))


if __name__ == "__main__":
    unittest.main()
