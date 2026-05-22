from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from random import Random
from typing import Any
from urllib.parse import urlparse

from .cards import Card
from .engine import HaggisState, Move
from .play import format_combination, format_move
from .tournament import Bot, make_bot
from .tournament import _next_dealer

STATIC_DIR = Path(__file__).with_name("web_static")


@dataclass
class WebGameSession:
    state: HaggisState
    cpu: Bot
    human_player: int = 0
    cpu_player: int = 1
    turn_log: list[str] | None = None
    betting_complete: bool = False
    bets_placed: tuple[bool, bool] = (False, False)
    max_cpu_turns: int = 100
    target_score: int = 350
    base_seed: int = 1
    hand_number: int = 1
    dealer: int = 1
    cumulative_score: tuple[int, int] = (0, 0)
    latest_played_cards: tuple[Card, ...] = ()
    latest_play_cleared: bool = False
    hand_score_breakdown: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.turn_log is None:
            self.turn_log = []

    def place_human_bet(self, amount: int) -> None:
        if self.bets_placed[self.human_player]:
            raise ValueError("you have already bet this hand")
        self.state = self.state.place_bet(self.human_player, amount)
        self.bets_placed = _replace_bool(self.bets_placed, self.human_player, True)
        self.betting_complete = all(self.bets_placed)
        self.turn_log.append(f"You bet {amount}.")
        self.play_cpu_until_human_turn()

    def play_human_cards(self, card_keys: list[str]) -> None:
        self._require_human_turn()
        move = self._move_from_card_keys(card_keys)
        if move is None:
            raise ValueError("selected cards are not a legal move")
        self._apply_move("You", move)
        self.play_cpu_until_human_turn()

    def pass_human_turn(self) -> None:
        self._require_human_turn()
        pass_move = next((move for move in self.state.legal_moves() if move.is_pass), None)
        if pass_move is None:
            raise ValueError("pass is not legal right now")
        self._apply_move("You", pass_move)
        self.play_cpu_until_human_turn()

    def start_next_hand(self) -> None:
        if self.state.hand_winner is None:
            raise ValueError("current hand is not complete")
        if self.game_winner is not None:
            raise ValueError("game is already complete")
        self.hand_number += 1
        self.dealer = _next_dealer(self.cumulative_score, last_hand_winner=self.state.hand_winner)
        self.state = HaggisState.new_deal(seed=self.base_seed + self.hand_number - 1, dealer=self.dealer).assert_invariants(full_deck=True)
        self.latest_played_cards = ()
        self.latest_play_cleared = False
        self.hand_score_breakdown = None
        self.bets_placed = (False, False)
        self.betting_complete = False
        self.turn_log.append(f"Hand {self.hand_number} started. First to {self.target_score}.")

    @property
    def game_winner(self) -> int | None:
        if max(self.cumulative_score) < self.target_score:
            return None
        if self.cumulative_score[0] == self.cumulative_score[1]:
            return None
        return 0 if self.cumulative_score[0] > self.cumulative_score[1] else 1

    def play_cpu_until_human_turn(self) -> None:
        turns = 0
        while (
            self.state.hand_winner is None
            and self.state.current_player == self.cpu_player
        ):
            if turns >= self.max_cpu_turns:
                raise RuntimeError("CPU turn loop exceeded safety limit")
            self._place_cpu_bet_if_needed()
            move = self.cpu.choose_move(self.state)
            if move not in self.state.legal_moves():
                raise ValueError(f"CPU chose an illegal move: {move}")
            self._apply_move("CPU", move)
            turns += 1

    def snapshot(self) -> dict[str, Any]:
        state = self.state
        human_must_bet = self._must_bet(self.human_player)
        legal_moves = state.legal_moves() if state.hand_winner is None and not human_must_bet else ()
        human_legal_moves = legal_moves if state.current_player == self.human_player else ()
        hand_score = state.score_hand().points if state.hand_winner is not None else None
        game_winner = self.game_winner
        return {
            "targetScore": self.target_score,
            "handNumber": self.hand_number,
            "cumulativeScore": {"human": self.cumulative_score[self.human_player], "cpu": self.cumulative_score[self.cpu_player]},
            "gameWinner": None if game_winner is None else ("human" if game_winner == self.human_player else "cpu"),
            "bettingComplete": self.betting_complete,
            "betsPlaced": {"human": self.bets_placed[self.human_player], "cpu": self.bets_placed[self.cpu_player]},
            "humanMustBet": human_must_bet,
            "currentPlayer": "human" if state.current_player == self.human_player else "cpu",
            "handWinner": None if state.hand_winner is None else ("human" if state.hand_winner == self.human_player else "cpu"),
            "score": None if hand_score is None else {"human": hand_score[self.human_player], "cpu": hand_score[self.cpu_player]},
            "bets": {"human": state.bets[self.human_player], "cpu": state.bets[self.cpu_player]},
            "humanHand": [card_json(card) for card in state.hands[self.human_player]],
            "cpuCards": len(state.hands[self.cpu_player]),
            "cpuWilds": [card_json(card) for card in state.hands[self.cpu_player] if card.is_wild],
            "capturedPoints": {
                "human": sum(card.points for card in state.captured[self.human_player]),
                "cpu": sum(card.points for card in state.captured[self.cpu_player]),
            },
            "trickPoints": sum(card.points for card in state.trick_cards),
            "lastCombination": None if state.last_combination is None else format_combination(state.last_combination),
            "lastPlayer": None if state.last_player is None else ("human" if state.last_player == self.human_player else "cpu"),
            "lastPlayedCards": [card_json(card) for card in _last_played_cards(state, self.latest_played_cards)],
            "currentPlayCards": [] if self.latest_play_cleared else [card_json(card) for card in self.latest_played_cards],
            "trickCards": [] if self.latest_play_cleared else [card_json(card) for card in _visible_trick_cards(state, self.latest_played_cards)],
            "currentPlayCleared": self.latest_play_cleared,
            "handScoreBreakdown": self.hand_score_breakdown,
            "canPass": any(move.is_pass for move in human_legal_moves),
            "selectedHint": "Select cards, then Play. Invalid selections are rejected by the rules engine.",
            "legalMoveCount": sum(1 for move in human_legal_moves if not move.is_pass),
            "legalMoves": [move_json(move) for move in human_legal_moves[:40]],
            "log": self.turn_log[-30:],
        }

    def _require_human_turn(self) -> None:
        if self._must_bet(self.human_player):
            raise ValueError("place a bet before playing your first card")
        if self.state.hand_winner is not None:
            raise ValueError("hand is already complete")
        if self.state.current_player != self.human_player:
            raise ValueError("it is not your turn")

    def _must_bet(self, player: int) -> bool:
        return not self.bets_placed[player] and not self.state.has_played[player] and self.state.current_player == player and self.state.hand_winner is None

    def _place_cpu_bet_if_needed(self) -> None:
        if not self._must_bet(self.cpu_player):
            return
        chooser = getattr(self.cpu, "choose_bet", None)
        cpu_bet = chooser(self.state, self.cpu_player) if chooser is not None else 0
        self.state = self.state.place_bet(self.cpu_player, cpu_bet)
        self.bets_placed = _replace_bool(self.bets_placed, self.cpu_player, True)
        self.betting_complete = all(self.bets_placed)
        self.turn_log.append(f"CPU bet {cpu_bet}.")

    def _move_from_card_keys(self, card_keys: list[str]) -> Move | None:
        requested = sorted(card_keys)
        for move in self.state.legal_moves():
            if sorted(card_key(card) for card in move.cards) == requested:
                return move
        return None

    def _record_completed_hand(self) -> None:
        score = self.state.score_hand()
        cumulative = list(self.cumulative_score)
        cumulative[0] += score.points[0]
        cumulative[1] += score.points[1]
        self.cumulative_score = tuple(cumulative)
        self.hand_score_breakdown = self._score_breakdown(score.winner)
        winner = "You" if score.winner == self.human_player else "CPU"
        self.turn_log.append(f"Hand complete. {winner} won {score.points[self.human_player]}–{score.points[self.cpu_player]}.")
        if self.game_winner is not None:
            game_winner = "You" if self.game_winner == self.human_player else "CPU"
            self.turn_log.append(f"Game complete. {game_winner} reached {self.target_score} points.")

    def _score_breakdown(self, winner: int) -> dict[str, Any]:
        loser = 1 - winner
        leftover_count_bonus = len(self.state.hands[loser]) * 5
        captured = [sum(card.points for card in self.state.captured[player]) for player in (0, 1)]
        leftover_points = sum(card.points for card in self.state.hands[loser])
        haggis_points = sum(card.points for card in self.state.haggis)
        bet_awards = [0, 0]
        rows = [
            {
                "label": "5xCaptured cards",
                "human": captured[self.human_player]
                + (leftover_points if winner == self.human_player else 0)
                + (haggis_points if winner == self.human_player else 0),
                "cpu": captured[self.cpu_player]
                + (leftover_points if winner == self.cpu_player else 0)
                + (haggis_points if winner == self.cpu_player else 0),
            },
            {
                "label": "Leftover-card bonus",
                "human": leftover_count_bonus if winner == self.human_player else 0,
                "cpu": leftover_count_bonus if winner == self.cpu_player else 0,
            },
        ]
        for player, bet in enumerate(self.state.bets):
            if not bet:
                continue
            label = "Big Bet" if bet == 30 else "Little Bet" if bet == 15 else f"Bet {bet}"
            award_player = player if player == winner else 1 - player
            rows.append({"label": label, "human": bet if award_player == self.human_player else 0, "cpu": bet if award_player == self.cpu_player else 0})
        return {
            "winner": "human" if winner == self.human_player else "cpu",
            "loserCardsRemaining": len(self.state.hands[loser]),
            "haggisPoints": haggis_points,
            "rows": rows,
            "total": {"human": sum(row["human"] for row in rows), "cpu": sum(row["cpu"] for row in rows)},
        }

    def _apply_move(self, actor: str, move: Move) -> None:
        self.turn_log.append(f"{actor}: {format_move(move)}")
        self.latest_play_cleared = move.is_pass
        if not move.is_pass:
            self.latest_played_cards = move.cards
        self.state = self.state.apply_move(move).assert_invariants(full_deck=True)
        if self.state.hand_winner is not None:
            self._record_completed_hand()


def _replace_bool(values: tuple[bool, bool], index: int, value: bool) -> tuple[bool, bool]:
    updated = list(values)
    updated[index] = value
    return tuple(updated)  # type: ignore[return-value]


def _last_played_cards(state: HaggisState, latest_played_cards: tuple[Card, ...] = ()) -> tuple[Card, ...]:
    if latest_played_cards:
        return latest_played_cards
    if state.last_combination is None:
        return ()
    card_count = state.last_combination.card_count
    if card_count <= 0:
        return ()
    return state.trick_cards[-card_count:]


def _visible_trick_cards(state: HaggisState, latest_played_cards: tuple[Card, ...] = ()) -> tuple[Card, ...]:
    if state.trick_cards:
        return state.trick_cards
    return latest_played_cards


def new_session(seed: int | None = None, cpu_name: str = "policy-rollout", target_score: int = 350) -> WebGameSession:
    if target_score < 1:
        raise ValueError("target_score must be at least 1")
    actual_seed = Random().randrange(1, 1_000_000_000) if seed is None else seed
    cpu = make_bot(cpu_name, seed=actual_seed * 2 + 1, policy_model="models/linear_policy.json", search_root_moves=4, search_rollout_turns=40)
    state = HaggisState.new_deal(seed=actual_seed, dealer=1).assert_invariants(full_deck=True)
    return WebGameSession(state=state, cpu=cpu, target_score=target_score, base_seed=actual_seed, dealer=1, turn_log=[f"Game started. First to {target_score}."])


def card_key(card: Card) -> str:
    suffix = "W" if card.is_wild else "N"
    return f"{card.rank.label}{card.suit.value}{suffix}"


def card_json(card: Card) -> dict[str, Any]:
    return {"key": card_key(card), "name": card.short_name(), "rank": card.rank.label, "suit": card.suit.value, "points": card.points, "wild": card.is_wild}


def move_json(move: Move) -> dict[str, Any]:
    return {"cards": [card_key(card) for card in move.cards], "label": format_move(move), "pass": move.is_pass}


class HaggisWebApp:
    def __init__(self) -> None:
        self.sessions: dict[str, WebGameSession] = {}

    def create_session(self, seed: int | None = None, cpu_name: str = "policy-rollout", target_score: int = 350) -> tuple[str, WebGameSession]:
        session_id = f"s{Random().randrange(1, 10**18):018d}"
        session = new_session(seed=seed, cpu_name=cpu_name, target_score=target_score)
        self.sessions[session_id] = session
        return session_id, session

    def get_session(self, session_id: str) -> WebGameSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise ValueError("unknown session") from exc


APP = HaggisWebApp()


class HaggisRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
        elif path == "/style.css":
            self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/new":
                session_id, session = APP.create_session(seed=payload.get("seed"), cpu_name=payload.get("cpu", "policy-rollout"), target_score=int(payload.get("targetScore", 350)))
                self._send_json({"sessionId": session_id, "state": session.snapshot()})
                return

            session = APP.get_session(str(payload.get("sessionId", "")))
            if path == "/api/bet":
                session.place_human_bet(int(payload.get("amount", 0)))
            elif path == "/api/play":
                session.play_human_cards(list(payload.get("cards", [])))
            elif path == "/api/pass":
                session.pass_human_turn()
            elif path == "/api/next-hand":
                session.start_next_hand()
            else:
                self._send_json({"error": "not found"}, status=404)
                return
            self._send_json({"state": session.snapshot()})
        except Exception as exc:  # keep local UI errors visible and recoverable
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local browser UI for Haggis")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), HaggisRequestHandler)
    print(f"Haggis web UI running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
