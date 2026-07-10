"""Local browser app for playing one Plump round against random legal bots."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from plump.cards import Card, Rank, Suit, card_str
from plump.env import PlumpEnv
from plump.policies import ModelPolicy
from plump.state import BidAction, GameConfig, IllegalActionError, Phase, PlayCardAction


HUMAN_PLAYER = 0


@dataclass
class GuiGame:
    env: PlumpEnv
    rng: random.Random
    messages: list[str]


class RandomLegalModel:
    """Temporary opponent model that samples only environment-legal actions."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def act(self, env: PlumpEnv) -> BidAction | PlayCardAction:
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available for model player.")
        return self.rng.choice(actions)


class CheckpointModel:
    """Greedy checkpoint policy plus per-player prediction helpers for the GUI."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.policy = ModelPolicy.from_checkpoint(checkpoint_path, device=device, greedy=True)

    def act(self, env: PlumpEnv) -> BidAction | PlayCardAction:
        return self.policy.act(env)

    def predictions(self, env: PlumpEnv) -> dict[int, dict[str, Any]]:
        predictions: dict[int, dict[str, Any]] = {}
        for observer in range(env.config.num_players):
            obs = env.get_observation(observer)
            encoded, output = self.policy.predict_observation(obs)
            trick_probs = torch.softmax(output.masked_trick_count_logits[0], dim=-1)
            score_probs = (
                output.score_probs[0]
                if hasattr(output, "score_probs")
                else output.point_probs[0]
            )

            rows = []
            for rel in range(encoded.num_players):
                abs_player = (observer + rel) % encoded.num_players
                probs = trick_probs[rel]
                counts = torch.arange(probs.shape[0], dtype=torch.float32, device=probs.device)
                expected = float((probs * counts).sum().item())
                top_count = int(probs.argmax(dim=-1).item())
                top_prob = float(probs[top_count].item())
                rows.append(
                    {
                        "player": abs_player,
                        "expected_tricks": expected,
                        "top_tricks": top_count,
                        "top_tricks_prob": top_prob,
                        "point_prob": float(score_probs[rel].item()),
                    }
                )

            predictions[observer] = {
                "observer": observer,
                "source": "checkpoint",
                "rows": rows,
            }
        return predictions


class GuiController:
    """Stateful game controller for the browser API."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None) -> None:
        self.game: GuiGame | None = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.model = CheckpointModel(self.checkpoint_path, device=device) if self.checkpoint_path is not None else None

    def new_game(self, opponents: int, hand_size: int, human_bid_position: int, seed: int | None = None) -> dict[str, Any]:
        if not 2 <= opponents <= 4:
            raise ValueError("Choose 2 to 4 opponents.")
        num_players = opponents + 1
        if not 3 <= hand_size <= 10:
            raise ValueError("Choose 3 to 10 cards.")
        if not 1 <= human_bid_position <= num_players:
            raise ValueError(f"Your bid order must be 1 to {num_players}.")

        bidding_start = (HUMAN_PLAYER - (human_bid_position - 1)) % num_players
        config = GameConfig(
            num_players=num_players,
            hand_sizes=[hand_size],
            bidding_start_players=[bidding_start],
            auto_advance_rounds=True,
        )
        rng = random.Random(seed)
        env = PlumpEnv(config, seed=seed)
        env.reset()
        self.game = GuiGame(
            env=env,
            rng=rng,
            messages=[
                f"New round: {num_players} players, {hand_size} cards.",
                f"You bid {ordinal(human_bid_position)} in order.",
            ],
        )
        self._advance_bidding_bots()
        return self.view()

    def bid(self, bid_value: int) -> dict[str, Any]:
        game = self._require_game()
        self._ensure_human_turn(Phase.BIDDING)
        game.env.step(BidAction(HUMAN_PLAYER, bid_value))
        game.messages.append(f"You bid {bid_value}.")
        self._advance_bidding_bots()
        return self.view()

    def play(self, suit: str, rank: int) -> dict[str, Any]:
        game = self._require_game()
        self._ensure_human_turn(Phase.PLAYING)
        card = Card(Suit(suit), Rank(rank))
        game.env.step(PlayCardAction(HUMAN_PLAYER, card))
        game.messages.append(f"You played {card_str(card)}.")
        self._announce_round_over()
        return self.view()

    def advance_bot(self) -> dict[str, Any]:
        game = self._require_game()
        if not game.env.is_done() and game.env.current_player() != HUMAN_PLAYER:
            self._advance_one_bot()
        return self.view()

    def view(self) -> dict[str, Any]:
        game = self._require_game()
        env = game.env
        state = env.state
        round_state = state.current_round if state.rounds else None
        human_obs = env.get_observation(HUMAN_PLAYER) if round_state is not None else None
        legal_bids = human_obs.legal_bids if human_obs else []
        legal_cards = {card_key(card) for card in human_obs.legal_cards} if human_obs else set()

        players = []
        bid_by_player = {bid.player: bid.value for bid in round_state.bids} if round_state else {}
        predictions = self.model.predictions(env) if self.model is not None and round_state is not None else {}
        for player in range(env.config.num_players):
            hand_count = len(round_state.current_hands[player]) if round_state else 0
            prediction = predictions.get(player)
            players.append(
                {
                    "id": player,
                    "name": "You" if player == HUMAN_PLAYER else f"Model {player}",
                    "is_human": player == HUMAN_PLAYER,
                    "is_current": state.current_player == player,
                    "cards": hand_count,
                    "bid": bid_by_player.get(player),
                    "tricks": round_state.tricks_won.get(player, 0) if round_state else 0,
                    "score": state.cumulative_scores.get(player, 0),
                    "prediction": prediction,
                }
            )

        current_trick = serialize_trick(round_state.tricks[-1]) if round_state and round_state.tricks else None
        completed_tricks = [serialize_trick(trick) for trick in round_state.tricks if trick.winner is not None] if round_state else []
        bids = [{"player": bid.player, "value": bid.value, "position": bid.position} for bid in round_state.bids] if round_state else []
        hand = [serialize_card(card, legal=card_key(card) in legal_cards) for card in human_obs.my_hand] if human_obs else []

        return {
            "ok": True,
            "phase": state.phase.value,
            "done": env.is_done(),
            "current_player": state.current_player,
            "human_turn": state.current_player == HUMAN_PLAYER,
            "hand_size": round_state.hand_size if round_state else None,
            "trump": round_state.trump_suit.value if round_state and round_state.trump_suit else None,
            "trump_label": suit_label(round_state.trump_suit) if round_state else "No trump",
            "bidding_order": round_state.bidding_order if round_state else [],
            "play_start_player": round_state.play_start_player if round_state else None,
            "players": players,
            "my_hand": hand,
            "legal_bids": legal_bids,
            "bids": bids,
            "current_trick": current_trick,
            "completed_tricks": completed_tricks,
            "last_trick": completed_tricks[-1] if completed_tricks else None,
            "round_scores": round_state.round_scores if round_state else {},
            "messages": game.messages[-8:],
            "event_log": serialize_events(state.event_log[-40:]),
            "model_checkpoint": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
        }

    def _advance_bidding_bots(self) -> None:
        game = self._require_game()
        while (
            not game.env.is_done()
            and game.env.phase() == Phase.BIDDING
            and game.env.current_player() != HUMAN_PLAYER
        ):
            self._advance_one_bot()

    def _advance_one_bot(self) -> None:
        game = self._require_game()
        if game.env.is_done() or game.env.current_player() == HUMAN_PLAYER:
            return
        model = self.model or RandomLegalModel(game.rng)
        action = model.act(game.env)
        if isinstance(action, BidAction):
            game.env.step(action)
            game.messages.append(f"Model {action.player} bid {action.bid}.")
        else:
            game.env.step(action)
            game.messages.append(f"Model {action.player} played {card_str(action.card)}.")
        self._announce_round_over()

    def _announce_round_over(self) -> None:
        game = self._require_game()
        if not game.env.is_done():
            return
        if any(message.startswith("Round over.") for message in game.messages[-3:]):
            return
        scores = game.env.state.current_round.round_scores
        score_text = ", ".join(f"{player_name(player)} {score}" for player, score in sorted(scores.items()))
        game.messages.append(f"Round over. Scores: {score_text}.")

    def _require_game(self) -> GuiGame:
        if self.game is None:
            raise RuntimeError("Start a game first.")
        return self.game

    def _ensure_human_turn(self, phase: Phase) -> None:
        game = self._require_game()
        if game.env.phase() != phase:
            raise IllegalActionError(f"Expected phase {phase.value}, got {game.env.phase().value}.")
        if game.env.current_player() != HUMAN_PLAYER:
            raise IllegalActionError("It is not your turn.")


def serialize_card(card: Card, legal: bool = False) -> dict[str, Any]:
    return {
        "suit": card.suit.value,
        "rank": int(card.rank),
        "label": card_str(card),
        "key": card_key(card),
        "color": "red" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "black",
        "legal": legal,
    }


def serialize_trick(trick) -> dict[str, Any]:
    return {
        "index": trick.trick_index,
        "leader": trick.leader,
        "led_suit": trick.led_suit.value if trick.led_suit else None,
        "winner": trick.winner,
        "plays": [
            {
                "player": play.player,
                "position": play.position,
                "card": serialize_card(play.card),
            }
            for play in trick.plays
        ],
    }


def serialize_events(events) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        rows.append(
            {
                "type": event.type.value,
                "round_index": event.round_index,
                "player": event.player,
                "card": serialize_card(event.card) if event.card else None,
                "bid": event.bid,
                "trick_index": event.trick_index,
                "position_in_trick": event.position_in_trick,
            }
        )
    return rows


def card_key(card: Card) -> str:
    return f"{card.suit.value}:{int(card.rank)}"


def suit_label(suit: Suit | None) -> str:
    if suit is None:
        return "No trump"
    return {
        Suit.SPADES: "Spades",
        Suit.HEARTS: "Hearts",
        Suit.DIAMONDS: "Diamonds",
        Suit.CLUBS: "Clubs",
    }[suit]


def player_name(player: int) -> str:
    return "You" if player == HUMAN_PLAYER else f"Model {player}"


def ordinal(value: int) -> str:
    suffix = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


class GuiRequestHandler(BaseHTTPRequestHandler):
    controller = GuiController()
    static_dir = Path(__file__).with_name("static")

    def do_HEAD(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_headers(HTTPStatus.OK, "text/html; charset=utf-8", 0)
        elif self.path == "/styles.css":
            self._send_headers(HTTPStatus.OK, "text/css; charset=utf-8", 0)
        elif self.path == "/app.js":
            self._send_headers(HTTPStatus.OK, "application/javascript; charset=utf-8", 0)
        else:
            self._send_headers(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", 0)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(self.static_dir / "index.html", "text/html; charset=utf-8")
        elif self.path == "/styles.css":
            self._send_file(self.static_dir / "styles.css", "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self._send_file(self.static_dir / "app.js", "application/javascript; charset=utf-8")
        elif self.path == "/api/state":
            self._send_json(self.controller.view())
        else:
            self._send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/new":
                response = self.controller.new_game(
                    opponents=int(payload.get("opponents", 3)),
                    hand_size=int(payload.get("hand_size", 5)),
                    human_bid_position=int(payload.get("bid_position", 1)),
                    seed=int(payload["seed"]) if payload.get("seed") not in (None, "") else None,
                )
            elif self.path == "/api/bid":
                response = self.controller.bid(int(payload["bid"]))
            elif self.path == "/api/play":
                response = self.controller.play(str(payload["suit"]), int(payload["rank"]))
            elif self.path == "/api/advance":
                response = self.controller.advance_bot()
            else:
                self._send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(response)
        except (KeyError, ValueError, RuntimeError, IllegalActionError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self._send_headers(HTTPStatus.OK, content_type, len(data))
        self.wfile.write(data)

    def _send_headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.end_headers()


def run(
    host: str = "127.0.0.1",
    port: int = 8765,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> None:
    GuiRequestHandler.controller = GuiController(checkpoint_path=checkpoint_path, device=device)
    server = ThreadingHTTPServer((host, port), GuiRequestHandler)
    print(f"Plump GUI running at http://{host}:{port}")
    if checkpoint_path is not None:
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"GUI inference device: {GuiRequestHandler.controller.model.device}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local Plump browser GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(host=args.host, port=args.port, checkpoint_path=args.checkpoint, device=args.device)
