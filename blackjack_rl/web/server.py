"""Tiny dependency-free HTTP server exposing a GameSession + the static web UI."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from ..engine import Rules
from ..cli.common import add_rules_args, bets_from_args, rules_from_args
from .session import GameSession

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def rules_from_json(d: Dict[str, Any]) -> Rules:
    base = Rules()
    double_on = d.get("double_on")
    if isinstance(double_on, str):
        double_on = tuple(int(x) for x in double_on.split(",") if x.strip()) or None
    return Rules(
        num_decks=int(d.get("num_decks", base.num_decks)),
        penetration=float(d.get("penetration", base.penetration)),
        dealer_hits_soft_17=bool(d.get("dealer_hits_soft_17", base.dealer_hits_soft_17)),
        blackjack_payout=float(d.get("blackjack_payout", base.blackjack_payout)),
        dealer_peeks=bool(d.get("dealer_peeks", base.dealer_peeks)),
        double_after_split=bool(d.get("double_after_split", base.double_after_split)),
        double_on=tuple(double_on) if double_on else None,
        max_splits=int(d.get("max_splits", base.max_splits)),
        resplit_aces=bool(d.get("resplit_aces", base.resplit_aces)),
        hit_split_aces=bool(d.get("hit_split_aces", base.hit_split_aces)),
        surrender=bool(d.get("surrender", base.surrender)),
    )


def make_handler(session: GameSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "blackjack-rl/0.1"

        def log_message(self, fmt, *args):  # quieter console
            if os.environ.get("BLACKJACK_WEB_LOG"):
                super().log_message(fmt, *args)

        # ---------------------------------------------------------- helpers
        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name: str) -> None:
            path = os.path.normpath(os.path.join(STATIC_DIR, name))
            if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
                self.send_error(404)
                return
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n == 0:
                return {}
            return json.loads(self.rfile.read(n) or b"{}")

        # ---------------------------------------------------------- routes
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            with session.lock:
                if path == "/api/state":
                    return self._json(session.state())
                if path == "/api/advice":
                    return self._json(session.advice())
            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._body()
                with session.lock:
                    if path == "/api/new":
                        rules = rules_from_json(body.get("rules", {}))
                        bets = body.get("bet_sizes") or [1, 2, 4, 8]
                        bets = [float(b) for b in bets]
                        if not bets or any(b <= 0 for b in bets):
                            raise ValueError("bet sizes must be positive")
                        bankroll = float(body.get("bankroll", 100))
                        seed = body.get("seed")
                        return self._json(session.new_game(rules, bets, bankroll,
                                                           int(seed) if seed not in (None, "") else None))
                    if path == "/api/bet":
                        return self._json(session.bet(int(body["index"])))
                    if path == "/api/action":
                        return self._json(session.act(str(body["action"])))
                    if path == "/api/next":
                        return self._json(session.next_round())
                    if path == "/api/agent_step":
                        return self._json(session.agent_step(str(body.get("agent", "basic"))))
                self.send_error(404)
            except (ValueError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, status=400)

    return Handler


def serve(session: GameSession, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(session))
    url = f"http://{host}:{port}/"
    print(f"Blackjack table open at {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Play blackjack in the browser (served from the RL env).")
    add_rules_args(p)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--bankroll", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default="checkpoints/dqn.pt",
                   help="DQN checkpoint for the advisor / autoplay (optional)")
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    session = GameSession(rules=rules_from_args(a), bet_sizes=bets_from_args(a), bankroll=a.bankroll,
                          seed=a.seed, checkpoint=a.checkpoint)
    if session.dqn_usable:
        print(f"DQN advisor loaded from {a.checkpoint}")
    elif a.checkpoint:
        print(f"(no DQN advisor: {session.dqn_error or a.checkpoint + ' not found'})")
    serve(session, a.host, a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
