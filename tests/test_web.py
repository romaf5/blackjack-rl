import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from blackjack_rl.engine import Rules
from blackjack_rl.web.server import make_handler, rules_from_json
from blackjack_rl.web.session import GameSession


def test_session_round_trip():
    s = GameSession(rules=Rules(), bet_sizes=(1, 2, 4, 8), bankroll=50, seed=1, checkpoint=None)
    st = s.state()
    assert st["phase"] == "bet" and st["agents"] == ["basic", "hilo"] and st["rl"]["loaded"] is False
    with pytest.raises(ValueError):
        s.act("hit")
    st = s.bet(1)
    assert st["current_bet"] == 2
    while st["phase"] == "play":
        assert st["dealer"]["hidden"] is True and st["dealer"]["total"] is None
        assert set(st["legal"]) <= {"stand", "hit", "double", "split", "surrender"}
        adv = s.advice()
        assert adv["basic"] in st["legal"]
        st = s.act("hit" if st["hands"][0]["total"] < 12 else "stand")
    assert st["phase"] == "done" and st["dealer"]["hidden"] is False
    assert st["bankroll"] == pytest.approx(50 + st["last"]["profit"])
    assert s.rounds == 1
    with pytest.raises(ValueError):
        s.bet(0)                     # must advance first
    assert s.next_round()["phase"] == "bet"


def test_agent_step_plays_whole_rounds():
    s = GameSession(rules=Rules(), bet_sizes=(1, 2, 4, 8), bankroll=100, seed=2)
    for _ in range(60):
        st = s.agent_step("hilo")
        assert st["agent_action"]
    assert s.rounds >= 10
    if s.round_over:
        s.next_round()
    with pytest.raises(ValueError):
        s.agent_step("rl")           # no checkpoint loaded


def test_rules_from_json():
    r = rules_from_json({"num_decks": 2, "dealer_hits_soft_17": False, "double_on": "9,10,11", "surrender": False})
    assert r.num_decks == 2 and not r.dealer_hits_soft_17 and r.double_on == (9, 10, 11) and not r.surrender
    assert rules_from_json({}) == Rules()


def test_http_api_end_to_end():
    session = GameSession(rules=Rules(), bet_sizes=(1, 5), bankroll=100, seed=3)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    def get(p):
        return json.load(urllib.request.urlopen(base + p))

    def post(p, body):
        req = urllib.request.Request(base + p, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            return json.load(urllib.request.urlopen(req)), 200
        except urllib.error.HTTPError as e:
            return json.load(e), e.code

    try:
        assert b"<title>Blackjack RL</title>" in urllib.request.urlopen(base + "/").read()
        assert urllib.request.urlopen(base + "/static/app.js").status == 200
        st = get("/api/state")
        assert st["phase"] == "bet" and st["bet_sizes"] == [1, 5]
        st, code = post("/api/bet", {"index": 1})
        assert code == 200 and st["phase"] in ("play", "done")
        bad, code = post("/api/bet", {"index": 0})
        assert code == 400 and "error" in bad
        while st["phase"] == "play":
            st, _ = post("/api/action", {"action": "stand"})
        assert st["phase"] == "done"
        st, _ = post("/api/next", {})
        assert st["phase"] == "bet"
        st, code = post("/api/new", {"rules": {"num_decks": 1}, "bet_sizes": [2, 4], "bankroll": 10})
        assert code == 200 and st["bet_sizes"] == [2, 4] and st["bankroll"] == 10 and "1 deck" in st["rules"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/static/../server.py")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_strategy_report_html_from_session():
    s = GameSession(rules=Rules(), bet_sizes=(1, 2, 4, 8), bankroll=100, seed=4)
    page = s.strategy_report_html("hilo", 2.0)
    assert page.startswith("<!doctype html>") and "Hard totals" in page and "Bet size by true count" in page
    with pytest.raises(ValueError):
        s.strategy_report_html("rl")
