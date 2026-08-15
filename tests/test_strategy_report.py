import os

from blackjack_rl.agents import BasicStrategyAgent
from blackjack_rl.cli.strategy import TC_RANGE, build_report, main, render_html, render_text
from blackjack_rl.env import BlackjackEnv


def test_build_report_basic_and_hilo_match_basic_strategy():
    env = BlackjackEnv()
    rep = build_report("basic", BasicStrategyAgent(env.rules), env, 0.0, 0.6, None)
    assert rep.agreement(rep.hard) == (150, 150)
    assert rep.agreement(rep.soft) == (80, 80) and rep.agreement(rep.pairs) == (100, 100)
    assert rep.hard[(16, 10)].action == "R" and rep.pairs[(8, 10)].action == "P" and rep.soft[(18, 3)].action == "D"
    assert rep.agent_bets == [1.0] * len(TC_RANGE)
    hilo = build_report("hilo", BasicStrategyAgent(env.rules, count_bets=True), env, 0.0, 0.6, None)
    assert hilo.agent_bets == hilo.hilo_bets and hilo.hilo_bets[-1] == 8.0 and hilo.hilo_bets[0] == 1.0
    txt = render_text(rep)
    assert "HARD" in txt and "A,7" in txt and "BET by true count" in txt


def test_render_html_is_self_contained():
    env = BlackjackEnv()
    rep = build_report("hilo", BasicStrategyAgent(env.rules, count_bets=True), env, 0.0, 0.6, None)
    page = render_html(rep)
    assert page.startswith("<!doctype html>")
    assert "<style>" in page and "<svg" in page and "Pairs" in page and "Hi-Lo bet spread" in page
    assert "http://" not in page and "https://" not in page   # no external resources
    assert page.count("<td class=") == 330


def test_cli_writes_html(tmp_path):
    out = tmp_path / "r" / "strategy.html"
    main(["--agent", "basic", "--html", str(out), "--quiet"])
    assert out.exists() and out.stat().st_size > 5000
    assert "Blackjack strategy" in out.read_text(encoding="utf-8")
