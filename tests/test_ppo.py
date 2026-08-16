import numpy as np
import pytest
import torch

from blackjack_rl.agents import load_rl_agent
from blackjack_rl.agents.ppo import ActorCritic, PPOAgent, PPOConfig, evaluate_greedy_batched, train_ppo
from blackjack_rl.cli.strategy import build_report, render_html
from blackjack_rl.env import BlackjackEnv, OBS_DIM
from blackjack_rl.env.vector_env import BlackjackVectorEnv
from blackjack_rl.evaluation import evaluate
from blackjack_rl.web.session import GameSession


def test_masking_never_picks_illegal_actions():
    torch.manual_seed(0)
    net = ActorCritic(OBS_DIM, 9)
    obs = torch.rand(64, OBS_DIM)
    mask = torch.zeros(64, 9, dtype=torch.bool)
    mask[:, [1, 4, 6]] = True
    for _ in range(5):
        a, logp, v = net.act(obs, mask)
        assert set(a.tolist()) <= {1, 4, 6}
        assert torch.isfinite(logp).all() and torch.isfinite(v).all()
    logp2, ent, v2 = net.evaluate(obs, mask, a)
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert torch.isfinite(ent).all() and (ent >= 0).all() and (ent <= np.log(3) + 1e-4).all()


def test_ppo_smoke_train_and_roundtrip(tmp_path):
    cfg = PPOConfig(total_rounds=3000, num_envs=64, workers=1, num_steps=8, minibatches=2, epochs=1,
                    log_every=1, eval_every=0, device="cpu", seed=0, hidden=(32, 32))
    agent, history = train_ppo(cfg)
    assert history and history[-1]["rounds"] >= 3000
    # batched greedy eval
    venv = BlackjackVectorEnv(32, seed=1)
    st = evaluate_greedy_batched(agent, venv, 300)
    venv.close()
    assert st.rounds >= 300 and st.total_wagered > 0
    # single-env eval through the generic Agent interface
    env = BlackjackEnv()
    stats = evaluate(agent, env, 200, seed=2)
    assert stats.rounds == 200
    # save / load / dispatch by kind
    path = tmp_path / "ppo.pt"
    agent.save(str(path), extra={"note": "test"})
    loaded = load_rl_agent(str(path))
    assert isinstance(loaded, PPOAgent) and loaded.name == "ppo"
    obs, info = env.reset(seed=3)
    p = loaded.action_probs(obs, info["action_mask"])
    assert p.shape == (9,) and abs(p.sum() - 1) < 1e-5 and (p[:5] == 0).all()
    assert loaded.greedy_action(obs, info["action_mask"]) == agent.greedy_action(obs, info["action_mask"])
    # strategy report + web session accept the PPO checkpoint
    rep = build_report("ppo", loaded, env, 0.0, 0.6, str(path))
    assert rep.agent_name == "ppo" and rep.bet_q is not None
    page = render_html(rep)
    assert "PPO agent" in page and "Policy probability" in page
    s = GameSession(bankroll=100, seed=4, checkpoint=str(path))
    assert s.state()["rl_agents"] == [{"key": "ppo", "kind": "ppo", "checkpoint": str(path), "usable": True, "error": None}]
    assert s.state()["agents"] == ["basic", "hilo", "ppo"]
    adv = s.advice()
    assert abs(sum(e["prob"] for e in adv["rl_agents"]["ppo"]["probs"]) - 1) < 1e-5
    st2 = s.agent_step("ppo")
    assert st2["agent_action"].startswith("bet")
    for name in ("ppo", "rl"):
        page = s.strategy_report_html(name, 1.5)
        assert "PPO agent" in page and "Bet size by true count" in page
