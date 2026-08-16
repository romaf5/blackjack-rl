"""Train the masked PPO agent on the vectorized env (uses the Apple GPU / CUDA when available)."""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..agents.ppo import PPOAgent, PPOConfig, pick_device, train_ppo
from ..evaluation import evaluate
from .common import add_rules_args, bets_from_args, env_from_args, rules_from_args


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train a PPO blackjack agent (vectorized env).")
    add_rules_args(p)
    p.add_argument("--total-rounds", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--hidden", type=str, default="256,256")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-coef-end", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-rounds", type=int, default=100_000)
    p.add_argument("--final-eval-rounds", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto (mps > cuda > cpu), mps, cuda or cpu")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--out", type=str, default="checkpoints/ppo.pt")
    a = p.parse_args(argv)

    rules = rules_from_args(a)
    bets = bets_from_args(a)
    cfg = PPOConfig(
        total_rounds=a.total_rounds, num_envs=a.num_envs, workers=a.workers, num_steps=a.num_steps,
        hidden=tuple(int(x) for x in a.hidden.split(",")), lr=a.lr, lr_end=a.lr_end, gamma=a.gamma,
        gae_lambda=a.gae_lambda, epochs=a.epochs, minibatches=a.minibatches, clip_coef=a.clip,
        ent_coef=a.ent_coef, ent_coef_end=a.ent_coef_end, vf_coef=a.vf_coef, log_every=a.log_every,
        eval_every=a.eval_every, eval_rounds=a.eval_rounds, seed=a.seed, device=a.device,
    )
    dev = pick_device(a.device)
    print(f"Rules: {rules.describe()} | bets {bets}")
    print(f"Device: {dev} (torch {torch.__version__}; mps available: {torch.backends.mps.is_available()})")
    print(f"Config: {json.dumps(cfg.to_dict())}")
    agent = None
    if a.resume:
        agent = PPOAgent.load(a.resume, device=str(dev))
        print(f"Resuming from {a.resume}")

    t0 = time.time()
    agent, history = train_ppo(cfg, rules=rules, bet_sizes=bets, agent=agent, reshuffle_each_round=a.reshuffle_each_round)
    print(f"Training done in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    agent.save(a.out, extra={"config": cfg.to_dict(), "rules": rules.__dict__, "bet_sizes": bets, "history": history})
    with open(os.path.splitext(a.out)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"Saved checkpoint to {a.out}")

    if a.final_eval_rounds > 0:
        print(f"\nFinal greedy evaluation over {a.final_eval_rounds:,} rounds (single env, CPU) ...")
        agent.to("cpu")
        env = env_from_args(a)
        stats = evaluate(agent, env, a.final_eval_rounds, seed=999)
        print(stats.summary("ppo"))


if __name__ == "__main__":
    main()
