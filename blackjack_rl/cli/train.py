"""Train the masked Double-DQN agent."""
from __future__ import annotations

import argparse
import json
import os
import time

from ..agents.dqn import DQNAgent, TrainConfig, train_dqn
from ..evaluation import evaluate
from .common import add_rules_args, env_from_args


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train a DQN blackjack agent.")
    add_rules_args(p)
    p.add_argument("--rounds", type=int, default=200_000)
    p.add_argument("--hidden", type=str, default="256,256")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-end", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--train-every", type=int, default=2)
    p.add_argument("--target-sync-every", type=int, default=2_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-rounds", type=int, default=20_000)
    p.add_argument("--final-eval-rounds", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="continue training from a checkpoint")
    p.add_argument("--out", type=str, default="checkpoints/dqn.pt")
    a = p.parse_args(argv)

    env = env_from_args(a)
    cfg = TrainConfig(
        rounds=a.rounds, hidden=tuple(int(x) for x in a.hidden.split(",")), lr=a.lr, lr_end=a.lr_end,
        batch_size=a.batch_size, buffer_size=a.buffer_size, train_every=a.train_every,
        target_sync_every=a.target_sync_every, eps_start=a.eps_start, eps_end=a.eps_end,
        eps_decay_frac=a.eps_decay_frac, log_every=a.log_every, eval_every=a.eval_every,
        eval_rounds=a.eval_rounds, seed=a.seed, device=a.device,
    )
    print(f"Rules: {env.rules.describe()} | bets {env.bet_sizes}")
    print(f"Config: {json.dumps(cfg.to_dict())}")
    agent = DQNAgent.load(a.resume) if a.resume else None
    if agent is not None:
        agent.q.train()
        agent.set_lr(cfg.lr)
        print(f"Resuming from {a.resume}")

    t0 = time.time()
    agent, history = train_dqn(env, cfg, agent=agent)
    print(f"Training done in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    agent.save(a.out, extra={"config": cfg.to_dict(), "rules": env.rules.__dict__, "bet_sizes": env.bet_sizes,
                             "history": history})
    with open(os.path.splitext(a.out)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"Saved checkpoint to {a.out}")

    if a.final_eval_rounds > 0:
        print(f"\nFinal greedy evaluation over {a.final_eval_rounds:,} rounds ...")
        stats = evaluate(agent, env, a.final_eval_rounds, seed=999)
        print(stats.summary("dqn"))


if __name__ == "__main__":
    main()
