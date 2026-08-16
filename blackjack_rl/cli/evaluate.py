"""Evaluate an agent (random / basic / hilo / ppo) over many rounds."""
from __future__ import annotations

import argparse

from ..evaluation import evaluate
from .common import add_rules_args, env_from_args, make_agent


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a blackjack agent.")
    add_rules_args(p)
    p.add_argument("--agent", choices=["random", "basic", "hilo", "ppo"], default="basic")
    p.add_argument("--checkpoint", type=str, default=None, help="path to a PPO checkpoint (.pt)")
    p.add_argument("--rounds", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress", type=int, default=0, help="print progress every N rounds")
    a = p.parse_args(argv)

    env = env_from_args(a)
    agent = make_agent(a.agent, env, a.checkpoint, seed=a.seed)
    print(f"Rules: {env.rules.describe()} | bets {env.bet_sizes}")
    stats = evaluate(agent, env, a.rounds, seed=a.seed, progress_every=a.progress)
    print(stats.summary(getattr(agent, "name", a.agent)))


if __name__ == "__main__":
    main()
