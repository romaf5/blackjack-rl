"""Shared argparse helpers for the command-line tools."""
from __future__ import annotations

import argparse
from typing import Tuple

from ..engine import Rules
from ..env import BlackjackEnv


def add_rules_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("table rules")
    g.add_argument("--decks", type=int, default=6)
    g.add_argument("--penetration", type=float, default=0.75, help="fraction of shoe dealt before reshuffle")
    g.add_argument("--h17", action="store_true", help="dealer hits soft 17 (default: stands)")
    g.add_argument("--bj-payout", type=float, default=1.5)
    g.add_argument("--no-das", action="store_true", help="no doubling after split")
    g.add_argument("--double-on", type=str, default=None, help="e.g. 9,10,11 to restrict doubling (default: any)")
    g.add_argument("--max-splits", type=int, default=3)
    g.add_argument("--rsa", action="store_true", help="allow re-splitting aces")
    g.add_argument("--hit-split-aces", action="store_true")
    g.add_argument("--no-surrender", action="store_true")
    g.add_argument("--no-peek", action="store_true", help="dealer does not check for blackjack first")
    g.add_argument("--bets", type=str, default="1,2,4,8", help="comma-separated bet sizes (units)")
    g.add_argument("--reshuffle-each-round", action="store_true", help="disable card counting (fresh shoe every round)")


def rules_from_args(a: argparse.Namespace) -> Rules:
    return Rules(
        num_decks=a.decks,
        penetration=a.penetration,
        dealer_hits_soft_17=a.h17,
        blackjack_payout=a.bj_payout,
        dealer_peeks=not a.no_peek,
        double_after_split=not a.no_das,
        double_on=tuple(int(x) for x in a.double_on.split(",")) if a.double_on else None,
        max_splits=a.max_splits,
        resplit_aces=a.rsa,
        hit_split_aces=a.hit_split_aces,
        surrender=not a.no_surrender,
    )


def bets_from_args(a: argparse.Namespace) -> Tuple[float, ...]:
    return tuple(float(x) for x in a.bets.split(","))


def env_from_args(a: argparse.Namespace, seed=None) -> BlackjackEnv:
    return BlackjackEnv(rules=rules_from_args(a), bet_sizes=bets_from_args(a),
                        reshuffle_each_round=a.reshuffle_each_round, seed=seed)


def make_agent(name: str, env: BlackjackEnv, checkpoint: str = None, seed=None):
    if name == "random":
        from ..agents import RandomAgent
        return RandomAgent(seed=seed)
    if name == "basic":
        from ..agents import BasicStrategyAgent
        return BasicStrategyAgent(env.rules, count_bets=False)
    if name == "hilo":
        from ..agents import BasicStrategyAgent
        return BasicStrategyAgent(env.rules, count_bets=True)
    if name == "dqn":
        if not checkpoint:
            raise SystemExit("--checkpoint is required for the dqn agent")
        from ..agents.dqn import DQNAgent
        agent = DQNAgent.load(checkpoint)
        if agent.n_actions != env.action_space.n:
            raise SystemExit(f"checkpoint was trained with {agent.n_actions - 5} bet sizes, env has {len(env.bet_sizes)}")
        return agent
    raise SystemExit(f"unknown agent {name!r}")
