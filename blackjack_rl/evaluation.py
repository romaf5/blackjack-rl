"""Evaluate an agent over many rounds and summarise the results."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .agents.base import Agent
from .env import BlackjackEnv


@dataclass
class EvalStats:
    rounds: int = 0
    total_profit: float = 0.0
    total_wagered: float = 0.0
    sum_sq_profit: float = 0.0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    blackjacks: int = 0
    surrenders: int = 0
    doubles: int = 0
    splits: int = 0
    steps: int = 0
    bet_by_tc: Dict[int, list] = field(default_factory=lambda: defaultdict(list))
    profit_by_tc: Dict[int, list] = field(default_factory=lambda: defaultdict(list))

    @property
    def ev_per_round(self) -> float:
        return self.total_profit / max(self.rounds, 1)

    @property
    def ev_per_unit_wagered(self) -> float:
        return self.total_profit / max(self.total_wagered, 1e-9)

    @property
    def std_per_round(self) -> float:
        n = max(self.rounds, 1)
        var = self.sum_sq_profit / n - self.ev_per_round ** 2
        return math.sqrt(max(var, 0.0))

    @property
    def stderr(self) -> float:
        return self.std_per_round / math.sqrt(max(self.rounds, 1))

    @property
    def mean_bet(self) -> float:
        return self.total_wagered / max(self.rounds, 1)

    def summary(self, name: str = "") -> str:
        n = max(self.rounds, 1)
        lines = [
            f"== {name or 'agent'}: {self.rounds:,} rounds ==",
            f"  EV/round        : {self.ev_per_round:+.4f} units  (± {1.96 * self.stderr:.4f} 95% CI)",
            f"  EV/unit wagered : {100 * self.ev_per_unit_wagered:+.3f} %",
            f"  mean wager/round: {self.mean_bet:.3f}   std/round: {self.std_per_round:.3f}",
            f"  win {100 * self.wins / n:.1f}%  lose {100 * self.losses / n:.1f}%  push {100 * self.pushes / n:.1f}%  "
            f"(per hand; bj {100 * self.blackjacks / n:.1f}%, dbl {100 * self.doubles / n:.1f}%, "
            f"split {100 * self.splits / n:.1f}%, surr {100 * self.surrenders / n:.1f}% of rounds)",
        ]
        if len(self.bet_by_tc) > 1:
            lines.append("  mean bet by true count (rounded down):")
            for tc in sorted(self.bet_by_tc):
                bets = self.bet_by_tc[tc]
                if len(bets) < 30:
                    continue
                p = self.profit_by_tc[tc]
                lines.append(f"    TC {tc:+3d}: bet {np.mean(bets):5.2f}   n={len(bets):>7,}   EV/round {np.mean(p):+.3f}")
        return "\n".join(lines)


def evaluate(agent: Agent, env: BlackjackEnv, rounds: int, seed: Optional[int] = None,
             progress_every: int = 0) -> EvalStats:
    stats = EvalStats()
    obs, info = env.reset(seed=seed)
    for i in range(rounds):
        if i > 0:
            obs, info = env.reset()
        tc_bucket = int(math.floor(info["true_count"]))
        done = False
        used_double = used_split = used_surrender = False
        while not done:
            action = agent.act(obs, info)
            obs, reward, done, _, info = env.step(action)
            stats.steps += 1
            if action == 2:
                used_double = True
            elif action == 3:
                used_split = True
            elif action == 4:
                used_surrender = True
        agent.on_round_end(info)
        stats.rounds += 1
        stats.total_profit += reward
        stats.sum_sq_profit += reward * reward
        stats.total_wagered += info["total_wagered"]
        stats.doubles += used_double
        stats.splits += used_split
        stats.surrenders += used_surrender
        for label, profit in info["results"]:
            if "blackjack!" in label:
                stats.blackjacks += 1
            if profit > 0:
                stats.wins += 1
            elif profit < 0:
                stats.losses += 1
            else:
                stats.pushes += 1
        stats.bet_by_tc[tc_bucket].append(info["bet"])
        stats.profit_by_tc[tc_bucket].append(reward)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... {i + 1:,}/{rounds:,} rounds, EV/round so far {stats.ev_per_round:+.4f}")
    return stats
