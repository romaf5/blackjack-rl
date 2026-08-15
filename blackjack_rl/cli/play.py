"""Play blackjack yourself, driving the exact same Gymnasium env the RL agent trains on."""
from __future__ import annotations

import argparse
import sys

from ..agents.basic_strategy import basic_strategy, hi_lo_bet_index
from ..engine import ACTION_NAMES, Action
from ..env.blackjack_env import N_PLAY_ACTIONS
from .common import add_rules_args, env_from_args

KEYS = {"s": Action.STAND, "h": Action.HIT, "d": Action.DOUBLE, "p": Action.SPLIT, "r": Action.SURRENDER}
LABELS = {Action.STAND: "(s)tand", Action.HIT: "(h)it", Action.DOUBLE: "(d)ouble",
          Action.SPLIT: "s(p)lit", Action.SURRENDER: "su(r)render"}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Play blackjack in the terminal (through the RL env).")
    add_rules_args(p)
    p.add_argument("--bankroll", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hide-count", action="store_true", help="don't show the running/true count")
    p.add_argument("--hint", action="store_true", help="show what basic strategy / Hi-Lo betting would do")
    a = p.parse_args(argv)

    env = env_from_args(a, seed=a.seed)
    bankroll = a.bankroll
    rounds = 0
    print(f"Rules: {env.rules.describe()}")
    print(f"Bet sizes: {', '.join(f'{b:g}' for b in env.bet_sizes)}   |   bankroll {bankroll:g}")
    print("Type q at any prompt to quit.\n")

    def ask(prompt: str) -> str:
        try:
            s = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "q"
        return s

    def show():
        text = env.render()
        if a.hide_count:
            text = "\n".join(l for l in text.splitlines() if not l.startswith("Count:"))
        print(text)

    obs, info = env.reset()
    while True:
        rounds += 1
        print(f"\n===== Round {rounds}   bankroll {bankroll:g} =====")
        if not a.hide_count:
            print(f"Count: running {info['running_count']:+d}, true {info['true_count']:+.1f} "
                  f"({info['decks_remaining']:.1f} decks left)")
        # ---- bet phase
        options = " ".join(f"[{i + 1}]={b:g}" for i, b in enumerate(env.bet_sizes))
        hint = ""
        if a.hint:
            hint = f"   (Hi-Lo suggests {env.bet_sizes[hi_lo_bet_index(info['true_count'], env.bet_sizes)]:g})"
        while True:
            s = ask(f"Bet {options}{hint} > ")
            if s == "q":
                print(f"Thanks for playing. Rounds: {rounds - 1}, final bankroll {bankroll:g}")
                return
            if s.isdigit() and 1 <= int(s) <= len(env.bet_sizes):
                bet_action = N_PLAY_ACTIONS + int(s) - 1
                break
            if s == "":
                bet_action = N_PLAY_ACTIONS  # enter = minimum bet
                break
            print("  ? choose a bet number")
        obs, reward, done, _, info = env.step(bet_action)

        # ---- play phase
        while not done:
            print()
            show()
            legal = [Action(x) for x in info["legal_actions"]]
            prompt = "  ".join(LABELS[x] for x in legal)
            hint = ""
            if a.hint:
                bs = basic_strategy(info["player_total"], info["is_soft"], info["is_pair"],
                                    info["dealer_upcard"], legal, env.rules)
                hint = f"   [basic strategy: {ACTION_NAMES[bs]}]"
            while True:
                s = ask(f"{prompt}{hint} > ")
                if s == "q":
                    print(f"Thanks for playing. Rounds: {rounds - 1}, final bankroll {bankroll:g}")
                    return
                act = KEYS.get(s[:1]) if s else None
                if act is not None and act in legal:
                    break
                print("  ? not a legal action")
            obs, reward, done, _, info = env.step(int(act))

        # ---- settlement
        print()
        show()
        bankroll += reward
        print(f">>> Round result: {reward:+g}   bankroll {bankroll:g}")
        if info.get("shuffled"):
            print("(the shoe was shuffled before this round)")
        obs, info = env.reset()


if __name__ == "__main__":
    main()
