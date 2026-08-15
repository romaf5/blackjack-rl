"""Print an agent's playing strategy as the classic hard/soft/pairs tables and its bet spread,
and compare it against basic strategy."""
from __future__ import annotations

import argparse
from typing import Callable, List, Tuple

import numpy as np

from ..agents.basic_strategy import basic_strategy, hi_lo_bet_index
from ..engine import ACTION_LETTERS, Action, Rules
from ..env.blackjack_env import N_PLAY_ACTIONS
from ..env.observation import encode_observation
from .common import add_rules_args, env_from_args, make_agent

DEALER_UPS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]  # 1 = Ace, shown last
DEALER_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]


def _legal_for(rules: Rules, is_pair: bool) -> List[Action]:
    legal = [Action.STAND, Action.HIT, Action.DOUBLE]
    if is_pair:
        legal.append(Action.SPLIT)
    if rules.surrender:
        legal.append(Action.SURRENDER)
    return legal


def _mask(legal: List[Action], n_actions: int) -> np.ndarray:
    m = np.zeros(n_actions, dtype=np.int8)
    m[[int(a) for a in legal]] = 1
    return m


def policy_tables(decide: Callable[[int, bool, bool, int, List[Action]], Action], rules: Rules):
    """Return (hard, soft, pairs) dicts of {(row, dealer): letter}."""
    hard, soft, pairs = {}, {}, {}
    for total in range(5, 20):
        for d in DEALER_UPS:
            hard[(total, d)] = ACTION_LETTERS[decide(total, False, False, d, _legal_for(rules, False))]
    for total in range(13, 21):
        for d in DEALER_UPS:
            soft[(total, d)] = ACTION_LETTERS[decide(total, True, False, d, _legal_for(rules, False))]
    for pair in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]:
        total, is_soft = (12, True) if pair == 1 else (2 * pair, False)
        for d in DEALER_UPS:
            pairs[(pair, d)] = ACTION_LETTERS[decide(total, is_soft, True, d, _legal_for(rules, True))]
    return hard, soft, pairs


def _fmt_table(title: str, rows: List[Tuple[str, object]], table: dict, ref: dict = None) -> str:
    out = [f"{title:<12}" + " ".join(f"{d:>3}" for d in DEALER_LABELS)]
    mismatches = 0
    cells = 0
    for label, key in rows:
        line = f"{label:<12}"
        for d in DEALER_UPS:
            v = table[(key, d)]
            cells += 1
            if ref is not None and ref[(key, d)] != v:
                mismatches += 1
                line += f" {v}/{ref[(key, d)]}"[:4].rjust(4)
            else:
                line += f"{v:>4}"
        out.append(line)
    if ref is not None:
        out.append(f"{'':<12}agreement with basic strategy: {cells - mismatches}/{cells} "
                   f"({100 * (cells - mismatches) / cells:.1f}%)   (cells shown as agent/basic where they differ)")
    return "\n".join(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Show an agent's strategy tables and bet spread.")
    add_rules_args(p)
    p.add_argument("--agent", choices=["basic", "hilo", "dqn"], default="dqn")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--true-count", type=float, default=0.0, help="true count to assume for the play tables")
    p.add_argument("--decks-frac", type=float, default=0.6, help="fraction of shoe remaining to assume")
    p.add_argument("--no-compare", action="store_true", help="don't diff against basic strategy")
    a = p.parse_args(argv)

    env = env_from_args(a)
    rules = env.rules
    agent = make_agent(a.agent, env, a.checkpoint)
    n_actions = env.action_space.n
    bet_frac = env.bet_sizes[0] / env.max_bet

    if a.agent == "dqn":
        def decide(total, is_soft, is_pair, dealer, legal):
            obs = encode_observation(
                phase=1, player_total=total, is_soft=is_soft, is_pair=is_pair,
                can_double=Action.DOUBLE in legal, can_split=Action.SPLIT in legal,
                can_surrender=Action.SURRENDER in legal, is_split_hand=False, dealer_upcard=dealer,
                true_count=a.true_count, decks_frac=a.decks_frac, bet_frac=bet_frac,
                num_hands=1, max_splits=rules.max_splits)
            return Action(agent.greedy_action(obs, _mask(legal, n_actions)))
    else:
        def decide(total, is_soft, is_pair, dealer, legal):
            return basic_strategy(total, is_soft, is_pair, dealer, legal, rules)

    def basic_decide(total, is_soft, is_pair, dealer, legal):
        return basic_strategy(total, is_soft, is_pair, dealer, legal, rules)

    hard, soft, pairs = policy_tables(decide, rules)
    ref = None if (a.no_compare or a.agent in ("basic", "hilo")) else policy_tables(basic_decide, rules)

    print(f"Rules: {rules.describe()}")
    print(f"Agent: {a.agent}{' (' + a.checkpoint + ')' if a.checkpoint else ''}   "
          f"assumed true count {a.true_count:+.1f}, {a.decks_frac:.0%} of shoe left")
    print("Legend: S=stand H=hit D=double P=split R=surrender  (2-card hands; dealer up-card across the top)\n")
    print(_fmt_table("HARD", [(str(t), t) for t in range(5, 20)], hard, ref and ref[0]))
    print()
    print(_fmt_table("SOFT", [(f"A,{t - 11}", t) for t in range(13, 21)], soft, ref and ref[1]))
    print()
    print(_fmt_table("PAIRS", [(("A,A" if pr == 1 else f"{'T' if pr == 10 else pr},{'T' if pr == 10 else pr}"), pr)
                               for pr in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]], pairs, ref and ref[2]))

    # ---- bet spread by true count
    print("\nBET by true count:")
    header = "TC        " + " ".join(f"{tc:>+5d}" for tc in range(-5, 9))
    print(header)
    if a.agent == "dqn":
        bet_mask = np.zeros(n_actions, dtype=np.int8)
        bet_mask[N_PLAY_ACTIONS:] = 1
        row = "agent     "
        qrows = []
        for tc in range(-5, 9):
            obs = encode_observation(phase=0, true_count=tc, decks_frac=a.decks_frac, bet_frac=0.0)
            act = agent.greedy_action(obs, bet_mask)
            row += f"{env.bet_sizes[act - N_PLAY_ACTIONS]:>5g} "
            q = agent.q_values(obs)[N_PLAY_ACTIONS:]
            qrows.append(q)
        print(row)
        print("hi-lo ref " + " ".join(f"{env.bet_sizes[hi_lo_bet_index(tc, env.bet_sizes)]:>5g}" for tc in range(-5, 9)))
        print("\nQ-values (scaled reward units) per bet size:")
        for i, b in enumerate(env.bet_sizes):
            print(f"bet {b:<6g}" + " ".join(f"{q[i]:>+5.3f}" for q in qrows))
    else:
        row = "hi-lo     " if a.agent == "hilo" else "flat      "
        for tc in range(-5, 9):
            idx = hi_lo_bet_index(tc, env.bet_sizes) if a.agent == "hilo" else 0
            row += f"{env.bet_sizes[idx]:>5g} "
        print(row)


if __name__ == "__main__":
    main()
