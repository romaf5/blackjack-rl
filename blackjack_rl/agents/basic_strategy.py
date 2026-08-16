"""Basic strategy (multi-deck) + optional Hi-Lo bet spread.

This is the benchmark the RL agent should converge towards. Basic strategy alone
plays at roughly -0.6% of the wager per round on 6D H17 DAS LS (-0.4% with S17);
adding a count-driven bet spread flips the edge slightly positive.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from ..engine import Action, Rules
from ..env.blackjack_env import N_PLAY_ACTIONS
from .base import Agent


def _pick(preferred: Action, fallback: Action, legal: Iterable[Action]) -> Action:
    return preferred if preferred in legal else fallback


def basic_strategy(
    total: int,
    is_soft: bool,
    is_pair: bool,
    dealer_up: int,
    legal: Iterable[Action],
    rules: Optional[Rules] = None,
) -> Action:
    """Return the basic-strategy action.

    ``dealer_up`` is the dealer's up-card blackjack value 1..10 (1 = Ace).
    ``legal`` is the set of currently legal actions; the table's first choice is
    replaced by its usual fallback when it isn't available (e.g. no double on 3+ cards).
    """
    rules = rules or Rules()
    legal = set(legal)
    h17 = rules.dealer_hits_soft_17
    das = rules.double_after_split
    d = 11 if dealer_up == 1 else dealer_up   # 2..11 for table lookups
    S, H, D, P, R = Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT, Action.SURRENDER

    def Dh():  # double, else hit
        return _pick(D, H, legal)

    def Ds():  # double, else stand
        return _pick(D, S, legal)

    def Rh():  # surrender, else hit
        return _pick(R, H, legal)

    def Rs():  # surrender, else stand
        return _pick(R, S, legal)

    # ---------------------------------------------------------------- pairs
    if is_pair and P in legal:
        pair = 1 if (is_soft and total == 12) else total // 2
        if pair == 1:
            return P
        if pair == 10:
            return S
        if pair == 9:
            return P if d in (2, 3, 4, 5, 6, 8, 9) else S
        if pair == 8:
            if h17 and d == 11 and R in legal:
                return R
            return P
        if pair == 7:
            return P if d <= 7 else H
        if pair == 6:
            return P if (2 if das else 3) <= d <= 6 else H
        if pair == 5:
            pass  # play as hard 10 below
        if pair == 4:
            return P if (das and d in (5, 6)) else H
        if pair in (2, 3):
            return P if (2 if das else 4) <= d <= 7 else H

    # ---------------------------------------------------------------- soft totals
    if is_soft:
        if total >= 20:
            return S
        if total == 19:
            return Ds() if (h17 and d == 6) else S
        if total == 18:
            if 3 <= d <= 6 or (h17 and d == 2):
                return Ds()
            return S if d in (2, 7, 8) else H
        if total == 17:
            return Dh() if 3 <= d <= 6 else H
        if total in (15, 16):
            return Dh() if 4 <= d <= 6 else H
        if total in (13, 14):
            return Dh() if 5 <= d <= 6 else H
        return H  # soft 12 (A,A that couldn't be split)

    # ---------------------------------------------------------------- hard totals
    if total >= 18:
        return S
    if total == 17:
        return Rs() if (h17 and d == 11) else S
    if total == 16:
        if d <= 6:
            return S
        return Rh() if d >= 9 else H
    if total == 15:
        if d <= 6:
            return S
        if d == 10 or (h17 and d == 11):
            return Rh()
        return H
    if total in (13, 14):
        return S if d <= 6 else H
    if total == 12:
        return S if 4 <= d <= 6 else H
    if total == 11:
        return Dh() if (d <= 10 or h17) else H
    if total == 10:
        return Dh() if d <= 9 else H
    if total == 9:
        return Dh() if 3 <= d <= 6 else H
    return H


def hi_lo_bet_index(true_count: float, bet_sizes: Sequence[float]) -> int:
    """Classic 1-2-4-8 style spread on the true count, mapped onto the available bet sizes.

    TC <= 1 -> 1 unit, TC 2 -> 2 units, TC 3 -> 4 units, TC >= 4 -> 8 units
    (relative to the smallest bet). Picks the largest available bet not exceeding the target.
    """
    tc = int(np.floor(true_count))
    units = {2: 2, 3: 4}.get(tc, 1 if tc <= 1 else 8)
    target = min(bet_sizes) * units
    best = int(np.argmin(bet_sizes))
    for i, b in enumerate(bet_sizes):
        if b <= target + 1e-9 and b >= bet_sizes[best]:
            best = i
    return best


class BasicStrategyAgent(Agent):
    """Plays perfect basic strategy; bets flat (min) or with a Hi-Lo spread."""

    def __init__(self, rules: Optional[Rules] = None, count_bets: bool = False):
        self.rules = rules or Rules()
        self.count_bets = count_bets
        self.name = "basic+hilo" if count_bets else "basic"

    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        if info["phase"] == "bet":
            sizes = info["bet_sizes"]
            idx = hi_lo_bet_index(info["true_count"], sizes) if self.count_bets else 0
            return N_PLAY_ACTIONS + idx
        legal = [Action(a) for a in info["legal_actions"]]
        return int(basic_strategy(
            info["player_total"], info["is_soft"], info["is_pair"], info["dealer_upcard"], legal, self.rules
        ))
