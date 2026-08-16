"""Configurable table rules."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class Rules:
    num_decks: int = 6
    penetration: float = 0.75          # fraction of the shoe dealt before a reshuffle
    dealer_hits_soft_17: bool = True    # H17 (the usual US rule); False = S17, dealer stands on all 17s
    blackjack_payout: float = 1.5       # 3:2. Use 1.2 for the (bad) 6:5 tables.
    dealer_peeks: bool = True           # dealer checks for blackjack before the player acts
    double_after_split: bool = True     # DAS
    double_on: Optional[Tuple[int, ...]] = None  # None = any two cards; e.g. (9, 10, 11) = Reno rule
    max_splits: int = 3                 # up to 4 hands
    resplit_aces: bool = False
    hit_split_aces: bool = False        # normally split aces receive exactly one card
    surrender: bool = True              # late surrender (only after the dealer has peeked)

    @property
    def max_hands(self) -> int:
        return self.max_splits + 1

    def with_(self, **changes) -> "Rules":
        return replace(self, **changes)

    def describe(self) -> str:
        parts = [
            f"{self.num_decks} deck{'s' if self.num_decks != 1 else ''}",
            "H17" if self.dealer_hits_soft_17 else "S17",
            f"BJ pays {self.blackjack_payout:g}:1",
            "DAS" if self.double_after_split else "no DAS",
            "double any" if self.double_on is None else f"double on {self.double_on}",
            f"resplit to {self.max_hands}",
            "RSA" if self.resplit_aces else "no RSA",
            "late surrender" if self.surrender else "no surrender",
            "peek" if self.dealer_peeks else "no peek",
            f"{self.penetration:.0%} pen",
        ]
        return ", ".join(parts)


# A few common presets
VEGAS_STRIP = Rules(num_decks=6, dealer_hits_soft_17=False, double_after_split=True, surrender=True)   # S17 (high-limit style)
VEGAS_DOWNTOWN = Rules(num_decks=6, dealer_hits_soft_17=True, double_after_split=True, surrender=True)  # H17 (= defaults)
SINGLE_DECK = Rules(num_decks=1, penetration=0.6, dealer_hits_soft_17=True, double_on=(9, 10, 11), surrender=False)
