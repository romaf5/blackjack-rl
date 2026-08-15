"""Cards and the dealing shoe (with Hi-Lo running/true count)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}
SUITS = ("♠", "♥", "♦", "♣")

# Hi-Lo tag keyed by blackjack *value* (1 = Ace, 10 = ten/face).
HI_LO = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}


@dataclass(frozen=True)
class Card:
    rank: int  # 1 = Ace, 2..10 = pips, 11 = J, 12 = Q, 13 = K
    suit: str = "♠"

    @property
    def value(self) -> int:
        """Blackjack value with Ace counted as 1 (soft handling lives in Hand)."""
        return min(self.rank, 10)

    @property
    def is_ace(self) -> bool:
        return self.rank == 1

    @property
    def name(self) -> str:
        return RANK_NAMES.get(self.rank, str(self.rank))

    def __str__(self) -> str:
        return f"{self.name}{self.suit}"


class Shoe:
    """A multi-deck shoe with a cut card and Hi-Lo counting.

    ``draw(visible=True)`` updates the running count immediately; the dealer's
    hole card is drawn with ``visible=False`` and counted later via ``observe``.
    """

    def __init__(self, num_decks: int = 6, penetration: float = 0.75, rng: Optional[random.Random] = None):
        if not 0 < penetration <= 1:
            raise ValueError("penetration must be in (0, 1]")
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = rng or random.Random()
        self.total_cards = num_decks * 52
        self.cut_card = int(self.total_cards * penetration)
        self._cards: List[Card] = []
        self._pos = 0
        self.running_count = 0
        self.num_shuffles = 0
        self.shuffle()

    def shuffle(self) -> None:
        self._cards = [Card(r, s) for _ in range(self.num_decks) for s in SUITS for r in range(1, 14)]
        self.rng.shuffle(self._cards)
        self._pos = 0
        self.running_count = 0
        self.num_shuffles += 1

    @property
    def cards_remaining(self) -> int:
        return len(self._cards) - self._pos

    @property
    def cards_dealt(self) -> int:
        return self._pos

    @property
    def needs_shuffle(self) -> bool:
        """True once the cut card has been passed (checked between rounds)."""
        return self._pos >= self.cut_card

    @property
    def decks_remaining(self) -> float:
        return self.cards_remaining / 52.0

    @property
    def true_count(self) -> float:
        # Floor the divisor at half a deck so the true count doesn't explode at the end of a shoe.
        return self.running_count / max(self.decks_remaining, 0.5)

    def draw(self, visible: bool = True) -> Card:
        if self._pos >= len(self._cards):
            # Extremely rare (deep penetration + many splits): shuffle and continue.
            self.shuffle()
        card = self._cards[self._pos]
        self._pos += 1
        if visible:
            self.observe(card)
        return card

    def observe(self, card: Card) -> None:
        """Register a card as seen by the player (updates the running count)."""
        self.running_count += HI_LO[card.value]
