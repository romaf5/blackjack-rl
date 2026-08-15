"""A blackjack hand (player or dealer)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .cards import Card


@dataclass
class Hand:
    cards: List[Card] = field(default_factory=list)
    bet: float = 0.0
    is_split: bool = False          # hand was created by (or has been) split
    from_split_aces: bool = False   # hand is one of the halves of split aces
    is_doubled: bool = False
    is_surrendered: bool = False
    finished: bool = False          # player can no longer act on this hand

    def add(self, card: Card) -> None:
        self.cards.append(card)

    @property
    def num_cards(self) -> int:
        return len(self.cards)

    @property
    def hard_total(self) -> int:
        """Total with every Ace counted as 1."""
        return sum(c.value for c in self.cards)

    @property
    def is_soft(self) -> bool:
        """True if an Ace is currently being counted as 11."""
        return any(c.is_ace for c in self.cards) and self.hard_total + 10 <= 21

    @property
    def total(self) -> int:
        return self.hard_total + 10 if self.is_soft else self.hard_total

    @property
    def is_bust(self) -> bool:
        return self.hard_total > 21

    @property
    def is_blackjack(self) -> bool:
        """A natural: two cards totalling 21 on an un-split hand."""
        return self.num_cards == 2 and self.total == 21 and not self.is_split

    @property
    def is_pair(self) -> bool:
        """Two cards of equal blackjack value (any two ten-value cards count as a pair)."""
        return self.num_cards == 2 and self.cards[0].value == self.cards[1].value

    def __str__(self) -> str:
        cards = " ".join(str(c) for c in self.cards)
        soft = "soft " if self.is_soft else ""
        return f"[{cards}] ({soft}{self.total})"
