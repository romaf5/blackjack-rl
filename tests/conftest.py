from __future__ import annotations

from typing import List

import pytest

from blackjack_rl.engine import Card, Shoe


class ScriptedShoe(Shoe):
    """A shoe that deals a predetermined sequence of ranks (1 = Ace, 10 = ten/face)."""

    def __init__(self, ranks: List[int], num_decks: int = 6):
        self.script = [Card(r) for r in ranks]
        super().__init__(num_decks=num_decks, penetration=0.75)

    def draw(self, visible: bool = True) -> Card:
        if self.script:
            card = self.script.pop(0)
            self._pos += 1
            if visible:
                self.observe(card)
            return card
        return super().draw(visible)


@pytest.fixture
def scripted():
    return ScriptedShoe
