"""Round-level blackjack state machine.

One ``BlackjackGame`` owns a shoe that persists across rounds (so card counting
works) and plays one round at a time:

    game.start_round(bet)      -> deals; returns True if the round ended immediately (naturals)
    game.legal_actions()       -> list[Action] for the hand currently being played
    game.step(action)          -> apply the action; advances to the next hand / dealer / settlement
    game.round_active          -> False once the round has been settled
    game.results / round_profit
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from .cards import Card, Shoe
from .hand import Hand
from .rules import Rules


class Action(IntEnum):
    STAND = 0
    HIT = 1
    DOUBLE = 2
    SPLIT = 3
    SURRENDER = 4


ACTION_NAMES = {a: a.name.lower() for a in Action}
ACTION_LETTERS = {Action.STAND: "S", Action.HIT: "H", Action.DOUBLE: "D", Action.SPLIT: "P", Action.SURRENDER: "R"}


@dataclass
class HandResult:
    hand: Hand
    profit: float   # in bet units, e.g. +1.5 for a natural on a 1-unit bet
    label: str


class BlackjackGame:
    def __init__(self, rules: Optional[Rules] = None, rng: Optional[random.Random] = None, shoe: Optional[Shoe] = None):
        self.rules = rules or Rules()
        self.rng = rng or random.Random()
        self.shoe = shoe or Shoe(self.rules.num_decks, self.rules.penetration, self.rng)
        self.player_hands: List[Hand] = []
        self.dealer_hand: Hand = Hand()
        self.current: int = 0
        self.round_active: bool = False
        self.hole_card_hidden: bool = True
        self.results: List[HandResult] = []
        self.round_profit: float = 0.0
        self.shuffled_before_round: bool = False
        self.rounds_played: int = 0

    # ------------------------------------------------------------------ round lifecycle
    def start_round(self, bet: float) -> bool:
        """Deal a new round. Returns True if the round is already over (naturals / dealer blackjack)."""
        if self.round_active:
            raise RuntimeError("a round is already in progress")
        if bet <= 0:
            raise ValueError("bet must be positive")

        self.shuffled_before_round = False
        if self.shoe.needs_shuffle:
            self.shoe.shuffle()
            self.shuffled_before_round = True

        self.player_hands = [Hand(bet=float(bet))]
        self.dealer_hand = Hand()
        self.current = 0
        self.results = []
        self.round_profit = 0.0
        self.round_active = True
        self.hole_card_hidden = True

        # Casino dealing order: player, dealer up, player, dealer hole (face down).
        self.player_hands[0].add(self.shoe.draw())
        self.dealer_hand.add(self.shoe.draw())
        self.player_hands[0].add(self.shoe.draw())
        self.dealer_hand.add(self.shoe.draw(visible=False))

        if self.rules.dealer_peeks and self.dealer_hand.is_blackjack:
            self._finish_round()
            return True
        if self.player_hands[0].is_blackjack:
            self.player_hands[0].finished = True
        self._advance()
        return not self.round_active

    def abort_round(self) -> None:
        """Abandon a round in progress (used when an env is reset mid-round)."""
        if self.round_active:
            self.hole_card_hidden = False
            self.shoe.observe(self.dealer_hand.cards[1])
            self.round_active = False
            self.results = []
            self.round_profit = 0.0

    # ------------------------------------------------------------------ player decisions
    @property
    def current_hand(self) -> Hand:
        return self.player_hands[self.current]

    @property
    def dealer_upcard(self) -> Card:
        return self.dealer_hand.cards[0]

    def _can_split_more(self) -> bool:
        return len(self.player_hands) < self.rules.max_hands

    def legal_actions(self) -> List[Action]:
        if not self.round_active:
            return []
        h = self.current_hand
        r = self.rules
        if h.from_split_aces and not r.hit_split_aces:
            # One card only on split aces -- the hand is only still open if it can be re-split.
            acts = [Action.STAND]
            if h.is_pair and r.resplit_aces and self._can_split_more():
                acts.append(Action.SPLIT)
            return acts
        acts = [Action.STAND, Action.HIT]
        if h.num_cards == 2:
            if (not h.is_split or r.double_after_split) and (r.double_on is None or h.total in r.double_on):
                acts.append(Action.DOUBLE)
            if h.is_pair and self._can_split_more() and (not h.from_split_aces or r.resplit_aces):
                acts.append(Action.SPLIT)
            if r.surrender and not h.is_split:
                acts.append(Action.SURRENDER)
        return acts

    def step(self, action: Action) -> None:
        if not self.round_active:
            raise RuntimeError("no round in progress")
        action = Action(action)
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action.name} for hand {self.current_hand}")
        h = self.current_hand
        if action == Action.STAND:
            h.finished = True
        elif action == Action.HIT:
            h.add(self.shoe.draw())
            self._after_card(h)
        elif action == Action.DOUBLE:
            h.bet *= 2
            h.is_doubled = True
            h.add(self.shoe.draw())
            h.finished = True
        elif action == Action.SURRENDER:
            h.is_surrendered = True
            h.finished = True
        elif action == Action.SPLIT:
            second = h.cards.pop()
            new = Hand(cards=[second], bet=h.bet, is_split=True, from_split_aces=second.is_ace)
            h.is_split = True
            h.from_split_aces = h.cards[0].is_ace
            self.player_hands.insert(self.current + 1, new)
            h.add(self.shoe.draw())      # the new (second) hand gets its card when it becomes current
            self._after_card(h)
        self._advance()

    def _after_card(self, h: Hand) -> None:
        """Set ``finished`` after a card was dealt to a player hand."""
        if h.is_bust:
            h.finished = True
        elif h.from_split_aces and not self.rules.hit_split_aces:
            if not (h.is_pair and self.rules.resplit_aces and self._can_split_more()):
                h.finished = True
        elif h.total == 21:
            h.finished = True   # nothing sensible left to do on 21

    def _advance(self) -> None:
        """Move to the next hand that still needs a decision; finish the round if none."""
        while self.current < len(self.player_hands):
            h = self.player_hands[self.current]
            if h.num_cards == 1:  # freshly split hand waiting for its second card
                h.add(self.shoe.draw())
                self._after_card(h)
            if not h.finished:
                return
            self.current += 1
        self._finish_round()

    # ------------------------------------------------------------------ dealer & settlement
    def _finish_round(self) -> None:
        self.hole_card_hidden = False
        self.shoe.observe(self.dealer_hand.cards[1])
        needs_dealer = any(not (h.is_bust or h.is_surrendered or h.is_blackjack) for h in self.player_hands)
        if needs_dealer and not self.dealer_hand.is_blackjack:
            self._dealer_play()
        self.results = [self._settle(h) for h in self.player_hands]
        self.round_profit = sum(r.profit for r in self.results)
        self.round_active = False
        self.rounds_played += 1

    def _dealer_play(self) -> None:
        d = self.dealer_hand
        while True:
            t = d.total
            if t > 17:
                break
            if t == 17 and not (d.is_soft and self.rules.dealer_hits_soft_17):
                break
            d.add(self.shoe.draw())

    def _settle(self, h: Hand) -> HandResult:
        d = self.dealer_hand
        if h.is_surrendered:
            return HandResult(h, -h.bet / 2, "surrender")
        if h.is_bust:
            return HandResult(h, -h.bet, "bust")
        if h.is_blackjack:
            if d.is_blackjack:
                return HandResult(h, 0.0, "push (both blackjack)")
            return HandResult(h, h.bet * self.rules.blackjack_payout, "blackjack!")
        if d.is_blackjack:
            return HandResult(h, -h.bet, "dealer blackjack")
        if d.is_bust:
            return HandResult(h, h.bet, "win (dealer busts)")
        if h.total > d.total:
            return HandResult(h, h.bet, "win")
        if h.total < d.total:
            return HandResult(h, -h.bet, "lose")
        return HandResult(h, 0.0, "push")

    # ------------------------------------------------------------------ display helpers
    def render(self, show_count: bool = True) -> str:
        lines = []
        d = self.dealer_hand
        if d.cards:
            if self.hole_card_hidden:
                lines.append(f"Dealer: [{d.cards[0]} ??]")
            else:
                status = " BUST" if d.is_bust else (" BLACKJACK" if d.is_blackjack else "")
                lines.append(f"Dealer: {d}{status}")
        for i, h in enumerate(self.player_hands):
            marker = ">" if (self.round_active and i == self.current) else " "
            tags = []
            if h.is_split:
                tags.append("split")
            if h.is_doubled:
                tags.append("doubled")
            if h.is_surrendered:
                tags.append("surrendered")
            if h.is_bust:
                tags.append("BUST")
            elif h.is_blackjack:
                tags.append("BLACKJACK")
            tag = f"  ({', '.join(tags)})" if tags else ""
            label = f"Hand {i + 1}" if len(self.player_hands) > 1 else "You   "
            lines.append(f"{marker} {label}: {h}  bet {h.bet:g}{tag}")
        if show_count:
            s = self.shoe
            lines.append(f"Count: running {s.running_count:+d}, true {s.true_count:+.1f}  "
                         f"({s.decks_remaining:.1f} decks left, {s.cards_dealt}/{s.total_cards} dealt)")
        if not self.round_active and self.results:
            for i, r in enumerate(self.results):
                label = f"Hand {i + 1}: " if len(self.results) > 1 else ""
                lines.append(f"Result: {label}{r.label} ({r.profit:+g})")
        return "\n".join(lines)
