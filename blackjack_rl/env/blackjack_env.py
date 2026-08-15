"""Gymnasium environment: full-rules blackjack with bet sizing and card counting.

Episode = one round of blackjack, in two phases:

  1. BET phase  -- the agent picks a bet size (actions 5 .. 5+len(bet_sizes)-1).
  2. PLAY phase -- the agent plays every hand: STAND=0, HIT=1, DOUBLE=2, SPLIT=3, SURRENDER=4.

The shoe persists across episodes (``reset`` starts a new round, not a new shoe),
so the running/true count in the observation carries real information.
Illegal actions raise ``ValueError``; always consult ``info["action_mask"]``.

Reward: 0 during the round, then the round's profit in bet units on the final step
(e.g. +1.5 for a natural on a 1-unit bet, -4 for a lost doubled 2-unit bet).
"""
from __future__ import annotations

import random
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..engine import ACTION_NAMES, Action, BlackjackGame, Rules
from .observation import OBS_DIM, encode_observation


class Phase(IntEnum):
    BET = 0
    PLAY = 1


N_PLAY_ACTIONS = len(Action)  # 5


class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        rules: Optional[Rules] = None,
        bet_sizes: Sequence[float] = (1, 2, 4, 8),
        reshuffle_each_round: bool = False,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if len(bet_sizes) == 0 or any(b <= 0 for b in bet_sizes):
            raise ValueError("bet_sizes must be a non-empty sequence of positive numbers")
        self.rules = rules or Rules()
        self.bet_sizes: Tuple[float, ...] = tuple(float(b) for b in bet_sizes)
        self.max_bet = max(self.bet_sizes)
        self.reshuffle_each_round = reshuffle_each_round
        self.render_mode = render_mode

        self._rng = random.Random(seed)
        self.game = BlackjackGame(self.rules, rng=self._rng)

        self.n_actions = N_PLAY_ACTIONS + len(self.bet_sizes)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.phase = Phase.BET
        self.current_bet = 0.0
        self.total_profit = 0.0
        self.rounds_played = 0

    # ------------------------------------------------------------------ helpers
    @property
    def bet_action_offset(self) -> int:
        return N_PLAY_ACTIONS

    def bet_action(self, index: int) -> int:
        """Action id for the ``index``-th entry of ``bet_sizes``."""
        return N_PLAY_ACTIONS + index

    def action_name(self, action: int) -> str:
        if action < N_PLAY_ACTIONS:
            return ACTION_NAMES[Action(action)]
        return f"bet {self.bet_sizes[action - N_PLAY_ACTIONS]:g}"

    def legal_actions(self) -> List[int]:
        if self.phase == Phase.BET:
            return [N_PLAY_ACTIONS + i for i in range(len(self.bet_sizes))]
        return [int(a) for a in self.game.legal_actions()]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.int8)
        mask[self.legal_actions()] = 1
        return mask

    def _obs(self) -> np.ndarray:
        shoe = self.game.shoe
        common = dict(
            true_count=shoe.true_count,
            decks_frac=shoe.cards_remaining / shoe.total_cards,
            bet_frac=self.current_bet / self.max_bet,
        )
        if self.phase == Phase.BET or not self.game.round_active:
            return encode_observation(phase=int(self.phase), **common)
        g = self.game
        h = g.current_hand
        legal = set(g.legal_actions())
        return encode_observation(
            phase=1,
            player_total=h.total,
            is_soft=h.is_soft,
            is_pair=h.is_pair,
            can_double=Action.DOUBLE in legal,
            can_split=Action.SPLIT in legal,
            can_surrender=Action.SURRENDER in legal,
            is_split_hand=h.is_split,
            dealer_upcard=g.dealer_upcard.value,
            num_hands=len(g.player_hands),
            max_splits=self.rules.max_splits,
            **common,
        )

    def _info(self, terminal: bool = False) -> Dict[str, Any]:
        g = self.game
        shoe = g.shoe
        info: Dict[str, Any] = {
            "phase": self.phase.name.lower(),
            "action_mask": self.action_mask(),
            "legal_actions": self.legal_actions(),
            "true_count": shoe.true_count,
            "running_count": shoe.running_count,
            "decks_remaining": shoe.decks_remaining,
            "bet": self.current_bet,
            "bet_sizes": self.bet_sizes,
        }
        if self.phase == Phase.PLAY and g.round_active:
            h = g.current_hand
            info.update(
                player_total=h.total,
                is_soft=h.is_soft,
                is_pair=h.is_pair,
                dealer_upcard=g.dealer_upcard.value,
                hand_index=g.current,
                num_hands=len(g.player_hands),
                is_split_hand=h.is_split,
            )
        if terminal:
            info.update(
                profit=g.round_profit,
                results=[(r.label, r.profit) for r in g.results],
                dealer_total=g.dealer_hand.total,
                dealer_bust=g.dealer_hand.is_bust,
                shuffled=g.shuffled_before_round,
                total_wagered=sum(h.bet for h in g.player_hands),
            )
        return info

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            self.game = BlackjackGame(self.rules, rng=self._rng)
        if self.game.round_active:
            self.game.abort_round()
        if self.reshuffle_each_round:
            self.game.shoe.shuffle()
        self.phase = Phase.BET
        self.current_bet = 0.0
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        if action not in self.legal_actions():
            raise ValueError(
                f"illegal action {action} ({self.action_name(action) if 0 <= action < self.n_actions else '?'}) "
                f"in phase {self.phase.name}; legal: {[self.action_name(a) for a in self.legal_actions()]}"
            )
        if self.phase == Phase.BET:
            self.current_bet = self.bet_sizes[action - N_PLAY_ACTIONS]
            over = self.game.start_round(self.current_bet)
            if not over:
                self.phase = Phase.PLAY
        else:
            self.game.step(Action(action))
            over = not self.game.round_active

        if over:
            reward = float(self.game.round_profit)
            self.total_profit += reward
            self.rounds_played += 1
            info = self._info(terminal=True)
            return self._obs(), reward, True, False, info

        return self._obs(), 0.0, False, False, self._info()

    def render(self):
        text = self.game.render()
        if self.phase == Phase.BET and not self.game.round_active:
            text = "--- new round: place your bet ---\n" + text
        if self.render_mode == "human":
            print(text)
            return None
        return text
