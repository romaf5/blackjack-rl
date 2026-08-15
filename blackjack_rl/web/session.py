"""A single interactive game session on top of ``BlackjackEnv`` (used by the web UI)."""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agents.basic_strategy import BasicStrategyAgent, basic_strategy, hi_lo_bet_index
from ..engine import ACTION_NAMES, Action, Card, Hand, Rules
from ..env import BlackjackEnv, Phase
from ..env.blackjack_env import N_PLAY_ACTIONS

ACTION_BY_NAME = {name: a for a, name in ACTION_NAMES.items()}


def card_json(c: Card) -> Dict[str, Any]:
    return {"rank": c.name, "suit": c.suit, "value": c.value, "red": c.suit in ("♥", "♦")}


class GameSession:
    def __init__(self, rules: Optional[Rules] = None, bet_sizes: Sequence[float] = (1, 2, 4, 8),
                 bankroll: float = 100.0, seed: Optional[int] = None, checkpoint: Optional[str] = None):
        self.lock = threading.Lock()
        self.checkpoint = checkpoint
        self.dqn = None
        self.dqn_error = None
        if checkpoint and os.path.exists(checkpoint):
            try:
                from ..agents.dqn import DQNAgent
                self.dqn = DQNAgent.load(checkpoint)
            except Exception as e:  # pragma: no cover - depends on local files
                self.dqn_error = str(e)
        self.new_game(rules, bet_sizes, bankroll, seed)

    # ------------------------------------------------------------------ lifecycle
    def new_game(self, rules: Optional[Rules], bet_sizes: Sequence[float], bankroll: float,
                 seed: Optional[int] = None) -> Dict[str, Any]:
        self.env = BlackjackEnv(rules=rules, bet_sizes=bet_sizes, seed=seed)
        self.rules = self.env.rules
        self.start_bankroll = float(bankroll)
        self.bankroll = float(bankroll)
        self.rounds = 0
        self.wins = self.losses = self.pushes = 0
        self.bankroll_history: List[float] = [self.bankroll]
        self.last_reward = 0.0
        self.last_info: Dict[str, Any] = {}
        self.round_over = False
        self.basic = BasicStrategyAgent(self.rules, count_bets=False)
        self.hilo = BasicStrategyAgent(self.rules, count_bets=True)
        if self.dqn is not None and self.dqn.n_actions != self.env.action_space.n:
            self.dqn_error = (f"checkpoint expects {self.dqn.n_actions - N_PLAY_ACTIONS} bet sizes, "
                              f"this table has {len(self.env.bet_sizes)}")
            self.dqn_usable = False
        else:
            self.dqn_usable = self.dqn is not None
            if self.dqn_usable:
                self.dqn_error = None
        self.obs, self.info = self.env.reset()
        return self.state()

    def next_round(self) -> Dict[str, Any]:
        if not self.round_over:
            return self.state()
        self.round_over = False
        self.obs, self.info = self.env.reset()
        return self.state()

    # ------------------------------------------------------------------ actions
    def _apply(self, action: int) -> Dict[str, Any]:
        self.obs, reward, done, _, self.info = self.env.step(action)
        if done:
            self.round_over = True
            self.last_reward = reward
            self.last_info = self.info
            self.bankroll += reward
            self.rounds += 1
            if reward > 0:
                self.wins += 1
            elif reward < 0:
                self.losses += 1
            else:
                self.pushes += 1
            self.bankroll_history.append(self.bankroll)
            if len(self.bankroll_history) > 500:
                self.bankroll_history = self.bankroll_history[-500:]
        return self.state()

    def bet(self, index: int) -> Dict[str, Any]:
        if self.round_over or self.env.phase != Phase.BET:
            raise ValueError("not in the betting phase")
        if not 0 <= index < len(self.env.bet_sizes):
            raise ValueError("bad bet index")
        return self._apply(N_PLAY_ACTIONS + index)

    def act(self, name: str) -> Dict[str, Any]:
        if self.round_over or self.env.phase != Phase.PLAY:
            raise ValueError("not in the playing phase")
        if name not in ACTION_BY_NAME:
            raise ValueError(f"unknown action {name}")
        action = int(ACTION_BY_NAME[name])
        if action not in self.info["legal_actions"]:
            raise ValueError(f"{name} is not legal right now")
        return self._apply(action)

    def agent_step(self, agent: str) -> Dict[str, Any]:
        """Let an agent take the next decision (bet, play, or advance to the next round)."""
        if self.round_over:
            st = self.next_round()
            st["agent_action"] = "next round"
            return st
        a = self._agent(agent)
        action = int(a.act(self.obs, self.info))
        st = self._apply(action)
        st["agent_action"] = self.env.action_name(action)
        st["agent_action_id"] = action
        return st

    def _agent(self, name: str):
        if name == "basic":
            return self.basic
        if name == "hilo":
            return self.hilo
        if name == "dqn":
            if not self.dqn_usable:
                raise ValueError(self.dqn_error or "no DQN checkpoint loaded")
            return self.dqn
        raise ValueError(f"unknown agent {name}")

    # ------------------------------------------------------------------ advice
    def advice(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"phase": self.phase_name()}
        if self.round_over:
            return out
        info = self.info
        if self.env.phase == Phase.BET:
            idx = hi_lo_bet_index(info["true_count"], self.env.bet_sizes)
            out["hilo_bet"] = self.env.bet_sizes[idx]
            out["hilo_bet_index"] = idx
        else:
            legal = [Action(x) for x in info["legal_actions"]]
            bs = basic_strategy(info["player_total"], info["is_soft"], info["is_pair"], info["dealer_upcard"],
                                legal, self.rules)
            out["basic"] = ACTION_NAMES[bs]
        if self.dqn_usable:
            q = self.dqn.q_values(self.obs) * self.env.max_bet   # back to bet units
            mask = info["action_mask"].astype(bool)
            entries = []
            for i in np.flatnonzero(mask):
                entries.append({"action": self.env.action_name(int(i)), "id": int(i), "q": float(q[i])})
            best = max(entries, key=lambda e: e["q"]) if entries else None
            out["dqn"] = {"best": best["action"] if best else None, "best_id": best["id"] if best else None,
                          "q": entries}
        return out

    # ------------------------------------------------------------------ strategy report
    def strategy_report_html(self, agent: str = "dqn", true_count: float = 0.0) -> str:
        from ..cli.strategy import build_report, render_html
        if agent not in ("basic", "hilo", "dqn"):
            raise ValueError(f"unknown agent {agent}")
        if agent == "dqn" and not self.dqn_usable:
            raise ValueError(self.dqn_error or "no DQN checkpoint loaded")
        a = self.dqn if agent == "dqn" else (self.hilo if agent == "hilo" else self.basic)
        shoe = self.env.game.shoe
        rep = build_report(agent, a, self.env, true_count, shoe.cards_remaining / shoe.total_cards,
                           self.checkpoint if agent == "dqn" else None)
        return render_html(rep)

    # ------------------------------------------------------------------ state
    def phase_name(self) -> str:
        if self.round_over:
            return "done"
        return "bet" if self.env.phase == Phase.BET else "play"

    def _hand_json(self, h: Hand, idx: int, g) -> Dict[str, Any]:
        result = None
        if self.round_over and g.results and idx < len(g.results):
            r = g.results[idx]
            result = {"label": r.label, "profit": r.profit}
        return {
            "cards": [card_json(c) for c in h.cards],
            "total": h.total,
            "soft": h.is_soft,
            "bet": h.bet,
            "is_split": h.is_split,
            "doubled": h.is_doubled,
            "surrendered": h.is_surrendered,
            "bust": h.is_bust,
            "blackjack": h.is_blackjack,
            "active": (not self.round_over) and g.round_active and idx == g.current,
            "result": result,
        }

    def state(self) -> Dict[str, Any]:
        env = self.env
        g = env.game
        shoe = g.shoe
        phase = self.phase_name()
        dealer_cards = list(g.dealer_hand.cards)
        dealer = None
        if dealer_cards and (phase != "bet"):
            hidden = g.hole_card_hidden and not self.round_over
            dealer = {
                "cards": [card_json(dealer_cards[0])] + ([{"hidden": True}] if hidden else
                                                        [card_json(c) for c in dealer_cards[1:]]),
                "total": None if hidden else g.dealer_hand.total,
                "soft": (not hidden) and g.dealer_hand.is_soft,
                "bust": (not hidden) and g.dealer_hand.is_bust,
                "blackjack": (not hidden) and g.dealer_hand.is_blackjack,
                "hidden": hidden,
                "upcard_value": dealer_cards[0].value,
            }
        hands = [] if phase == "bet" else [self._hand_json(h, i, g) for i, h in enumerate(g.player_hands)]
        legal = []
        if phase == "play":
            legal = [ACTION_NAMES[Action(a)] for a in self.info["legal_actions"]]
        state = {
            "phase": phase,
            "rules": self.rules.describe(),
            "rules_dict": self.rules.__dict__,
            "bet_sizes": list(env.bet_sizes),
            "bankroll": self.bankroll,
            "start_bankroll": self.start_bankroll,
            "rounds": self.rounds,
            "wins": self.wins, "losses": self.losses, "pushes": self.pushes,
            "bankroll_history": self.bankroll_history[-200:],
            "dealer": dealer,
            "hands": hands,
            "legal": legal,
            "current_bet": env.current_bet,
            "count": {
                "running": shoe.running_count,
                "true": round(shoe.true_count, 2),
                "decks_remaining": round(shoe.decks_remaining, 2),
                "cards_dealt": shoe.cards_dealt,
                "total_cards": shoe.total_cards,
                "cut_card": shoe.cut_card,
                "penetration": shoe.penetration,
                "num_shuffles": shoe.num_shuffles,
            },
            "agents": ["basic", "hilo"] + (["dqn"] if self.dqn_usable else []),
            "dqn": {"loaded": self.dqn_usable, "checkpoint": self.checkpoint, "error": self.dqn_error},
        }
        if self.round_over:
            state["last"] = {
                "profit": self.last_reward,
                "results": [{"label": l, "profit": p} for l, p in self.last_info.get("results", [])],
                "shuffled": bool(self.last_info.get("shuffled")),
            }
        return state
