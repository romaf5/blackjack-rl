"""A single interactive game session on top of ``BlackjackEnv`` (used by the web UI)."""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..agents.basic_strategy import BasicStrategyAgent, basic_strategy, hi_lo_bet_index
from ..engine import ACTION_NAMES, Action, Card, Hand, Rules
from ..env import BlackjackEnv, Phase
from ..env.blackjack_env import N_PLAY_ACTIONS

ACTION_BY_NAME = {name: a for a, name in ACTION_NAMES.items()}
RL_NAMES = ("rl", "ppo")


def card_json(c: Card) -> Dict[str, Any]:
    return {"rank": c.name, "suit": c.suit, "value": c.value, "red": c.suit in ("♥", "♦")}


class RLSlot:
    """One loaded PPO checkpoint as offered to the UI (several can be loaded to compare runs)."""

    def __init__(self, key: str, checkpoint: str, agent=None, error: Optional[str] = None):
        self.key = key
        self.checkpoint = checkpoint
        self.agent = agent
        self.error = error
        self.usable = agent is not None and error is None

    @property
    def kind(self) -> Optional[str]:
        return getattr(self.agent, "name", None) if self.agent is not None else None

    def to_json(self) -> Dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "checkpoint": self.checkpoint, "usable": self.usable, "error": self.error}


class GameSession:
    def __init__(self, rules: Optional[Rules] = None, bet_sizes: Sequence[float] = (1, 2, 4, 8),
                 bankroll: float = 100.0, seed: Optional[int] = None,
                 checkpoint: Optional[Union[str, Sequence[str]]] = None):
        self.lock = threading.Lock()
        paths = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint or [])
        self.checkpoint = paths[0] if paths else None          # kept for backwards compatibility
        self.rl_slots: List[RLSlot] = []
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            try:
                from ..agents import load_rl_agent
                agent = load_rl_agent(path)
                key = agent.name
                if any(sl.key == key for sl in self.rl_slots):
                    key = f"{key}{sum(1 for sl in self.rl_slots if sl.kind == agent.name) + 1}"
                self.rl_slots.append(RLSlot(key, path, agent))
            except Exception as e:  # pragma: no cover - depends on local files
                self.rl_slots.append(RLSlot(os.path.basename(path), path, None, str(e)))
        self.new_game(rules, bet_sizes, bankroll, seed)

    # ---- helpers over the RL slots
    @property
    def usable_slots(self) -> List["RLSlot"]:
        return [sl for sl in self.rl_slots if sl.usable]

    def slot(self, name: str) -> "RLSlot":
        """Resolve 'rl' / 'ppo' (first usable) or a slot key ('ppo2', ...) to a usable slot, or raise."""
        cands = self.usable_slots
        if name == "rl":
            if cands:
                return cands[0]
        else:
            for sl in cands:
                if sl.key == name or sl.kind == name:
                    return sl
        errors = [sl.error for sl in self.rl_slots if sl.error]
        raise ValueError(errors[0] if errors else "no RL checkpoint loaded")

    # backwards-compatible views used by older code paths / tests
    @property
    def rl(self):
        return self.usable_slots[0].agent if self.usable_slots else None

    @property
    def rl_usable(self) -> bool:
        return bool(self.usable_slots)

    @property
    def rl_error(self) -> Optional[str]:
        errs = [sl.error for sl in self.rl_slots if sl.error]
        return errs[0] if errs else None

    @property
    def rl_kind(self) -> Optional[str]:
        return self.usable_slots[0].kind if self.usable_slots else None

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
        for sl in self.rl_slots:
            if sl.agent is None:
                continue
            if sl.agent.n_actions != self.env.action_space.n:
                sl.error = (f"checkpoint expects {sl.agent.n_actions - N_PLAY_ACTIONS} bet sizes, "
                            f"this table has {len(self.env.bet_sizes)}")
                sl.usable = False
            else:
                sl.error = None
                sl.usable = True
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
        return self.slot(name).agent

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
        mask = info["action_mask"]
        rl: Dict[str, Any] = {}
        for sl in self.usable_slots:
            probs = sl.agent.action_probs(self.obs, mask)
            entries = [{"action": self.env.action_name(int(i)), "id": int(i), "prob": float(probs[i])}
                       for i in np.flatnonzero(mask)]
            best = max(entries, key=lambda e: e["prob"]) if entries else None
            rl[sl.key] = {"agent": sl.kind, "best": best["action"] if best else None,
                          "best_id": best["id"] if best else None, "probs": entries}
        out["rl_agents"] = rl
        return out

    # ------------------------------------------------------------------ strategy report
    def strategy_report_html(self, agent: str = "rl", true_count: float = 0.0) -> str:
        from ..cli.strategy import build_report, render_html
        shoe = self.env.game.shoe
        decks_frac = shoe.cards_remaining / shoe.total_cards
        if agent in ("basic", "hilo"):
            a = self.hilo if agent == "hilo" else self.basic
            rep = build_report(agent, a, self.env, true_count, decks_frac, None)
        else:
            sl = self.slot(agent)
            rep = build_report("rl", sl.agent, self.env, true_count, decks_frac, sl.checkpoint)
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
            "agents": ["basic", "hilo"] + [sl.key for sl in self.usable_slots],
            "rl_agents": [sl.to_json() for sl in self.rl_slots],
        }
        if self.round_over:
            state["last"] = {
                "profit": self.last_reward,
                "results": [{"label": l, "profit": p} for l, p in self.last_info.get("results", [])],
                "shuffled": bool(self.last_info.get("shuffled")),
            }
        return state
