from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    """Uniformly random legal actions (bets and plays)."""

    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        legal = np.flatnonzero(info["action_mask"])
        return int(self.rng.choice(legal))
