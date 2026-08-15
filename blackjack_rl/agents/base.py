"""Minimal agent interface used by the evaluation loop and the scripts."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class Agent:
    name: str = "agent"

    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """Return a legal action id. ``info["action_mask"]`` marks the legal actions."""
        raise NotImplementedError

    def on_round_end(self, info: Dict[str, Any]) -> None:  # optional hook
        pass
