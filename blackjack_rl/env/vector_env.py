"""Batched ("vectorized") blackjack environments with auto-reset.

``BlackjackVectorEnv`` steps N independent ``BlackjackEnv`` instances (each with its own
shoe, so their counts are independent) and returns stacked arrays, which lets a policy
run one batched forward pass per step. With ``workers > 1`` the envs are sharded across
subprocesses so the pure-Python game logic runs on several cores.

    venv = BlackjackVectorEnv(1024, workers=8, seed=0)
    obs, mask = venv.reset()                       # (N, obs_dim) float32, (N, n_actions) bool
    obs, rew, done, mask, info = venv.step(actions) # actions: (N,) ints; finished envs auto-reset
    venv.close()

``info`` is a dict of arrays: ``profit`` (round profit for envs that finished this step, else 0),
``wagered`` (total wager of finished rounds), ``true_count`` (before the step) and ``bet``.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..engine import Rules
from .blackjack_env import BlackjackEnv, N_PLAY_ACTIONS
from .observation import OBS_DIM


class _EnvShard:
    """A list of envs living in one process."""

    def __init__(self, num_envs: int, rules: Optional[Rules], bet_sizes: Sequence[float],
                 reshuffle_each_round: bool, seed: Optional[int]):
        self.envs: List[BlackjackEnv] = [
            BlackjackEnv(rules=rules, bet_sizes=bet_sizes, reshuffle_each_round=reshuffle_each_round,
                         seed=None if seed is None else seed + i)
            for i in range(num_envs)
        ]
        self.n_actions = self.envs[0].action_space.n

    def reset(self):
        obs = np.zeros((len(self.envs), OBS_DIM), dtype=np.float32)
        mask = np.zeros((len(self.envs), self.n_actions), dtype=bool)
        for i, e in enumerate(self.envs):
            o, info = e.reset()
            obs[i] = o
            mask[i] = info["action_mask"].astype(bool)
        return obs, mask

    def step(self, actions: np.ndarray):
        n = len(self.envs)
        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        mask = np.zeros((n, self.n_actions), dtype=bool)
        rew = np.zeros(n, dtype=np.float32)
        done = np.zeros(n, dtype=bool)
        profit = np.zeros(n, dtype=np.float32)
        wagered = np.zeros(n, dtype=np.float32)
        tc = np.zeros(n, dtype=np.float32)
        bet = np.zeros(n, dtype=np.float32)
        for i, e in enumerate(self.envs):
            tc[i] = e.game.shoe.true_count
            o, r, d, _, info = e.step(int(actions[i]))
            rew[i] = r
            bet[i] = info["bet"]
            if d:
                done[i] = True
                profit[i] = info["profit"]
                wagered[i] = info["total_wagered"]
                o, info = e.reset()
            obs[i] = o
            mask[i] = info["action_mask"].astype(bool)
        return obs, rew, done, mask, {"profit": profit, "wagered": wagered, "true_count": tc, "bet": bet}


def _worker(conn, num_envs, rules, bet_sizes, reshuffle_each_round, seed):  # pragma: no cover - runs in subprocess
    shard = _EnvShard(num_envs, rules, bet_sizes, reshuffle_each_round, seed)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(shard.reset())
            elif cmd == "step":
                conn.send(shard.step(payload))
            elif cmd == "close":
                break
    finally:
        conn.close()


class BlackjackVectorEnv:
    def __init__(self, num_envs: int, rules: Optional[Rules] = None, bet_sizes: Sequence[float] = (1, 2, 4, 8),
                 reshuffle_each_round: bool = False, seed: Optional[int] = None, workers: int = 1):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        workers = max(1, min(int(workers), num_envs))
        self.num_envs = num_envs
        self.workers = workers
        self.rules = rules or Rules()
        self.bet_sizes = tuple(float(b) for b in bet_sizes)
        self.max_bet = max(self.bet_sizes)
        self.n_actions = N_PLAY_ACTIONS + len(self.bet_sizes)
        self.obs_dim = OBS_DIM
        self.rounds_played = 0
        self._closed = False

        # split envs into (almost) equal shards
        sizes = [num_envs // workers + (1 if i < num_envs % workers else 0) for i in range(workers)]
        self._slices = []
        start = 0
        for s in sizes:
            self._slices.append(slice(start, start + s))
            start += s

        if workers == 1:
            self._shard = _EnvShard(num_envs, self.rules, self.bet_sizes, reshuffle_each_round, seed)
            self._conns = None
        else:
            ctx = mp.get_context("spawn")
            self._conns = []
            self._procs = []
            for i, s in enumerate(sizes):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker,
                                args=(child, s, self.rules, self.bet_sizes, reshuffle_each_round,
                                      None if seed is None else seed + self._slices[i].start),
                                daemon=True)
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)

    # ------------------------------------------------------------------ api
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._conns is None:
            return self._shard.reset()
        for c in self._conns:
            c.send(("reset", None))
        parts = [c.recv() for c in self._conns]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).reshape(-1)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {actions.shape[0]}")
        if self._conns is None:
            out = self._shard.step(actions)
        else:
            for c, s in zip(self._conns, self._slices):
                c.send(("step", actions[s]))
            parts = [c.recv() for c in self._conns]
            obs = np.concatenate([p[0] for p in parts])
            rew = np.concatenate([p[1] for p in parts])
            done = np.concatenate([p[2] for p in parts])
            mask = np.concatenate([p[3] for p in parts])
            info = {k: np.concatenate([p[4][k] for p in parts]) for k in parts[0][4]}
            out = (obs, rew, done, mask, info)
        self.rounds_played += int(out[2].sum())
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._conns is not None:
            for c in self._conns:
                try:
                    c.send(("close", None))
                    c.close()
                except (BrokenPipeError, OSError):
                    pass
            for p in self._procs:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
