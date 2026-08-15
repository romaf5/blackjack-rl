"""Double DQN with action masking (PyTorch).

The same network handles both phases of a round: during the bet phase only the
bet actions are unmasked, during the play phase only the legal play actions are.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env import BlackjackEnv
from .base import Agent


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, obs, action, reward, next_obs, next_mask, done) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_mask[i] = next_mask
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.next_mask[idx], self.dones[idx])


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256),
                 lr: float = 5e-4, gamma: float = 1.0, device: Optional[str] = None,
                 seed: Optional[int] = None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.gamma = gamma
        self.device = torch.device(device or "cpu")
        if seed is not None:
            torch.manual_seed(seed)
        self.q = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = copy.deepcopy(self.q).eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.epsilon = 0.0
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        q = self.q(x).cpu().numpy()
        return q[0] if single else q

    def greedy_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        q = self.q_values(obs)
        q = np.where(mask.astype(bool), q, -np.inf)
        return int(np.argmax(q))

    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        mask = info["action_mask"]
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return self.greedy_action(obs, mask)

    # ------------------------------------------------------------------ learning
    def update(self, batch) -> float:
        obs, actions, rewards, next_obs, next_mask, dones = batch
        obs_t = torch.as_tensor(obs, device=self.device)
        act_t = torch.as_tensor(actions, device=self.device)
        rew_t = torch.as_tensor(rewards, device=self.device)
        nobs_t = torch.as_tensor(next_obs, device=self.device)
        nmask_t = torch.as_tensor(next_mask, device=self.device)
        done_t = torch.as_tensor(dones, device=self.device)

        q_sa = self.q(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: pick a* with the online net (masked), evaluate with the target net.
            q_next_online = self.q(nobs_t).masked_fill(~nmask_t, -1e9)
            a_star = q_next_online.argmax(dim=1, keepdim=True)
            q_next = self.target(nobs_t).gather(1, a_star).squeeze(1)
            # Terminal transitions have an empty mask; (1 - done) removes them anyway.
            target = rew_t + self.gamma * (1.0 - done_t) * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.q.state_dict())

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    # ------------------------------------------------------------------ persistence
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save({
            "obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": self.hidden,
            "gamma": self.gamma, "state_dict": self.q.state_dict(), "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DQNAgent":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"], gamma=ckpt["gamma"], device=device)
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.sync_target()
        agent.q.eval()
        agent.extra = ckpt.get("extra", {})
        return agent


# ---------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    rounds: int = 200_000
    hidden: Tuple[int, ...] = (256, 256)
    lr: float = 5e-4
    lr_end: float = 5e-5            # linear decay target (reached at the end of training)
    gamma: float = 1.0              # episodes are a single round; no discounting needed
    buffer_size: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 5_000
    train_every: int = 2            # gradient step every N env steps
    target_sync_every: int = 2_000  # env steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5     # fraction of training rounds over which epsilon decays
    reward_scale: Optional[float] = None  # default: 1 / max_bet
    log_every: int = 10_000         # rounds
    eval_every: int = 50_000        # rounds (0 = no intermediate greedy evaluations)
    eval_rounds: int = 20_000
    seed: Optional[int] = 0
    device: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def train_dqn(env: BlackjackEnv, cfg: TrainConfig, log: Callable[[str], None] = print,
              agent: Optional[DQNAgent] = None) -> Tuple[DQNAgent, List[Dict[str, float]]]:
    """Train (or continue training) a masked Double-DQN agent on ``env``."""
    from ..evaluation import evaluate  # local import to avoid a cycle

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    if agent is None:
        agent = DQNAgent(obs_dim, n_actions, cfg.hidden, lr=cfg.lr, gamma=cfg.gamma,
                         device=cfg.device, seed=cfg.seed)
    reward_scale = cfg.reward_scale or (1.0 / env.max_bet)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, n_actions)
    rng = np.random.default_rng(cfg.seed)
    eps_decay_rounds = max(1, int(cfg.rounds * cfg.eps_decay_frac))

    history: List[Dict[str, float]] = []
    recent_rewards: List[float] = []
    recent_losses: List[float] = []
    step = 0
    t0 = time.time()

    obs, info = env.reset(seed=cfg.seed)
    for rnd in range(1, cfg.rounds + 1):
        if rnd > 1:
            obs, info = env.reset()
        frac = min(1.0, (rnd - 1) / eps_decay_rounds)
        agent.epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac
        agent.set_lr(cfg.lr + (cfg.lr_end - cfg.lr) * (rnd - 1) / max(1, cfg.rounds - 1))

        done = False
        while not done:
            mask = info["action_mask"]
            action = agent.act(obs, info)
            next_obs, reward, done, _, info = env.step(action)
            buffer.add(obs, action, reward * reward_scale, next_obs, info["action_mask"].astype(bool), done)
            obs = next_obs
            step += 1
            if buffer.size >= cfg.warmup_steps and step % cfg.train_every == 0:
                recent_losses.append(agent.update(buffer.sample(cfg.batch_size, rng)))
            if step % cfg.target_sync_every == 0:
                agent.sync_target()
        recent_rewards.append(reward)

        if rnd % cfg.log_every == 0:
            entry = {
                "round": rnd, "steps": step, "epsilon": agent.epsilon,
                "avg_reward": float(np.mean(recent_rewards)),
                "loss": float(np.mean(recent_losses)) if recent_losses else float("nan"),
                "elapsed_s": time.time() - t0,
            }
            msg = (f"round {rnd:>8,} | steps {step:>9,} | eps {agent.epsilon:.3f} | "
                   f"train EV/round {entry['avg_reward']:+.4f} | loss {entry['loss']:.4f} | "
                   f"{entry['elapsed_s']:.0f}s")
            if cfg.eval_every and rnd % cfg.eval_every == 0:
                agent.epsilon = 0.0
                agent.q.eval()
                stats = evaluate(agent, env, cfg.eval_rounds, seed=10_000 + rnd)
                agent.q.train()
                entry["eval_ev_per_round"] = stats.ev_per_round
                entry["eval_ev_per_unit"] = stats.ev_per_unit_wagered
                entry["eval_mean_bet"] = stats.mean_bet
                msg += (f" | greedy eval: {stats.ev_per_round:+.4f}/round "
                        f"({100 * stats.ev_per_unit_wagered:+.2f}% of wager, mean bet {stats.mean_bet:.2f})")
                obs, info = env.reset()  # the eval consumed the env's round state; start fresh
            history.append(entry)
            log(msg)
            recent_rewards.clear()
            recent_losses.clear()

    agent.epsilon = 0.0
    agent.q.eval()
    return agent, history
