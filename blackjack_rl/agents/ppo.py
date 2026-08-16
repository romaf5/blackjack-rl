"""PPO with action masking on the vectorized env (PyTorch; runs on Apple MPS / CUDA / CPU).

Actor and critic are separate MLPs. Illegal actions are masked out of the policy logits
before sampling and before computing log-probs / entropy, so the agent never picks an
illegal action and never gets gradient towards one.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine import Rules
from ..env import BlackjackEnv
from ..env.vector_env import BlackjackVectorEnv
from .base import Agent

NEG = -1e8  # logit for masked (illegal) actions


def pick_device(device: Optional[str] = None) -> torch.device:
    """'auto' / None -> MPS on Apple silicon, else CUDA, else CPU."""
    if device and device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mlp(inp: int, hidden: Sequence[int], out: int, out_gain: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = inp
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = h
    head = nn.Linear(last, out)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden, n_actions, 0.01)
        self.critic = _mlp(obs_dim, hidden, 1, 1.0)

    def logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).masked_fill(~mask, NEG)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def _logp_entropy(logits: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        p = logp_all.exp()
        entropy = -(p * logp_all.clamp(min=NEG)).sum(-1)   # masked entries: p = 0
        return logp, entropy

    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        """Sample actions (Gumbel-max, works on every device). Returns action, logp, value."""
        logits = self.logits(obs, mask)
        u = torch.rand_like(logits).clamp_(1e-10, 1.0)
        action = (logits - torch.log(-torch.log(u))).argmax(dim=-1)
        logp, _ = self._logp_entropy(logits, action)
        return action, logp, self.value(obs)

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        logits = self.logits(obs, mask)
        logp, entropy = self._logp_entropy(logits, actions)
        return logp, entropy, self.value(obs)


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256),
                 device: Optional[str] = "cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.device = pick_device(device) if device else torch.device("cpu")
        self.net = ActorCritic(obs_dim, n_actions, hidden).to(self.device)
        self.extra: Dict[str, Any] = {}

    def to(self, device: str) -> "PPOAgent":
        self.device = torch.device(device)
        self.net.to(self.device)
        return self

    # ------------------------------------------------------------------ acting (single obs, numpy)
    @torch.no_grad()
    def action_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        m = torch.as_tensor(np.asarray(mask).astype(bool), device=self.device).unsqueeze(0)
        return F.softmax(self.net.logits(o, m), dim=-1)[0].cpu().numpy()

    def greedy_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        p = self.action_probs(obs, mask)
        p = np.where(np.asarray(mask).astype(bool), p, -1.0)
        return int(np.argmax(p))

    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        return self.greedy_action(obs, info["action_mask"])

    @torch.no_grad()
    def greedy_batch(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, device=self.device)
        m = torch.as_tensor(mask, device=self.device)
        return self.net.logits(o, m).argmax(dim=-1).cpu().numpy()

    # ------------------------------------------------------------------ persistence
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save({"kind": "ppo", "obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": self.hidden,
                    "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = "cpu") -> "PPOAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("kind") != "ppo":
            raise ValueError(f"{path} is not a PPO checkpoint")
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"], device=device)
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.net.eval()
        agent.extra = ckpt.get("extra", {})
        return agent


# ---------------------------------------------------------------------- training
@dataclass
class PPOConfig:
    total_rounds: int = 5_000_000
    num_envs: int = 1024
    workers: int = 8
    num_steps: int = 32              # rollout length per env (transitions per update = num_envs * num_steps)
    hidden: Tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    lr_end: float = 3e-5             # linear anneal
    gamma: float = 1.0
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 8
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.005
    ent_coef_end: float = 0.0        # linear anneal
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    reward_scale: Optional[float] = None   # default 1 / max_bet
    log_every: int = 10              # updates
    eval_every: int = 50             # updates (0 = off); greedy, batched on the vector env
    eval_rounds: int = 100_000
    seed: Optional[int] = 0
    device: Optional[str] = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def evaluate_greedy_batched(agent: PPOAgent, venv: BlackjackVectorEnv, min_rounds: int):
    """Fast greedy evaluation on a vector env. Returns (ev_per_round, ev_per_unit, mean_bet, rounds, bet_by_tc)."""
    from ..evaluation import EvalStats
    stats = EvalStats()
    obs, mask = venv.reset()
    tc_at_bet = np.floor(np.zeros(venv.num_envs))
    while stats.rounds < min_rounds:
        a = agent.greedy_batch(obs, mask)
        bet_phase = obs[:, 0] == 0
        tc_at_bet = np.where(bet_phase, np.floor(obs[:, 18] * 10.0), tc_at_bet)  # OBS_TRUE_COUNT = 18
        obs, rew, done, mask, info = venv.step(a)
        if done.any():
            p = info["profit"][done]
            stats.rounds += int(done.sum())
            stats.total_profit += float(p.sum())
            stats.sum_sq_profit += float((p ** 2).sum())
            stats.total_wagered += float(info["wagered"][done].sum())
            stats.wins += int((p > 0).sum()); stats.losses += int((p < 0).sum()); stats.pushes += int((p == 0).sum())
            for tc, b, pr in zip(tc_at_bet[done], info["bet"][done], p):
                stats.bet_by_tc[int(tc)].append(float(b))
                stats.profit_by_tc[int(tc)].append(float(pr))
    return stats


def train_ppo(cfg: PPOConfig, rules: Optional[Rules] = None, bet_sizes: Sequence[float] = (1, 2, 4, 8),
              log: Callable[[str], None] = print, agent: Optional[PPOAgent] = None,
              reshuffle_each_round: bool = False) -> Tuple[PPOAgent, List[Dict[str, float]]]:
    device = pick_device(cfg.device)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    venv = BlackjackVectorEnv(cfg.num_envs, rules=rules, bet_sizes=bet_sizes, workers=cfg.workers, seed=cfg.seed,
                              reshuffle_each_round=reshuffle_each_round)
    eval_venv = None
    obs_dim, n_actions = venv.obs_dim, venv.n_actions
    if agent is None:
        agent = PPOAgent(obs_dim, n_actions, cfg.hidden, device=str(device))
    else:
        agent.to(str(device))
    net = agent.net
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    reward_scale = cfg.reward_scale or (1.0 / venv.max_bet)

    T, N = cfg.num_steps, cfg.num_envs
    batch_size = T * N
    mb_size = batch_size // cfg.minibatches
    obs_buf = torch.zeros((T, N, obs_dim), device=device)
    mask_buf = torch.zeros((T, N, n_actions), dtype=torch.bool, device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    obs_np, mask_np = venv.reset()
    next_obs = torch.as_tensor(obs_np, device=device)
    next_mask = torch.as_tensor(mask_np, device=device)

    history: List[Dict[str, float]] = []
    rounds_done = 0
    profit_sum = 0.0
    wagered_sum = 0.0
    window_rounds = 0
    window_profit = 0.0
    update = 0
    t0 = time.time()
    t_env = t_upd = 0.0
    log(f"PPO on {device} | envs {N} x steps {T} = {batch_size:,} transitions/update, minibatch {mb_size:,} | "
        f"workers {venv.workers}")

    while rounds_done < cfg.total_rounds:
        update += 1
        progress = min(1.0, rounds_done / cfg.total_rounds)
        lr_now = cfg.lr + (cfg.lr_end - cfg.lr) * progress
        ent_now = cfg.ent_coef + (cfg.ent_coef_end - cfg.ent_coef) * progress
        for g in opt.param_groups:
            g["lr"] = lr_now

        # ------------------------------------------------ rollout
        te = time.time()
        with torch.no_grad():
            for step in range(T):
                obs_buf[step] = next_obs
                mask_buf[step] = next_mask
                action, logp, value = net.act(next_obs, next_mask)
                act_buf[step] = action
                logp_buf[step] = logp
                val_buf[step] = value
                obs_np, rew_np, done_np, mask_np, info = venv.step(action.cpu().numpy())
                rew_buf[step] = torch.as_tensor(rew_np * reward_scale, device=device)
                done_buf[step] = torch.as_tensor(done_np.astype(np.float32), device=device)
                next_obs = torch.as_tensor(obs_np, device=device)
                next_mask = torch.as_tensor(mask_np, device=device)
                nd = int(done_np.sum())
                rounds_done += nd
                window_rounds += nd
                ps = float(info["profit"].sum())
                profit_sum += ps
                window_profit += ps
                wagered_sum += float(info["wagered"].sum())
            next_value = net.value(next_obs)
            # GAE (done_buf[t] == 1 means the episode ended after step t; obs[t+1] is a fresh episode)
            adv = torch.zeros_like(rew_buf)
            lastgaelam = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nextvalues = next_value if t == T - 1 else val_buf[t + 1]
                nonterminal = 1.0 - done_buf[t]
                delta = rew_buf[t] + cfg.gamma * nextvalues * nonterminal - val_buf[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nonterminal * lastgaelam
                adv[t] = lastgaelam
            returns = adv + val_buf
        t_env += time.time() - te

        # ------------------------------------------------ update
        tu = time.time()
        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_mask = mask_buf.reshape(batch_size, n_actions)
        b_act = act_buf.reshape(batch_size)
        b_logp = logp_buf.reshape(batch_size)
        b_adv = adv.reshape(batch_size)
        b_ret = returns.reshape(batch_size)
        b_val = val_buf.reshape(batch_size)
        pg_losses, v_losses, ents, kls, clipfracs = [], [], [], [], []
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                idx = perm[start:start + mb_size]
                newlogp, entropy, newvalue = net.evaluate(b_obs[idx], b_mask[idx], b_act[idx])
                logratio = newlogp - b_logp[idx]
                ratio = logratio.exp()
                mb_adv = b_adv[idx]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                if cfg.clip_vloss:
                    v_unclipped = (newvalue - b_ret[idx]) ** 2
                    v_clipped = (b_val[idx] + torch.clamp(newvalue - b_val[idx], -cfg.clip_coef, cfg.clip_coef) - b_ret[idx]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_ret[idx]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - ent_now * ent + cfg.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - logratio).mean())
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean())
                pg_losses.append(pg_loss.detach()); v_losses.append(v_loss.detach()); ents.append(ent.detach())
        t_upd += time.time() - tu

        # ------------------------------------------------ logging / eval
        if update % cfg.log_every == 0 or rounds_done >= cfg.total_rounds:
            elapsed = time.time() - t0
            entry = {
                "update": update, "rounds": rounds_done, "elapsed_s": elapsed,
                "rounds_per_s": rounds_done / max(elapsed, 1e-9),
                "train_ev_per_round": window_profit / max(window_rounds, 1),
                "pg_loss": float(torch.stack(pg_losses).mean()), "v_loss": float(torch.stack(v_losses).mean()),
                "entropy": float(torch.stack(ents).mean()), "approx_kl": float(torch.stack(kls).mean()),
                "clipfrac": float(torch.stack(clipfracs).mean()), "lr": lr_now, "ent_coef": ent_now,
                "t_env_frac": t_env / max(t_env + t_upd, 1e-9),
            }
            msg = (f"upd {update:>5} | rounds {rounds_done:>11,} | {entry['rounds_per_s']:>7,.0f} r/s | "
                   f"train EV/round {entry['train_ev_per_round']:+.4f} | pg {entry['pg_loss']:+.4f} v {entry['v_loss']:.4f} "
                   f"ent {entry['entropy']:.3f} kl {entry['approx_kl']:.4f} | env {100 * entry['t_env_frac']:.0f}% | {elapsed:.0f}s")
            window_rounds = 0
            window_profit = 0.0
            if cfg.eval_every and (update % cfg.eval_every == 0 or rounds_done >= cfg.total_rounds):
                if eval_venv is None:
                    eval_venv = BlackjackVectorEnv(min(N, 512), rules=rules, bet_sizes=bet_sizes, workers=1,
                                                   seed=None if cfg.seed is None else cfg.seed + 10_000,
                                                   reshuffle_each_round=reshuffle_each_round)
                net.eval()
                st = evaluate_greedy_batched(agent, eval_venv, cfg.eval_rounds)
                net.train()
                entry.update(eval_ev_per_round=st.ev_per_round, eval_ev_per_unit=st.ev_per_unit_wagered,
                             eval_mean_bet=st.mean_bet, eval_rounds=st.rounds)
                msg += (f"\n      greedy eval ({st.rounds:,} rounds): {st.ev_per_round:+.4f}/round "
                        f"({100 * st.ev_per_unit_wagered:+.2f}% of wager, mean bet {st.mean_bet:.2f}, ±{1.96 * st.stderr:.4f})")
            history.append(entry)
            log(msg)

    venv.close()
    if eval_venv is not None:
        eval_venv.close()
    net.eval()
    return agent, history
