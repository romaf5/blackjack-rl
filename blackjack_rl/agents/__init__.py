from .base import Agent
from .basic_strategy import BasicStrategyAgent, basic_strategy, hi_lo_bet_index
from .random_agent import RandomAgent

__all__ = ["Agent", "BasicStrategyAgent", "basic_strategy", "hi_lo_bet_index", "RandomAgent", "load_rl_agent"]


def load_rl_agent(path: str, device: str = "cpu"):
    """Load a DQN or PPO checkpoint (dispatches on the ``kind`` stored in the file)."""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") == "ppo":
        from .ppo import PPOAgent
        return PPOAgent.load(path, device=device)
    from .dqn import DQNAgent
    return DQNAgent.load(path, device=device)


def __getattr__(name):  # lazy import so torch isn't required for the engine / baselines
    if name in ("DQNAgent", "TrainConfig", "train_dqn", "QNetwork", "ReplayBuffer"):
        from . import dqn
        return getattr(dqn, name)
    if name in ("PPOAgent", "PPOConfig", "train_ppo", "ActorCritic"):
        from . import ppo
        return getattr(ppo, name)
    raise AttributeError(name)
