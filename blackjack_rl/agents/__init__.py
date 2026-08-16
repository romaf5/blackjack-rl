from .base import Agent
from .basic_strategy import BasicStrategyAgent, basic_strategy, hi_lo_bet_index
from .random_agent import RandomAgent

__all__ = ["Agent", "BasicStrategyAgent", "basic_strategy", "hi_lo_bet_index", "RandomAgent", "load_rl_agent"]


def load_rl_agent(path: str, device: str = "cpu"):
    """Load a trained PPO checkpoint (``blackjack-train-ppo``)."""
    from .ppo import PPOAgent
    return PPOAgent.load(path, device=device)


def __getattr__(name):  # lazy import so torch isn't required for the engine / baselines
    if name in ("PPOAgent", "PPOConfig", "train_ppo", "ActorCritic"):
        from . import ppo
        return getattr(ppo, name)
    raise AttributeError(name)
