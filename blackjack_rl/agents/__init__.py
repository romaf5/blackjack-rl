from .base import Agent
from .basic_strategy import BasicStrategyAgent, basic_strategy, hi_lo_bet_index
from .random_agent import RandomAgent

__all__ = ["Agent", "BasicStrategyAgent", "basic_strategy", "hi_lo_bet_index", "RandomAgent"]


def __getattr__(name):  # lazy import so torch isn't required for the engine / baselines
    if name in ("DQNAgent", "TrainConfig", "train_dqn", "QNetwork", "ReplayBuffer"):
        from . import dqn
        return getattr(dqn, name)
    raise AttributeError(name)
