import gymnasium as gym

from .blackjack_env import N_PLAY_ACTIONS, BlackjackEnv, Phase
from .observation import OBS_DIM, encode_observation

ENV_ID = "BlackjackFull-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="blackjack_rl.env.blackjack_env:BlackjackEnv")

__all__ = ["BlackjackEnv", "Phase", "N_PLAY_ACTIONS", "OBS_DIM", "encode_observation", "ENV_ID"]
