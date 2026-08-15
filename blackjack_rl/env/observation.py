"""Fixed-size observation vector shared by the env, the agents and the analysis scripts."""
from __future__ import annotations

import numpy as np

# Layout of the observation vector (all features scaled to [-1, 1]).
OBS_PHASE = 0            # 0 = betting, 1 = playing
OBS_TOTAL = 1            # player hand total / 21
OBS_SOFT = 2             # 1 if an ace counts as 11
OBS_PAIR = 3             # 1 if the hand is a splittable-looking pair (two equal-value cards)
OBS_CAN_DOUBLE = 4
OBS_CAN_SPLIT = 5
OBS_CAN_SURRENDER = 6
OBS_IS_SPLIT_HAND = 7    # hand came from a split
OBS_DEALER_START = 8     # 10 one-hot slots: A, 2, 3, ..., 10
OBS_TRUE_COUNT = 18      # true count / 10, clipped
OBS_DECKS_FRAC = 19      # fraction of the shoe still undealt
OBS_BET = 20             # current bet / max bet
OBS_NUM_HANDS = 21       # (number of hands - 1) / max_splits
OBS_DIM = 22

TRUE_COUNT_SCALE = 10.0


def encode_observation(
    *,
    phase: int,
    player_total: int = 0,
    is_soft: bool = False,
    is_pair: bool = False,
    can_double: bool = False,
    can_split: bool = False,
    can_surrender: bool = False,
    is_split_hand: bool = False,
    dealer_upcard: int = 0,          # blackjack value 1..10 (1 = Ace); 0 = unknown / bet phase
    true_count: float = 0.0,
    decks_frac: float = 1.0,
    bet_frac: float = 0.0,
    num_hands: int = 1,
    max_splits: int = 3,
) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[OBS_PHASE] = float(phase)
    if phase == 1:
        obs[OBS_TOTAL] = min(player_total, 21) / 21.0
        obs[OBS_SOFT] = float(is_soft)
        obs[OBS_PAIR] = float(is_pair)
        obs[OBS_CAN_DOUBLE] = float(can_double)
        obs[OBS_CAN_SPLIT] = float(can_split)
        obs[OBS_CAN_SURRENDER] = float(can_surrender)
        obs[OBS_IS_SPLIT_HAND] = float(is_split_hand)
        if 1 <= dealer_upcard <= 10:
            obs[OBS_DEALER_START + dealer_upcard - 1] = 1.0
        obs[OBS_NUM_HANDS] = (num_hands - 1) / max(max_splits, 1)
    obs[OBS_TRUE_COUNT] = float(np.clip(true_count / TRUE_COUNT_SCALE, -1.0, 1.0))
    obs[OBS_DECKS_FRAC] = float(np.clip(decks_frac, 0.0, 1.0))
    obs[OBS_BET] = float(np.clip(bet_frac, 0.0, 1.0))
    return obs
