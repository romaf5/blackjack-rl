import gymnasium as gym
import numpy as np
import pytest

import blackjack_rl  # noqa: F401  (registers the env)
from blackjack_rl.env import OBS_DIM, BlackjackEnv, N_PLAY_ACTIONS
from blackjack_rl.env.observation import OBS_BET, OBS_PHASE, OBS_TRUE_COUNT


def test_gym_make_and_spaces():
    env = gym.make("BlackjackFull-v0")
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.n == N_PLAY_ACTIONS + 4
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert info["phase"] == "bet"
    assert list(np.flatnonzero(info["action_mask"])) == [5, 6, 7, 8]


def test_full_episode_reward_matches_profit():
    env = BlackjackEnv(bet_sizes=(1, 5))
    obs, info = env.reset(seed=3)
    total = 0.0
    for _ in range(500):
        obs, info = env.reset()
        assert obs[OBS_PHASE] == 0
        done = False
        while not done:
            a = env.action_space.sample(mask=info["action_mask"])
            obs, r, done, trunc, info = env.step(a)
            assert env.observation_space.contains(obs)
            if not done:
                assert r == 0 and info["phase"] == "play"
                assert obs[OBS_PHASE] == 1
                assert info["action_mask"][N_PLAY_ACTIONS:].sum() == 0
        assert r == pytest.approx(info["profit"])
        assert info["bet"] in (1, 5)
        assert obs[OBS_BET] == pytest.approx(info["bet"] / 5)
        total += r
    assert env.rounds_played == 500 and env.total_profit == pytest.approx(total)


def test_illegal_actions_raise():
    env = BlackjackEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(0)  # can't stand during the bet phase
    env.step(5)
    with pytest.raises(ValueError):
        env.step(6)  # can't bet during play


def test_shoe_persists_across_episodes_unless_reshuffle_flag():
    env = BlackjackEnv()
    env.reset(seed=0)
    dealt = []
    for _ in range(5):
        env.reset()
        env.step(5)
        dealt.append(env.game.shoe.cards_dealt)
    assert dealt == sorted(dealt) and dealt[-1] > dealt[0]

    env2 = BlackjackEnv(reshuffle_each_round=True)
    env2.reset(seed=0)
    for _ in range(5):
        env2.reset()
        env2.step(5)
        assert env2.game.shoe.cards_dealt <= 12
        assert env2.game.shoe.running_count == sum(  # count restarts from the fresh shoe
            __import__("blackjack_rl").engine.HI_LO[c.value]
            for c in env2.game.player_hands[0].cards + env2.game.dealer_hand.cards[:1])


def test_seeded_reset_is_reproducible():
    env = BlackjackEnv()
    o1, i1 = env.reset(seed=42)
    env.step(5)
    r1 = env.render()
    o2, i2 = env.reset(seed=42)
    env.step(5)
    assert env.render() == r1
    assert np.array_equal(o1, o2)


def test_true_count_in_observation_is_scaled():
    env = BlackjackEnv()
    obs, info = env.reset(seed=0)
    assert obs[OBS_TRUE_COUNT] == pytest.approx(np.clip(info["true_count"] / 10, -1, 1))
