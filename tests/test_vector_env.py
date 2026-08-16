import numpy as np
import pytest

from blackjack_rl.env import OBS_DIM
from blackjack_rl.env.vector_env import BlackjackVectorEnv


def _random_legal(mask, rng):
    r = rng.random(mask.shape) * mask
    return r.argmax(1)


@pytest.mark.parametrize("workers", [1, 2])
def test_vector_env_shapes_and_autoreset(workers):
    rng = np.random.default_rng(0)
    venv = BlackjackVectorEnv(6, workers=workers, seed=0, bet_sizes=(1, 5))
    try:
        obs, mask = venv.reset()
        assert obs.shape == (6, OBS_DIM) and mask.shape == (6, venv.n_actions) and mask.dtype == bool
        assert (obs[:, 0] == 0).all()                       # everyone starts in the bet phase
        assert mask[:, 5:].all() and not mask[:, :5].any()  # only bet actions legal
        total_profit = 0.0
        finished = 0
        for _ in range(40):
            obs, rew, done, mask, info = venv.step(_random_legal(mask, rng))
            assert obs.shape == (6, OBS_DIM) and rew.shape == (6,) and done.shape == (6,)
            assert mask.any(1).all()                          # a legal action always exists (auto-reset)
            assert (rew[~done] == 0).all()                    # reward only on the terminal step
            assert np.allclose(rew[done], info["profit"][done])
            assert (obs[done, 0] == 0).all()                  # finished envs were reset to the bet phase
            assert (info["wagered"][done] >= 1).all()
            total_profit += rew.sum()
            finished += int(done.sum())
        assert finished > 20 and venv.rounds_played == finished
        with pytest.raises(ValueError):
            venv.step(np.zeros(3, dtype=int))
    finally:
        venv.close()


def test_vector_env_seeding_is_reproducible():
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
    a = BlackjackVectorEnv(4, seed=7)
    b = BlackjackVectorEnv(4, seed=7)
    try:
        oa, ma = a.reset(); ob, mb = b.reset()
        assert np.array_equal(oa, ob)
        for _ in range(10):
            oa, ra, da, ma, _ = a.step(_random_legal(ma, rng1))
            ob, rb, db, mb, _ = b.step(_random_legal(mb, rng2))
            assert np.array_equal(oa, ob) and np.array_equal(ra, rb)
    finally:
        a.close(); b.close()
