from blackjack_rl.agents import BasicStrategyAgent, basic_strategy, hi_lo_bet_index
from blackjack_rl.engine import Action, Rules
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.evaluation import evaluate

S, H, D, P, R = Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT, Action.SURRENDER
ALL = [S, H, D, P, R]
NO_DOUBLE = [S, H]


S17 = Rules(dealer_hits_soft_17=False)


def test_default_rules_are_h17():
    assert Rules().dealer_hits_soft_17 is True


def test_h17_specific_cells():
    r = Rules()  # H17
    assert basic_strategy(11, False, False, 1, ALL, r) == D              # 11 vs A: double under H17
    assert basic_strategy(19, True, False, 6, ALL, r) == D               # A,8 vs 6: double under H17
    assert basic_strategy(18, True, False, 2, ALL, r) == D               # A,7 vs 2: double under H17
    assert basic_strategy(15, False, False, 1, ALL, r) == R              # 15 vs A: surrender under H17
    assert basic_strategy(17, False, False, 1, ALL, r) == R              # 17 vs A: surrender under H17
    assert basic_strategy(17, False, False, 1, NO_DOUBLE, r) == S        # ... else stand
    assert basic_strategy(16, False, True, 1, ALL, r) == R               # 8,8 vs A: surrender under H17
    assert basic_strategy(16, False, True, 1, [S, H, D, P], r) == P      # ... else split


def test_known_cells_s17():
    r = S17
    assert basic_strategy(16, False, False, 10, ALL, r) == R
    assert basic_strategy(16, False, False, 10, NO_DOUBLE, r) == H       # can't surrender -> hit
    assert basic_strategy(16, False, False, 6, ALL, r) == S
    assert basic_strategy(11, False, False, 1, ALL, r) == H              # S17: 11 vs A hit
    assert basic_strategy(11, False, False, 1, ALL, Rules(dealer_hits_soft_17=True)) == D
    assert basic_strategy(18, True, False, 3, ALL, r) == D               # A,7 vs 3 double
    assert basic_strategy(18, True, False, 3, NO_DOUBLE, r) == S         # ... else stand
    assert basic_strategy(18, True, False, 9, ALL, r) == H
    assert basic_strategy(16, False, True, 5, ALL, r) == P               # 8,8 split
    assert basic_strategy(20, False, True, 6, ALL, r) == S               # T,T never split
    assert basic_strategy(12, True, True, 10, ALL, r) == P               # A,A split
    assert basic_strategy(12, True, True, 10, NO_DOUBLE, r) == H         # A,A can't split -> hit soft 12
    assert basic_strategy(10, False, True, 9, ALL, r) == D               # 5,5 double
    assert basic_strategy(8, False, True, 5, ALL, r) == P                # 4,4 vs 5 split (DAS)
    assert basic_strategy(8, False, True, 5, ALL, Rules(dealer_hits_soft_17=False, double_after_split=False)) == H
    assert basic_strategy(12, False, False, 2, ALL, r) == H
    assert basic_strategy(12, False, False, 4, ALL, r) == S


def test_hi_lo_bet_index():
    sizes = (1, 2, 4, 8)
    assert [hi_lo_bet_index(tc, sizes) for tc in (-3, 0, 1, 1.9, 2, 3, 4, 7)] == [0, 0, 0, 0, 1, 2, 3, 3]
    assert hi_lo_bet_index(5, (1, 3)) == 1
    assert hi_lo_bet_index(5, (1,)) == 0
    assert hi_lo_bet_index(3, (2, 5, 10, 25)) == 1   # 4 units of 2 = 8 -> largest bet <= 8 is 5


def test_basic_strategy_beats_random_and_edge_is_small():
    env = BlackjackEnv(bet_sizes=(1,))
    stats = evaluate(BasicStrategyAgent(env.rules), env, 30_000, seed=5)
    # true value is about -0.6% of the initial bet (H17); allow generous Monte-Carlo slack
    assert -0.025 < stats.ev_per_round < 0.01
    from blackjack_rl.agents import RandomAgent
    rnd = evaluate(RandomAgent(seed=1), env, 5_000, seed=5)
    assert rnd.ev_per_round < -0.2
