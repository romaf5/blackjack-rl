from blackjack_rl.engine import Card, Hand


def H(*ranks, **kw):
    return Hand(cards=[Card(r) for r in ranks], **kw)


def test_hard_and_soft_totals():
    assert H(10, 7).total == 17 and not H(10, 7).is_soft
    assert H(1, 6).total == 17 and H(1, 6).is_soft
    assert H(1, 6, 10).total == 17 and not H(1, 6, 10).is_soft   # ace drops to 1
    assert H(1, 1).total == 12 and H(1, 1).is_soft
    assert H(1, 1, 9).total == 21 and H(1, 1, 9).is_soft
    assert H(13, 12).total == 20  # K, Q


def test_bust_blackjack_pair():
    assert H(10, 6, 9).is_bust
    assert H(1, 13).is_blackjack
    assert not H(1, 10, is_split=True).is_blackjack   # 21 after a split is not a natural
    assert not H(7, 7, 7).is_blackjack
    assert H(10, 13).is_pair        # T + K count as a pair
    assert H(1, 1).is_pair
    assert not H(1, 1, 1).is_pair
