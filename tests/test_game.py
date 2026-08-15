import pytest

from blackjack_rl.engine import Action, BlackjackGame, Rules

# Deal order: player, dealer up, player, dealer hole, then draws in order.


def play(game, *actions):
    for a in actions:
        game.step(a)


def test_player_blackjack_pays_3_to_2(scripted):
    g = BlackjackGame(shoe=scripted([1, 5, 10, 9]))
    assert g.start_round(2) is True
    assert g.round_profit == pytest.approx(3.0)
    assert g.results[0].label == "blackjack!"


def test_dealer_blackjack_with_peek_ends_round(scripted):
    g = BlackjackGame(shoe=scripted([10, 1, 7, 10]))
    assert g.start_round(1) is True
    assert g.round_profit == -1
    assert not g.round_active


def test_both_blackjack_push(scripted):
    g = BlackjackGame(shoe=scripted([1, 10, 10, 1]))
    assert g.start_round(1) is True
    assert g.round_profit == 0


def test_no_peek_player_loses_doubled_bet(scripted):
    g = BlackjackGame(rules=Rules(dealer_peeks=False), shoe=scripted([6, 1, 5, 10, 9]))
    assert g.start_round(1) is False
    play(g, Action.DOUBLE)  # 6+5 -> double, draw 9 -> 20 vs dealer BJ
    assert g.round_profit == -2


def test_dealer_stands_soft_17_by_default(scripted):
    g = BlackjackGame(shoe=scripted([10, 1, 8, 6]))  # dealer A,6 = soft 17
    g.start_round(1)
    play(g, Action.STAND)
    assert g.dealer_hand.total == 17 and len(g.dealer_hand.cards) == 2
    assert g.round_profit == 1  # 18 beats 17


def test_dealer_hits_soft_17_with_h17(scripted):
    g = BlackjackGame(rules=Rules(dealer_hits_soft_17=True), shoe=scripted([10, 1, 8, 6, 3]))
    g.start_round(1)
    play(g, Action.STAND)
    assert g.dealer_hand.total == 20 and len(g.dealer_hand.cards) == 3
    assert g.round_profit == -1


def test_dealer_hits_16_and_busts(scripted):
    g = BlackjackGame(shoe=scripted([10, 10, 8, 6, 9]))
    g.start_round(1)
    play(g, Action.STAND)
    assert g.dealer_hand.is_bust
    assert g.results[0].label == "win (dealer busts)"
    assert g.round_profit == 1


def test_hit_and_bust(scripted):
    g = BlackjackGame(shoe=scripted([10, 7, 6, 10, 9]))
    g.start_round(1)
    play(g, Action.HIT)
    assert g.player_hands[0].is_bust and not g.round_active
    assert g.round_profit == -1
    assert len(g.dealer_hand.cards) == 2  # dealer does not draw when everyone busted


def test_auto_stand_on_21(scripted):
    g = BlackjackGame(shoe=scripted([10, 7, 5, 10, 6]))
    g.start_round(1)
    play(g, Action.HIT)  # 15 + 6 = 21 -> auto finished, dealer plays
    assert not g.round_active and g.round_profit == 1


def test_double_takes_one_card_and_doubles_bet(scripted):
    g = BlackjackGame(shoe=scripted([6, 6, 5, 10, 10, 9]))  # player 6,5 vs dealer 6 (hole T); double draws T; dealer draws 9
    g.start_round(1)
    assert Action.DOUBLE in g.legal_actions()
    play(g, Action.DOUBLE)
    h = g.player_hands[0]
    assert h.bet == 2 and h.is_doubled and len(h.cards) == 3 and h.total == 21
    assert not g.round_active
    assert g.dealer_hand.is_bust
    assert g.round_profit == 2


def test_no_double_after_three_cards(scripted):
    g = BlackjackGame(shoe=scripted([2, 6, 3, 10, 4]))
    g.start_round(1)
    play(g, Action.HIT)  # 5 + 4 = 9
    assert Action.DOUBLE not in g.legal_actions()
    assert Action.SURRENDER not in g.legal_actions()


def test_surrender_costs_half(scripted):
    g = BlackjackGame(shoe=scripted([10, 10, 6, 7]))
    g.start_round(2)
    assert Action.SURRENDER in g.legal_actions()
    play(g, Action.SURRENDER)
    assert g.round_profit == -1
    assert g.results[0].label == "surrender"


def test_split_creates_two_hands_played_in_order(scripted):
    # player 8,8 vs dealer 6 (hole 10). Split -> hand1 gets 3, hand2 gets 2 when reached.
    g = BlackjackGame(shoe=scripted([8, 6, 8, 10, 3, 2, 10, 10, 9]))
    g.start_round(1)
    assert Action.SPLIT in g.legal_actions()
    play(g, Action.SPLIT)
    assert len(g.player_hands) == 2 and g.current == 0
    assert [c.rank for c in g.player_hands[0].cards] == [8, 3]
    assert len(g.player_hands[1].cards) == 1  # second hand waits for its card
    assert Action.SURRENDER not in g.legal_actions()
    assert Action.DOUBLE in g.legal_actions()  # DAS
    play(g, Action.STAND)                       # hand 1: 11 stands (silly but legal)
    assert g.current == 1 and [c.rank for c in g.player_hands[1].cards] == [8, 2]
    play(g, Action.HIT)                         # 10 -> 20
    play(g, Action.STAND)
    assert not g.round_active
    # dealer 6+10=16 hits 9 -> 25 bust: both hands win
    assert g.round_profit == 2


def test_no_das_blocks_double_after_split(scripted):
    g = BlackjackGame(rules=Rules(double_after_split=False), shoe=scripted([8, 6, 8, 10, 3, 2, 10, 10, 9]))
    g.start_round(1)
    play(g, Action.SPLIT)
    assert Action.DOUBLE not in g.legal_actions()


def test_split_aces_get_one_card_and_21_is_not_blackjack(scripted):
    g = BlackjackGame(shoe=scripted([1, 6, 1, 10, 10, 10, 10]))  # dealer 6+10=16 -> hits 10 -> bust
    g.start_round(1)
    play(g, Action.SPLIT)
    assert not g.round_active  # both hands auto-finished with one card each
    assert all(len(h.cards) == 2 and h.total == 21 for h in g.player_hands)
    assert all(not h.is_blackjack for h in g.player_hands)
    assert g.round_profit == 2  # 1:1 each, not 3:2


def test_max_splits_respected(scripted):
    ranks = [8, 6, 8, 10] + [8, 8, 8, 8, 8, 8]  # keep drawing 8s
    g = BlackjackGame(rules=Rules(max_splits=2), shoe=scripted(ranks))
    g.start_round(1)
    play(g, Action.SPLIT)   # 2 hands
    play(g, Action.SPLIT)   # 3 hands (max)
    assert len(g.player_hands) == 3
    assert Action.SPLIT not in g.legal_actions()


def test_hi_lo_count_tracks_visible_cards_and_hole_card_at_end(scripted):
    g = BlackjackGame(shoe=scripted([2, 10, 3, 5, 10]))  # player 2,3 ; dealer T up, 5 hole
    g.start_round(1)
    # visible so far: 2 (+1), T (-1), 3 (+1) = +1 ; hole 5 not yet counted
    assert g.shoe.running_count == 1
    play(g, Action.STAND)
    # hole 5 (+1) revealed, dealer draws T (-1) -> total +1
    assert g.shoe.running_count == 1
    assert g.dealer_hand.total == 25


def test_shoe_reshuffles_at_penetration():
    g = BlackjackGame(rules=Rules(num_decks=1, penetration=0.5))
    seen_shuffle = False
    for _ in range(40):
        g.start_round(1)
        while g.round_active:
            g.step(Action.STAND)
        seen_shuffle |= g.shuffled_before_round
    assert seen_shuffle
    assert g.shoe.num_shuffles >= 2


def test_illegal_action_raises(scripted):
    g = BlackjackGame(shoe=scripted([2, 6, 3, 10, 4]))
    g.start_round(1)
    g.step(Action.HIT)
    with pytest.raises(ValueError):
        g.step(Action.DOUBLE)


def test_random_play_never_breaks_invariants():
    import random
    rng = random.Random(7)
    g = BlackjackGame(rules=Rules(num_decks=2, penetration=0.9, resplit_aces=True), rng=rng)
    for _ in range(3000):
        over = g.start_round(1)
        while g.round_active:
            g.step(rng.choice(g.legal_actions()))
        assert not g.round_active
        assert len(g.results) == len(g.player_hands) <= g.rules.max_hands
        for r in g.results:
            assert -r.hand.bet <= r.profit <= r.hand.bet * 1.5
        assert g.dealer_hand.total >= 17 or all(h.is_bust or h.is_surrendered or h.is_blackjack for h in g.player_hands) \
            or g.dealer_hand.is_blackjack
