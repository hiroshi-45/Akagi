import pytest
from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P, ACTION_CONFIG_3P, ACTION_TILE_NAMES
from mjai_bot.akagi_supreme.game_state import GameState, PlayerInfo
from mjai_bot.akagi_supreme.placement_strategy import PlacementAdjustment, should_damaten

def test_negative_q_riichi_override():
    # p_adj.riichi_multiplier >= 1.05 and negative riichi_q
    engine = StrategyEngine(ACTION_CONFIG_4P)
    engine.gs._initialized = True
    
    q_values = [-1.0]*46
    q_values[ACTION_CONFIG_4P.idx_reach] = -0.5
    # discard
    q_values[10] = -0.4  # -0.4 is better than -0.5
    
    mask = [True]*46
    
    p_adj = PlacementAdjustment(riichi_multiplier=1.2)
    # riichi_q is -0.5, multiplier is 1.2
    # expected division: -0.5 / 1.2 = -0.4166
    # if it used multiplication: -0.5 * 1.2 = -0.6
    # condition: -0.4166 >= -0.4 - 0.05 => -0.4166 >= -0.45 (True)
    
    # fake hand value and acceptance
    res = engine._check_riichi_override(10, q_values, mask, p_adj)
    assert res == ACTION_CONFIG_4P.idx_reach

def test_three_player_acceptance():
    gs = GameState()
    gs.num_players = 3
    # 1m and 9m in hand
    gs.my_hand = ["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "S"]
    # 2m to 8m do not exist in 3P, so they should not count towards acceptance
    acc = gs.estimate_acceptance_count()
    # 1m and 9m alone without 2m,3m should not give acceptance of m tiles
    details = gs.wait_tile_details()
    m_waits = [idx for idx, remaining in details if 1 <= idx <= 7]
    assert len(m_waits) == 0, f"Wait details should not include 2m-8m. Got {m_waits}"

def test_damaten_bad_wait_fix():
    gs = GameState()
    gs._initialized = True
    gs.turn = 20
    gs.num_players = 4
    
    gs.players = [PlayerInfo() for _ in range(4)]
    # opponent in riichi
    gs.players[1].riichi_declared = True
    
    # Valid 13-tile hand waiting on 3m (penchan). E is pair.
    gs.my_hand = ["1m", "2m", "1p", "1p", "1p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "E"]
    
    adj = PlacementAdjustment(prefer_damaten=True) # allow damaten consideration
    # Calling should_damaten used to crash due to bad_wait NameError here.
    # After 11th eval fix: vs riichi, top players always chase-riichi for
    # the +1 han and intimidation effect, even with bad wait + cheap hand.
    result = should_damaten(gs, adj, hand_value=2000, acceptance_count=4)
    assert result is False  # リーチ推奨（追っかけリーチ）
