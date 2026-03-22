# -*- coding: utf-8 -*-
"""Tests for akagi_supreme evaluation round 8+.

Tests cover:
- Negative Q-value riichi override handling
- 3-player acceptance count
- Bad-wait damaten fix (chase riichi vs riichi)
- Late-game damaten fix (turn >= 14)
- All-last 4th riichi preference at late game
- Tonpu set_tonpu() API
"""
import pytest
from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P, ACTION_CONFIG_3P, ACTION_TILE_NAMES
from mjai_bot.akagi_supreme.game_state import GameState, PlayerInfo
from mjai_bot.akagi_supreme.placement_strategy import PlacementAdjustment, should_damaten, compute_placement_adjustment


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

    adj = PlacementAdjustment(prefer_damaten=True)  # allow damaten consideration
    # After 11th eval fix: vs riichi, top players always chase-riichi for
    # the +1 han and intimidation effect, even with bad wait + cheap hand.
    result = should_damaten(gs, adj, hand_value=2000, acceptance_count=4)
    assert result is False  # リーチ推奨（追っかけリーチ）


# ============================================================
# Late-game damaten fix tests (turn >= 14)
# ============================================================

class TestLateGameDamaten:
    """Test should_damaten behavior at turn >= 14."""

    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = kwargs.get("round_wind", "S")
        gs.round_number = kwargs.get("round_number", 4)
        if gs.round_wind in ("S", "W", "N"):
            gs._is_tonpu = False
        gs.turn = kwargs.get("turn", 60)  # my_turn = 15
        gs.num_players = kwargs.get("num_players", 4)
        gs.my_hand = kwargs.get("my_hand", ["1m"] * 13)
        gs.visible_counts = [0] * 34
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return gs

    def test_late_game_non_all_last_damaten(self):
        """At turn 14+ in non-all-last, should damaten (save 1000pt, few draws)."""
        gs = self._make_gs(
            round_wind="S", round_number=2,
            scores=[25000, 25000, 25000, 25000]
        )
        adj = PlacementAdjustment(prefer_damaten=True)
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is True, "Non-all-last at turn 14+ should damaten"

    def test_all_last_4th_late_game_riichi(self):
        """All-last 4th at turn 14+ should riichi for +1 han."""
        gs = self._make_gs(scores=[15000, 30000, 30000, 25000])
        assert gs.my_placement == 4
        assert gs.is_all_last is True
        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is False, "All-last 4th should riichi for +1 han even at turn 14+"

    def test_all_last_1st_late_game(self):
        """All-last 1st at turn 14+ should NOT be forced to damaten by turn rule."""
        gs = self._make_gs(scores=[40000, 25000, 20000, 15000])
        assert gs.my_placement == 1
        adj = compute_placement_adjustment(gs)
        # Big lead all-last 1st with haneman: should damaten to end game safely
        result = should_damaten(gs, adj, hand_value=12000, acceptance_count=4)
        assert result is True  # big lead + haneman = damaten to end game safely

    def test_all_last_2nd_late_game_damaten(self):
        """All-last 2nd/3rd at turn 14+ should damaten to save 1000pt."""
        gs = self._make_gs(scores=[25000, 30000, 25000, 20000])
        adj = PlacementAdjustment(prefer_damaten=True)
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is True, "Late game should damaten for non-4th, non-1st"


# ============================================================
# Tonpu set_tonpu() API tests
# ============================================================

class TestSetTonpu:
    """Test the set_tonpu() API for explicit game format configuration."""

    def test_set_tonpu_makes_e4_all_last(self):
        """Setting tonpu mode makes E4 the all-last round."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.set_tonpu(True)
        gs.round_wind = "E"
        gs.round_number = 4
        assert gs.is_all_last is True

    def test_default_hanchan_e4_not_all_last(self):
        """Default (hanchan) E4 is NOT all-last."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.round_wind = "E"
        gs.round_number = 4
        assert gs._is_tonpu is False
        assert gs.is_all_last is False

    def test_tonpu_cleared_on_south(self):
        """Seeing south wind auto-corrects tonpu flag."""
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
        gs.set_tonpu(True)
        gs.process_event({
            "type": "start_kyoku", "bakaze": "S", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13
            ]
        })
        assert gs._is_tonpu is False

    def test_reset_game_clears_tonpu(self):
        """reset_game() should clear tonpu flag."""
        gs = GameState()
        gs._initialized = True
        gs.set_tonpu(True)
        gs.reset_game()
        assert gs._is_tonpu is False

    def test_e4_no_auto_tonpu_detection(self):
        """E4 should NOT auto-detect tonpu (removed unreliable heuristic)."""
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
        # Simulate E1 through E4 (could be hanchan with renchan)
        for kyoku in range(1, 5):
            gs.process_event({
                "type": "start_kyoku", "bakaze": "E", "kyoku": kyoku,
                "honba": 0, "kyotaku": 0, "oya": 0,
                "scores": [25000, 25000, 25000, 25000],
                "dora_marker": "1m",
                "tehais": [
                    ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
                    ["?"] * 13, ["?"] * 13, ["?"] * 13
                ]
            })
        # Should NOT auto-detect tonpu (removed heuristic)
        assert gs._is_tonpu is False
