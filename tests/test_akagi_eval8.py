# -*- coding: utf-8 -*-
"""Additional tests for akagi_supreme evaluation round 8+.

Tests cover:
- Late-game damaten fix (should_damaten at turn >= 14)
- All-last 4th riichi preference at late game
- Negative Q-value riichi override handling
- Tonpu set_tonpu() API
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mjai_bot.akagi_supreme.game_state import GameState, PlayerInfo
from mjai_bot.akagi_supreme.placement_strategy import (
    PlacementAdjustment, should_damaten, compute_placement_adjustment,
)


# ============================================================
# Late-game damaten fix tests
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

    def test_all_last_1st_late_game_not_forced(self):
        """All-last 1st at turn 14+ should NOT be forced to damaten by turn rule."""
        gs = self._make_gs(scores=[40000, 25000, 20000, 15000])
        assert gs.my_placement == 1
        adj = compute_placement_adjustment(gs)
        # Big lead all-last 1st with haneman: should damaten via later logic
        result = should_damaten(gs, adj, hand_value=12000, acceptance_count=4)
        assert result is True  # big lead + haneman = damaten to end game safely

    def test_all_last_2nd_late_game_damaten(self):
        """All-last 2nd at turn 14+ should damaten to save 1000pt."""
        gs = self._make_gs(scores=[25000, 30000, 25000, 20000])
        assert gs.my_placement == 3 or gs.my_placement == 2  # tied with player 2
        adj = PlacementAdjustment(prefer_damaten=True)
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is True, "Late game should damaten for non-4th, non-1st"


# ============================================================
# Negative Q-value riichi override tests
# ============================================================

class TestNegativeQRiichiOverride:
    """Test that negative Q-values are handled correctly for riichi boost."""

    def test_negative_q_riichi_boost_by_division(self):
        """When riichi_multiplier >= 1.05 and riichi Q is negative, divide to boost."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, IDX_REACH

        engine = StrategyEngine()
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.dealer = 1
        engine.gs.my_hand = ["1m"] * 13

        q_values = [-1.0] * 46
        q_values[IDX_REACH] = -0.5  # riichi Q
        q_values[10] = -0.4  # best discard Q (better than riichi raw)
        mask = [True] * 46

        p_adj = PlacementAdjustment(riichi_multiplier=1.2)
        # With division: -0.5 / 1.2 = -0.4167
        # Condition: -0.4167 >= -0.4 - 0.05 = -0.45 → True → riichi
        # With multiplication (bug): -0.5 * 1.2 = -0.6
        # Condition: -0.6 >= -0.45 → False → no override

        # The riichi boost path should apply
        result = engine._adjust_riichi(q_values, mask, p_adj)
        assert result == IDX_REACH, "Negative Q with multiplier >= 1.05 should boost riichi via division"

    def test_positive_q_riichi_boost_by_multiplication(self):
        """When riichi Q is positive, multiply normally to boost."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, IDX_REACH

        engine = StrategyEngine()
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.dealer = 1
        engine.gs.my_hand = ["1m"] * 13

        q_values = [0.0] * 46
        q_values[IDX_REACH] = 0.4  # riichi Q
        q_values[10] = 0.45  # best discard Q (slightly better)
        mask = [True] * 46

        p_adj = PlacementAdjustment(riichi_multiplier=1.2)
        # With multiplication: 0.4 * 1.2 = 0.48
        # Condition: 0.48 >= 0.45 - 0.05 = 0.40 → True → riichi

        result = engine._adjust_riichi(q_values, mask, p_adj)
        assert result == IDX_REACH, "Positive Q should boost riichi via multiplication"


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
