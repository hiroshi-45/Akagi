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
from mjai_bot.akagi_supreme.game_state import GameState, PlayerInfo, MeldInfo
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


# ============================================================
# 3P detection persistence tests
# ============================================================

class TestThreePlayerDetectionPersistence:
    """Test that 3P detection persists even if a player reaches 0 points."""

    def test_3p_persists_when_player_hits_zero(self):
        """Once 3P is detected, it should persist even if a player hits 0 points."""
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
        # Round 1: clear 3P detection (4th seat = 0, others > 0)
        gs.process_event({
            "type": "start_kyoku", "bakaze": "E", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [35000, 35000, 35000, 0],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "9m", "1p", "2p", "3p", "5p", "6p", "7p", "1s", "2s", "3s", "E", "E"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13
            ]
        })
        assert gs.num_players == 3

        # Round 2: player 0 has been knocked to 0 points
        # Without persistence fix, this would fail 3P detection
        gs.process_event({
            "type": "start_kyoku", "bakaze": "E", "kyoku": 2,
            "honba": 0, "kyotaku": 0, "oya": 1,
            "scores": [0, 50000, 55000, 0],
            "dora_marker": "2m",
            "tehais": [
                ["1m", "9m", "1p", "2p", "3p", "5p", "6p", "7p", "1s", "2s", "3s", "E", "E"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13
            ]
        })
        assert gs.num_players == 3, \
            "3P detection should persist even when a player hits 0 points"
        assert gs.remaining_tiles == 55, \
            "Wall size should remain 55 for 3P"

    def test_4p_not_falsely_detected_as_3p(self):
        """A 4P game where player 3 has 0 points should NOT be detected as 3P."""
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
        # All players start with 25000 — clearly 4P
        gs.process_event({
            "type": "start_kyoku", "bakaze": "E", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13
            ]
        })
        assert gs.num_players == 4

        # Later round: player 3 hits 0 points
        gs.process_event({
            "type": "start_kyoku", "bakaze": "E", "kyoku": 2,
            "honba": 0, "kyotaku": 0, "oya": 1,
            "scores": [30000, 35000, 35000, 0],
            "dora_marker": "2m",
            "tehais": [
                ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13
            ]
        })
        assert gs.num_players == 4, \
            "4P game should NOT be falsely detected as 3P when player 3 hits 0"
        assert gs.remaining_tiles == 70

    def test_reset_game_resets_num_players(self):
        """reset_game() should reset num_players to 4 for clean state."""
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
        gs.process_event({
            "type": "start_kyoku", "bakaze": "E", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [35000, 35000, 35000, 0],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "9m", "1p", "2p", "3p", "5p", "6p", "7p", "1s", "2s", "3s", "E", "E"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13
            ]
        })
        assert gs.num_players == 3

        # New game starts — should reset
        gs.process_event({"type": "end_game"})
        gs.process_event({"type": "start_game", "id": 0})
        assert gs.num_players == 4, \
            "New game should reset num_players to 4"


# ============================================================
# Dealer base value correction test
# ============================================================

class TestDoubleRiichiRisk:
    """Test that double riichi risk calculation applies individual bonuses."""

    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.turn = kwargs.get("turn", 20)
        gs.num_players = 4
        gs.my_hand = ["1m"] * 13
        gs.players = [PlayerInfo() for _ in range(4)]
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return gs

    def test_double_riichi_plain_unchanged(self):
        """Double non-dealer non-early riichi should still use 1.8x base."""
        from mjai_bot.akagi_supreme.push_fold import estimate_risk_of_deal_in
        gs = self._make_gs()
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 8  # late riichi
        gs.players[2].riichi_declared = True
        gs.players[2].riichi_turn = 9  # late riichi
        risk = estimate_risk_of_deal_in(gs)
        # 5200 * 1.8 = 9360 (no dealer/early bonuses)
        assert abs(risk - 9360) < 100, f"Double plain riichi risk should be ~9360, got {risk}"

    def test_double_riichi_dealer_bonus(self):
        """Double riichi with one dealer should be higher than plain double riichi."""
        from mjai_bot.akagi_supreme.push_fold import estimate_risk_of_deal_in
        gs_plain = self._make_gs()
        gs_plain.players[1].riichi_declared = True
        gs_plain.players[1].riichi_turn = 8
        gs_plain.players[2].riichi_declared = True
        gs_plain.players[2].riichi_turn = 9
        risk_plain = estimate_risk_of_deal_in(gs_plain)

        gs_dealer = self._make_gs(dealer=1)
        gs_dealer.players[1].riichi_declared = True
        gs_dealer.players[1].riichi_turn = 8
        gs_dealer.players[1].is_dealer = True
        gs_dealer.players[2].riichi_declared = True
        gs_dealer.players[2].riichi_turn = 9
        risk_dealer = estimate_risk_of_deal_in(gs_dealer)

        assert risk_dealer > risk_plain, \
            f"Dealer double riichi ({risk_dealer}) should be > plain double riichi ({risk_plain})"

    def test_double_riichi_early_bonus(self):
        """Double riichi with early declaration should be higher than late."""
        from mjai_bot.akagi_supreme.push_fold import estimate_risk_of_deal_in
        gs_late = self._make_gs()
        gs_late.players[1].riichi_declared = True
        gs_late.players[1].riichi_turn = 8
        gs_late.players[2].riichi_declared = True
        gs_late.players[2].riichi_turn = 9
        risk_late = estimate_risk_of_deal_in(gs_late)

        gs_early = self._make_gs()
        gs_early.players[1].riichi_declared = True
        gs_early.players[1].riichi_turn = 3  # early
        gs_early.players[2].riichi_declared = True
        gs_early.players[2].riichi_turn = 5  # early
        risk_early = estimate_risk_of_deal_in(gs_early)

        assert risk_early > risk_late, \
            f"Early double riichi ({risk_early}) should be > late double riichi ({risk_late})"


class TestDealerBaseValue:
    """Verify dealer hand value estimate is reasonable."""

    def test_dealer_weak_hand_not_overestimated(self):
        """Dealer with no visible yaku should not be estimated at 3900+ points."""
        from mjai_bot.akagi_supreme.push_fold import estimate_hand_value
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 0  # is dealer
        # Hand with no dora, no yakuhai, random tiles
        gs.my_hand = ["1m", "3m", "5p", "7p", "9s", "E", "S", "W", "N", "P", "F", "C", "1s"]
        gs.dora_indicators = []
        gs.players[0].melds = [MeldInfo("chi", ["2s", "3s", "4s"])]  # open hand, no menzen

        val = estimate_hand_value(gs)
        # Open hand with no visible yaku: should be at base value (~2900)
        # Previously was 3900 (overestimate)
        assert val <= 3500, \
            f"Dealer weak open hand should not be overestimated, got {val}"
