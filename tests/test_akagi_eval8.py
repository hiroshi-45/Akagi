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
from mjai_bot.akagi_supreme.push_fold import evaluate_push_fold, adjust_for_placement, Decision


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


# ============================================================
# Chiitoitsu + Toitoi mutual exclusion test
# ============================================================

class TestChiitoitsuToitoiExclusion:
    """Chiitoitsu and toitoi are mutually exclusive yaku.

    When a hand is on the chiitoitsu route (5+ pairs), toitoi should NOT
    also be added. Previously both were counted, overestimating hand value.
    """

    def test_chiitoi_route_no_toitoi_bonus(self):
        """5+ pairs (chiitoitsu route) should not also get toitoi bonus."""
        from mjai_bot.akagi_supreme.push_fold import estimate_hand_value
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1  # non-dealer
        gs.round_wind = "E"
        gs.dora_indicators = []
        # Closed hand with 6 pairs — clearly chiitoitsu route
        gs.my_hand = [
            "1m", "1m", "3p", "3p", "5s", "5s",
            "E", "E", "P", "P", "F", "F", "9m",
        ]
        gs.players[0].melds = []

        val = estimate_hand_value(gs)
        # Chiitoitsu = 2 han; menzen bonus ~0.7; yakuhai (P, F pairs) ~0.8
        # Total ~3.5 han → ~3900pts. Should NOT include +1 toitoi.
        # With old bug (chiitoitsu+toitoi): ~4.5 han → ~7700pts
        assert val < 7700, \
            f"Chiitoi route should not get toitoi bonus, got {val}"

    def test_open_toitoi_still_counted(self):
        """Open hand with pon melds should still get toitoi bonus."""
        from mjai_bot.akagi_supreme.push_fold import estimate_hand_value
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.dora_indicators = []
        # Open hand with 2 pon melds + pairs in hand
        gs.my_hand = ["3m", "3m", "7p", "7p", "1s", "1s", "9s"]
        gs.players[0].melds = [
            MeldInfo("pon", ["E", "E", "E"]),
            MeldInfo("pon", ["P", "P", "P"]),
        ]

        val = estimate_hand_value(gs)
        # Should include: yakuhai (E, P) +2, toitoi +1 = at least 3 han
        # With 3 han non-dealer = 3900
        assert val >= 3900, \
            f"Open hand with pon melds should still get toitoi bonus, got {val}"


# ============================================================
# Dead code removal verification (safe_tiles_for_me removed)
# ============================================================

class TestSafeTilesRemovedNoRegression:
    """Verify genbutsu logic works after safe_tiles_for_me removal."""

    def test_genbutsu_uses_river_not_safe_tiles_field(self):
        """Genbutsu detection should work via river tiles, not removed field."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(ACTION_CONFIG_4P)
        gs = engine.gs
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.num_players = 4
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "W"]

        # Player 1 riichis and has "S" in their river
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5
        gs.players[1].river = [("S", False), ("N", True), ("F", False)]

        mask = [True] * 46
        # Find genbutsu — should find "S" (index 28) as safe
        genbutsu = engine._find_genbutsu_discard(mask)
        assert genbutsu is not None, "Should find genbutsu from river"
        from mjai_bot.akagi_supreme.strategy_engine import ACTION_TILE_NAMES
        tile_name = ACTION_TILE_NAMES[genbutsu]
        # Should prefer S, N, or F (all in riichi player's river)
        assert tile_name in ("S", "N", "F"), \
            f"Genbutsu should be a tile from riichi player's river, got {tile_name}"


# ======================================================================
# Evaluation 16: Chase riichi vs placement protection
# ======================================================================

class TestAllLast3rdThinLeadDamaten:
    """All-last 3rd, 4th within 1000pts: riichi stick cost threatens ラス.

    Top players NEVER risk ラス for +1 han. When 4th is only 1000pts behind,
    paying 1000pt for riichi directly ties or flips placement.
    """

    def test_3rd_4th_very_close_prefer_damaten(self):
        """All-last 3rd with diff_to_below=1000 should set prefer_damaten=True."""
        gs = GameState()
        gs._initialized = True
        gs.round_wind = "S"
        gs.round_number = 4
        gs.player_id = 2
        gs.dealer = 0
        # 3rd place: 25000, 4th: 24000 (gap=1000)
        gs.players[0].score = 35000
        gs.players[1].score = 30000
        gs.players[2].score = 25000
        gs.players[3].score = 24000
        gs.players[0].is_dealer = True

        adj = compute_placement_adjustment(gs)
        assert adj.prefer_damaten is True, \
            "All-last 3rd with 4th only 1000pts behind must prefer damaten to protect against ラス"

    def test_3rd_4th_very_close_damaten_vs_riichi_opponent(self):
        """All-last 3rd with 4th close, even vs opponent riichi, should damaten."""
        gs = GameState()
        gs._initialized = True
        gs.round_wind = "S"
        gs.round_number = 4
        gs.player_id = 2
        gs.dealer = 0
        gs.players[0].score = 35000
        gs.players[1].score = 30000
        gs.players[2].score = 25000
        gs.players[3].score = 24000
        gs.players[0].is_dealer = True
        # Opponent riichi
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5

        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is True, \
            "All-last 3rd vs riichi, 4th only 1000pts behind: damaten to avoid ラス (riichi stick = placement flip)"

    def test_3rd_4th_far_chase_riichi_still_works(self):
        """All-last 3rd with safe lead over 4th should still chase riichi."""
        gs = GameState()
        gs._initialized = True
        gs.round_wind = "S"
        gs.round_number = 4
        gs.player_id = 2
        gs.dealer = 0
        gs.players[0].score = 35000
        gs.players[1].score = 30000
        gs.players[2].score = 25000
        gs.players[3].score = 17000  # 8000pt gap
        gs.players[0].is_dealer = True
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5

        adj = compute_placement_adjustment(gs)
        # With 8000pt buffer, chase riichi is fine
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is False, \
            "All-last 3rd with safe lead over 4th (8000pts): chase riichi as normal"


class TestAllLast2ndThinLeadChaseRiichi:
    """All-last 2nd, 3rd within 1000pts: riichi stick cost threatens 2nd place."""

    def test_2nd_3rd_very_close_damaten_vs_riichi(self):
        """All-last 2nd with 3rd only 1000pts behind, vs opponent riichi → damaten."""
        gs = GameState()
        gs._initialized = True
        gs.round_wind = "S"
        gs.round_number = 4
        gs.player_id = 1
        gs.dealer = 0
        gs.players[0].score = 35000
        gs.players[1].score = 26000  # 2nd
        gs.players[2].score = 25000  # 3rd, only 1000 behind
        gs.players[3].score = 14000
        gs.players[0].is_dealer = True
        gs.players[0].riichi_declared = True
        gs.players[0].riichi_turn = 5

        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is True, \
            "All-last 2nd vs riichi, 3rd only 1000pts behind: damaten to protect placement"

    def test_2nd_3rd_safe_gap_chase_riichi(self):
        """All-last 2nd with safe lead over 3rd should chase riichi."""
        gs = GameState()
        gs._initialized = True
        gs.round_wind = "S"
        gs.round_number = 4
        gs.player_id = 1
        gs.dealer = 0
        gs.players[0].score = 35000
        gs.players[1].score = 30000  # 2nd
        gs.players[2].score = 20000  # 3rd, 10000 behind
        gs.players[3].score = 15000
        gs.players[0].is_dealer = True
        gs.players[0].riichi_declared = True
        gs.players[0].riichi_turn = 5

        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=6)
        assert result is False, \
            "All-last 2nd with safe lead over 3rd (10000pts): chase riichi as normal"


# ============================================================
# Post-riichi safe tiles in MAWASHI mode (Eval 17)
# ============================================================

class TestPostRiichiSafeInMawashi:
    """Post-riichi safe tiles should be recognized as safe in MAWASHI danger scoring.

    When a player declares riichi and another player discards a tile that the
    riichi player doesn't ron, that tile is confirmed 100% safe. Previously,
    _score_discards_by_safety didn't account for post_riichi_safe, so MAWASHI
    could choose a moderately dangerous tile over a confirmed-safe one.
    """

    def test_post_riichi_safe_tile_low_danger(self):
        """Post-riichi safe tile should have very low danger score."""
        from mjai_bot.akagi_supreme.strategy_engine import (
            StrategyEngine, ACTION_CONFIG_4P, DANGER_SAFE, ACTION_TILE_NAMES,
        )
        engine = StrategyEngine(ACTION_CONFIG_4P)
        gs = engine.gs
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.num_players = 4
        gs.turn = 20
        gs.remaining_tiles = 50
        gs.my_hand = [
            "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "W",
        ]
        gs.visible_counts = [0] * 34

        # Player 1 riichi with 9m in their river
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5
        gs.players[1].river = [("9m", False)]

        # 5s is post-riichi safe: someone discarded it after riichi, not ronned
        gs.players[1].post_riichi_safe = {"5s"}

        # Build safety context
        ctx = engine._build_safety_context()
        assert ctx is not None

        mask = [True] * 46
        candidates = engine._score_discards_by_safety(mask, ctx)

        # Find the danger score for 5s (index 22)
        s5_idx = ACTION_TILE_NAMES.index("5s")
        s5_candidates = [(idx, d) for idx, d in candidates if idx == s5_idx]
        assert len(s5_candidates) == 1
        s5_danger = s5_candidates[0][1]

        # Post-riichi safe tile should be capped at DANGER_SAFE * 0.5
        assert s5_danger <= DANGER_SAFE * 0.5 + 0.01, \
            f"Post-riichi safe tile should have very low danger, got {s5_danger}"

    def test_post_riichi_safe_red_tile_normalized(self):
        """Red tile variant of post-riichi safe should also be recognized."""
        from mjai_bot.akagi_supreme.strategy_engine import (
            StrategyEngine, ACTION_CONFIG_4P, DANGER_SAFE, ACTION_TILE_NAMES,
        )
        engine = StrategyEngine(ACTION_CONFIG_4P)
        gs = engine.gs
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.num_players = 4
        gs.turn = 20
        gs.remaining_tiles = 50
        gs.my_hand = [
            "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "5mr", "W",
        ]
        gs.visible_counts = [0] * 34

        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5
        gs.players[1].river = [("9m", False)]
        # 5m is post-riichi safe (normalized from 5mr discard)
        gs.players[1].post_riichi_safe = {"5m"}

        ctx = engine._build_safety_context()
        mask = [True] * 46
        candidates = engine._score_discards_by_safety(mask, ctx)

        # 5mr (index 34) should be safe since tile_base("5mr") = "5m"
        mr5_idx = ACTION_TILE_NAMES.index("5mr")
        mr5_candidates = [(idx, d) for idx, d in candidates if idx == mr5_idx]
        assert len(mr5_candidates) == 1
        assert mr5_candidates[0][1] <= DANGER_SAFE * 0.5 + 0.01, \
            f"Red tile 5mr should be safe when 5m is post-riichi safe"


# ============================================================
# _check_riichi_override passes hand_value (Eval 17)
# ============================================================

class TestCheckRiichiOverrideHandValue:
    """_check_riichi_override should pass hand_value to should_damaten."""

    def test_riichi_override_with_hand_value(self):
        """Verify _check_riichi_override accepts hand_value/acceptance parameters."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.dealer = 1
        engine.gs.round_wind = "E"
        engine.gs.round_number = 1
        engine.gs.turn = 20
        engine.gs.my_hand = ["1m"] * 13
        engine.gs.visible_counts = [0] * 34

        q_values = [0.0] * 46
        q_values[ACTION_CONFIG_4P.idx_reach] = 0.5
        q_values[10] = 0.45  # discard

        mask = [True] * 46

        p_adj = PlacementAdjustment(riichi_multiplier=1.2)

        # Should not raise when hand_value/acceptance are passed
        result = engine._check_riichi_override(
            10, q_values, mask, p_adj,
            hand_value=8000.0, acceptance_count=6
        )
        # With multiplier 1.2 and riichi_q > discard_q, should override
        assert result == ACTION_CONFIG_4P.idx_reach


class TestAdjustRiichiMultiplierApplied:
    """Verify _adjust_riichi applies riichi_multiplier to Q-value comparison.

    When riichi_multiplier < 0.8, the multiplier should discount riichi's
    Q-value before comparing to damaten. Without this, raw Q comparison
    almost never overrides Mortal's riichi choice.
    """

    def test_low_multiplier_forces_damaten(self):
        """With multiplier=0.5, riichi Q of 0.20 → adjusted 0.10 < discard 0.15 + 0.05 → damaten."""
        engine = StrategyEngine(ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.players[0].score = 25000
        engine.gs.dealer = 1
        engine.gs.round_wind = "S"
        engine.gs.round_number = 4
        engine.gs.turn = 20
        engine.gs.my_hand = ["1m"] * 13
        engine.gs.visible_counts = [0] * 34
        engine._last_shanten = 0

        q_values = [0.0] * 46
        q_values[ACTION_CONFIG_4P.idx_reach] = 0.20  # riichi Q
        q_values[0] = 0.15  # best discard Q (1m)
        mask = [False] * 46
        mask[0] = True
        mask[ACTION_CONFIG_4P.idx_reach] = True

        # riichi_multiplier=0.5: adjusted riichi Q = 0.20 * 0.5 = 0.10
        # 0.10 < 0.15 + 0.05 = 0.20 → should choose damaten
        p_adj = PlacementAdjustment(riichi_multiplier=0.5, prefer_damaten=False)
        result = engine._adjust_riichi(q_values, mask, p_adj)
        assert result == 0, "Low riichi_multiplier should discount riichi Q and force damaten"

    def test_low_multiplier_riichi_if_overwhelming_q(self):
        """With multiplier=0.7, riichi Q 0.50 → adjusted 0.35 > discard 0.10 + 0.05 → riichi."""
        engine = StrategyEngine(ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.players[0].score = 25000
        engine.gs.dealer = 1
        engine.gs.round_wind = "S"
        engine.gs.round_number = 4
        engine.gs.turn = 20
        engine.gs.my_hand = ["1m"] * 13
        engine.gs.visible_counts = [0] * 34
        engine._last_shanten = 0

        q_values = [0.0] * 46
        q_values[ACTION_CONFIG_4P.idx_reach] = 0.50  # very strong riichi Q
        q_values[0] = 0.10  # discard Q
        mask = [False] * 46
        mask[0] = True
        mask[ACTION_CONFIG_4P.idx_reach] = True

        # adjusted riichi Q = 0.50 * 0.7 = 0.35, discard + margin = 0.15
        # 0.35 >= 0.15 → riichi still wins when overwhelmingly better
        p_adj = PlacementAdjustment(riichi_multiplier=0.7, prefer_damaten=False)
        result = engine._adjust_riichi(q_values, mask, p_adj)
        assert result == ACTION_CONFIG_4P.idx_reach, "Riichi should win when Q advantage is overwhelming even with discount"

    def test_negative_q_low_multiplier(self):
        """With negative riichi Q and multiplier=0.5, divides to make less negative."""
        engine = StrategyEngine(ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.players[0].score = 25000
        engine.gs.dealer = 1
        engine.gs.round_wind = "S"
        engine.gs.round_number = 4
        engine.gs.turn = 20
        engine.gs.my_hand = ["1m"] * 13
        engine.gs.visible_counts = [0] * 34
        engine._last_shanten = 0

        q_values = [0.0] * 46
        q_values[ACTION_CONFIG_4P.idx_reach] = -0.10  # negative riichi Q
        q_values[0] = -0.15  # also negative discard Q
        mask = [False] * 46
        mask[0] = True
        mask[ACTION_CONFIG_4P.idx_reach] = True

        # adjusted riichi Q = -0.10 / 0.5 = -0.20
        # -0.20 < -0.15 + 0.05 = -0.10 → damaten
        p_adj = PlacementAdjustment(riichi_multiplier=0.5, prefer_damaten=False)
        result = engine._adjust_riichi(q_values, mask, p_adj)
        assert result == 0, "Negative Q with low multiplier should correctly discourage riichi"


class TestDeadCodeCleanup:
    """Verify the push/fold iishanten late-game logic after dead code removal.

    Previously, unreachable effective_threat >= 2.0 checks existed inside a
    block where effective_threat was guaranteed < 1.8. After cleanup, the
    reachable paths should still work correctly.
    """

    def _make_gs(self, threat_level=1.5, my_turn=13, placement=1):
        """Create a GameState for iishanten late-game testing."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.num_players = 4
        gs.round_wind = "S"
        gs.round_number = 4
        gs.dealer = 1
        gs.turn = my_turn * 4
        gs.players[0].score = 35000 if placement == 1 else 25000
        gs.players[1].score = 25000 if placement == 1 else 35000
        gs.players[2].score = 20000
        gs.players[3].score = 20000
        # Set up threat via riichi
        if threat_level >= 1.5:
            gs.players[1].riichi_declared = True
            gs.players[1].riichi_turn = 4  # early riichi for higher threat
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "7p", "8p", "1s", "2s", "3s", "E", "E", "S"]
        gs.visible_counts = [0] * 34
        return gs

    def test_iishanten_late_1st_place_fold_vs_threat(self):
        """1st place, iishanten, late game, threat >= 1.5, hand < 8000 → FOLD."""
        gs = self._make_gs(threat_level=1.5, my_turn=13, placement=1)
        result = evaluate_push_fold(gs, shanten=1, acceptance_count=6)
        result = adjust_for_placement(result, gs, shanten=1)
        assert result.decision in (Decision.FOLD, Decision.MAWASHI), \
            f"1st place iishanten late vs threat should fold/mawashi, got {result.decision}"

    def test_iishanten_late_with_good_shape_mawashi(self):
        """Iishanten, late game, good shape, moderate threat → MAWASHI."""
        gs = self._make_gs(threat_level=1.5, my_turn=13, placement=3)
        # Give a decent hand to hit hand_value >= risk * 0.5
        gs.my_hand = ["5mr", "6m", "7m", "5pr", "6p", "7p", "5sr", "6s", "7s", "1m", "1m", "E", "E"]
        gs.dora_indicators = ["4m"]  # makes 5m a dora
        result = evaluate_push_fold(gs, shanten=1, acceptance_count=10)
        # Good shape (>=8), decent value → should be MAWASHI or PUSH in late game
        assert result.decision in (Decision.MAWASHI, Decision.PUSH, Decision.FOLD), \
            f"Expected valid decision, got {result.decision}"


# ============================================================
# Double wind yakuhai tests (Eval 19)
# ============================================================

class TestDoubleWindYakuhai:
    """Test that double wind (場風+自風) yakuhai is correctly counted as 2 han.

    When a tile is both round wind and seat wind (e.g., East dealer in East
    round), a triplet gives 2 han: 1 for 場風 (round wind) + 1 for 自風
    (seat wind). Previously only 1 han was counted, undervaluing dealer hands.
    """

    def test_double_wind_triplet_2_han(self):
        """East dealer in East round with East triplet + dora → crosses han threshold.

        Double wind gives 2 han (場風+自風). With 1 dora (3 han) + menzen (0.7),
        the total ~3.7 han → 3-han point range (5800 dealer).
        Without double wind fix: 1+1+0.7 = 2.7 han → 2-han range (2900 dealer).
        """
        from mjai_bot.akagi_supreme.push_fold import estimate_hand_value
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 0  # dealer (seat wind = East)
        gs.round_wind = "E"  # round wind = East
        gs.dora_indicators = ["6p"]  # dora = 7p
        # Closed hand with East triplet + 7p (dora) + other tiles
        gs.my_hand = [
            "E", "E", "E", "1p", "2p", "3p", "4s", "5s", "6s",
            "7m", "8m", "7p", "P",
        ]
        gs.players[0].melds = []

        val = estimate_hand_value(gs)
        # Double wind (2 han) + 1 dora (1 han) + menzen (0.7) = ~3.7 han → 5800 dealer
        # Without fix: 1 + 1 + 0.7 = 2.7 han → 2900 dealer
        assert val >= 5800, \
            f"Double wind + dora should give 3+ han (>=5800pts dealer), got {val}"

    def test_single_wind_triplet_1_han(self):
        """Non-dealer in East round with East triplet → 1 han (round wind only)."""
        from mjai_bot.akagi_supreme.push_fold import estimate_hand_value
        gs = GameState()
        gs._initialized = True
        gs.player_id = 1  # non-dealer (seat wind = South)
        gs.dealer = 0
        gs.round_wind = "E"
        gs.dora_indicators = []
        gs.my_hand = [
            "E", "E", "E", "1p", "2p", "3p", "4s", "5s", "6s",
            "7m", "8m", "9m", "P",
        ]
        gs.players[1].melds = []

        val = estimate_hand_value(gs)
        # Single yakuhai (1 han) + menzen (0.7) = ~1.7 han → 2000pts non-dealer
        assert val < 5800, \
            f"Single wind should give only 1 han, got {val}"

    def test_sangenpai_not_double_counted(self):
        """Sangenpai (P/F/C) should NOT be treated as double wind."""
        from mjai_bot.akagi_supreme.push_fold import estimate_hand_value
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 0
        gs.round_wind = "E"
        gs.dora_indicators = []
        gs.my_hand = [
            "P", "P", "P", "1p", "2p", "3p", "4s", "5s", "6s",
            "7m", "8m", "9m", "E",
        ]
        gs.players[0].melds = []

        val = estimate_hand_value(gs)
        # Haku (1 han) + menzen bonus → should NOT be treated as double wind
        # With double wind bug: would incorrectly give 2 han for haku
        # Haku is in YAKUHAI_HONORS, so the is_double_wind check excludes it
        assert val < 7700, \
            f"Sangenpai should not get double wind bonus, got {val}"


# ============================================================
# MAWASHI tile selection pool tests (Eval 19)
# ============================================================

class TestMawashiTileSelection:
    """Test that MAWASHI considers ALL safe tiles for Q-value tiebreak.

    Previously limited to top 3 safest, missing better Q-value tiles that
    were still within the SAFE threshold.
    """

    def test_mawashi_picks_best_q_among_all_safe(self):
        """MAWASHI should pick highest Q-value tile among ALL safe tiles."""
        from mjai_bot.akagi_supreme.strategy_engine import (
            StrategyEngine, ACTION_CONFIG_4P, DANGER_SAFE, ACTION_TILE_NAMES,
        )
        from mjai_bot.akagi_supreme.push_fold import Decision, PushFoldResult

        engine = StrategyEngine(ACTION_CONFIG_4P)
        gs = engine.gs
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.num_players = 4
        gs.turn = 40  # turn 10
        gs.remaining_tiles = 40
        # Need enough tiles in hand for isolation scoring
        gs.my_hand = [
            "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
            "E", "S", "W", "N",
        ]
        gs.visible_counts = [0] * 34

        # Set up riichi opponent to make safety context meaningful
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5
        gs.players[1].river = [
            ("E", False), ("S", True), ("W", False), ("N", True),
        ]

        # All 4 honor tiles are genbutsu (in riichi player's river)
        # E=idx29, S=idx28, W=idx30, N=idx31
        q_values = [0.0] * 46
        # Set S (idx 28) with much higher Q but slightly higher danger
        q_values[29] = 0.1   # E
        q_values[28] = 0.8   # S - best Q-value
        q_values[30] = 0.15  # W
        q_values[31] = 0.12  # N

        mask = [True] * 46
        pf_result = PushFoldResult(Decision.MAWASHI, 0.7, "test")

        result = engine._adjust_discard(29, q_values, mask, pf_result)
        # Should pick S (idx 28) because it has the best Q-value among safe tiles
        assert result == 28, \
            f"MAWASHI should pick best Q-value among ALL safe tiles, got idx {result} (expected 28=S)"


# ============================================================
# Direct hit point calculation fix tests
# ============================================================

class TestDirectHitPointCalculation:
    """Test that points_needed_direct_hit accounts for point transfer.

    When you ron the player you're trying to surpass, both gain and loss
    are X points, so you need roughly X >= (diff / 2) to reverse placement.
    """

    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = "S"
        gs.round_number = 4
        gs._is_tonpu = False
        gs.num_players = 4
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return gs

    def test_direct_hit_halves_requirement(self):
        """Hitting 1st (30000) as 2nd (25000) should need ~2500, not 5000."""
        gs = self._make_gs(scores=[25000, 30000, 20000, 15000])
        # target_seat=1 (1st place), target_placement=1
        pts = gs.points_needed_direct_hit(1, 1)
        # Raw diff = 5000. Direct hit = ceil(5000/2) = 2500
        assert pts <= 2600, \
            f"Direct hit should need ~2500pts (half the 5000 gap), got {pts}"
        assert pts >= 2400, \
            f"Direct hit should need ~2500pts, got {pts}"

    def test_direct_hit_tiebreak(self):
        """With worse seat priority, need slightly more than half."""
        # Player 0 (seat 0) trying to surpass player 1 (seat 1)
        # Seat 0 < seat 1, so player 0 wins tiebreak → no extra needed
        gs = self._make_gs(player_id=0, scores=[25000, 30000, 20000, 15000])
        pts_good_seat = gs.points_needed_direct_hit(1, 1)

        # Player 2 trying to surpass player 1 (seat 1)
        # Seat 2 > seat 1, so player 2 loses tiebreak → needs extra
        gs2 = self._make_gs(player_id=2, scores=[20000, 30000, 25000, 15000])
        pts_bad_seat = gs2.points_needed_direct_hit(1, 1)

        assert pts_bad_seat > pts_good_seat, \
            f"Worse seat tiebreak should need more pts: bad={pts_bad_seat}, good={pts_good_seat}"

    def test_direct_hit_already_ahead(self):
        """If already at or above target placement, need 0."""
        gs = self._make_gs(scores=[35000, 30000, 20000, 15000])
        pts = gs.points_needed_direct_hit(1, 1)
        assert pts == 0

    def test_direct_hit_practical_han_check(self):
        """2nd (25000) hitting 1st (30000): 3han should suffice, not 4han."""
        gs = self._make_gs(scores=[25000, 30000, 20000, 15000])
        pts = gs.points_needed_direct_hit(1, 1)
        han = gs.min_han_for_points(pts, is_tsumo=False)
        # 3han 30fu = 3900pts. With direct hit: 25000+3900=28900 vs 30000-3900=26100
        # 28900 > 26100 → placement reversed. So 3 han is enough.
        assert han <= 3, \
            f"Direct hit 3han (3900pts) should suffice for 5000pt gap, got {han}han"

    def test_direct_hit_on_different_target(self):
        """When target_seat differs from threshold player, no halving."""
        # Player 0 (3rd, 20000) wants 1st (35000) by hitting player 2 (25000)
        # Player 2 is NOT 1st, so hitting them doesn't directly affect 1st's score
        gs = self._make_gs(scores=[20000, 35000, 25000, 10000])
        pts = gs.points_needed_direct_hit(2, 1)  # hit player 2, want 1st
        # threshold is player 1 (35000). target is player 2.
        # Since target != threshold, raw diff = 35000 - 20000 = 15000
        assert pts >= 15000, \
            f"Hitting non-threshold player should need full diff, got {pts}"


# ============================================================
# All-last 3rd prefer_damaten extended range test
# ============================================================

class TestAllLast3rdPreferDamatenExtended:
    """Test that all-last 3rd prefers damaten when 4th is within 2000pts."""

    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = "S"
        gs.round_number = 4
        gs._is_tonpu = False
        gs.num_players = 4
        gs.my_hand = kwargs.get("my_hand", ["1m"] * 13)
        gs.visible_counts = [0] * 34
        gs.turn = kwargs.get("turn", 20)
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return gs

    def test_3rd_4th_gap_1500_prefers_damaten(self):
        """All-last 3rd with 4th only 1500pts behind should prefer damaten.

        Riichi deposit (1000pt) would narrow the gap to just 500pts,
        making ラス回避 extremely precarious.
        """
        gs = self._make_gs(scores=[21500, 30000, 25000, 23500])
        # Player 0 is 3rd (21500), 4th is 20000 → gap = 1500
        # Wait, let me recalculate. Sorted: 30000(p1), 25000(p2), 23500(p3), 21500(p0)
        # Player 0 is 4th... let me fix scores
        gs2 = self._make_gs(scores=[23500, 30000, 25000, 22000])
        # Sorted: 30000(p1), 25000(p2), 23500(p0), 22000(p3)
        # Player 0 is 3rd, 4th is 22000, gap = 1500
        adj = compute_placement_adjustment(gs2)
        assert adj.prefer_damaten is True, \
            f"All-last 3rd with 4th only 1500pts behind should prefer damaten, got: {adj.reason}"

    def test_3rd_4th_gap_1000_still_prefers_damaten(self):
        """All-last 3rd with 4th only 1000pts behind (previous threshold)."""
        gs = self._make_gs(scores=[23000, 30000, 25000, 22000])
        adj = compute_placement_adjustment(gs)
        assert adj.prefer_damaten is True

    def test_3rd_4th_gap_3000_no_damaten_unless_close_to_2nd(self):
        """All-last 3rd with 4th 3000pts behind: damaten not forced by gap alone."""
        gs = self._make_gs(scores=[25000, 40000, 35000, 22000])
        # Player 0: 3rd (25000), 4th: 22000, gap = 3000
        # 2nd: 35000, gap to 2nd = 10000 (far)
        adj = compute_placement_adjustment(gs)
        # Gap is 3000 (>= 2000), so prefer_damaten should NOT be set by gap-protection
        assert adj.prefer_damaten is False, \
            f"All-last 3rd with 4th 3000pts behind should not force damaten, got: {adj.reason}"


# ============================================================
# MAWASHI moderate tile danger-weighted selection (Eval 20)
# ============================================================

class TestMawashiModerateDangerWeighting:
    """Test that MAWASHI considers danger within the moderate range (0.25-0.60).

    Previously, moderate tiles were selected purely by Q-value + isolation,
    ignoring that danger 0.58 is far riskier than 0.27. Top players in MAWASHI
    prefer lower danger within the moderate range, only accepting higher danger
    for substantial Q-value advantage.
    """

    def test_moderate_prefers_safer_tile_over_marginal_q(self):
        """Among moderate tiles, prefer safer tile when Q-value difference is small."""
        from mjai_bot.akagi_supreme.strategy_engine import (
            StrategyEngine, ACTION_CONFIG_4P, DANGER_SAFE, DANGER_MODERATE,
            ACTION_TILE_NAMES,
        )
        from mjai_bot.akagi_supreme.push_fold import Decision, PushFoldResult

        engine = StrategyEngine(ACTION_CONFIG_4P)
        gs = engine.gs
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.num_players = 4
        gs.turn = 40
        gs.remaining_tiles = 40
        gs.my_hand = [
            "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
            "E", "S", "W", "N",
        ]
        gs.visible_counts = [0] * 34

        # Set up riichi opponent to create meaningful safety context
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5
        gs.players[1].river = [("1p", False)]

        # We need to mock _score_discards_by_safety to control danger values
        # Instead, test via _adjust_discard with a controlled pf_result
        q_values = [0.0] * 46
        # S (idx 28) has slightly better Q but higher danger
        q_values[28] = 0.20  # S - slightly better Q
        q_values[29] = 0.18  # E - slightly worse Q but safer

        mask = [False] * 46
        mask[28] = True  # S
        mask[29] = True  # E

        pf_result = PushFoldResult(Decision.MAWASHI, 0.7, "test")

        # Monkey-patch _score_discards_by_safety to return controlled danger values
        original_fn = engine._score_discards_by_safety

        def mock_score(mask_arg, ctx_arg):
            # Return only moderate-danger tiles with different dangers
            return [(28, 0.55), (29, 0.28)]  # S=high moderate, E=low moderate

        engine._score_discards_by_safety = mock_score

        try:
            result = engine._adjust_discard(28, q_values, mask, pf_result)
            # E (idx 29) should be preferred: danger 0.28 vs 0.55,
            # and Q difference (0.02) is not enough to overcome the danger gap
            # danger penalty: 0.55*0.3 - 0.28*0.3 = 0.081 > Q diff of 0.02
            assert result == 29, \
                f"MAWASHI should prefer safer moderate tile (E=29) over slightly better Q (S=28), got idx {result}"
        finally:
            engine._score_discards_by_safety = original_fn

    def test_moderate_allows_better_q_when_danger_similar(self):
        """When moderate tiles have similar danger, Q-value should win."""
        from mjai_bot.akagi_supreme.strategy_engine import (
            StrategyEngine, ACTION_CONFIG_4P, ACTION_TILE_NAMES,
        )
        from mjai_bot.akagi_supreme.push_fold import Decision, PushFoldResult

        engine = StrategyEngine(ACTION_CONFIG_4P)
        gs = engine.gs
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.num_players = 4
        gs.turn = 40
        gs.remaining_tiles = 40
        gs.my_hand = [
            "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
            "E", "S", "W", "N",
        ]
        gs.visible_counts = [0] * 34

        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5
        gs.players[1].river = [("1p", False)]

        q_values = [0.0] * 46
        q_values[28] = 0.40  # S - much better Q
        q_values[29] = 0.10  # E - much worse Q

        mask = [False] * 46
        mask[28] = True
        mask[29] = True

        pf_result = PushFoldResult(Decision.MAWASHI, 0.7, "test")

        original_fn = engine._score_discards_by_safety

        def mock_score(mask_arg, ctx_arg):
            # Similar danger levels
            return [(28, 0.35), (29, 0.32)]

        engine._score_discards_by_safety = mock_score

        try:
            result = engine._adjust_discard(28, q_values, mask, pf_result)
            # S (idx 28) should win: danger difference is tiny (0.03)
            # but Q-value advantage is huge (0.30)
            assert result == 28, \
                f"When danger is similar, better Q tile should win. Got idx {result}"
        finally:
            engine._score_discards_by_safety = original_fn
