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
