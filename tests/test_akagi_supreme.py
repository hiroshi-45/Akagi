# -*- coding: utf-8 -*-
"""Unit tests for akagi_supreme core logic.

Tests cover:
- Point calculation (_calculate_points)
- Push/fold decisions
- Placement strategy (all-last scenarios)
- Game state tracking
- Acceptance count estimation
- Open hand point estimation
- Damaten judgment
- Strategy engine hora/fold behavior
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mjai_bot.akagi_supreme.game_state import (
    GameState, PlayerInfo, MeldInfo, _calculate_points,
    _estimate_deficiency, _hand_to_34, tile_to_index, tile_base,
    indicator_to_dora,
)
from mjai_bot.akagi_supreme.push_fold import (
    Decision, PushFoldResult, evaluate_push_fold, adjust_for_placement,
    estimate_hand_value, estimate_risk_of_deal_in,
)
from mjai_bot.akagi_supreme.placement_strategy import (
    PlacementAdjustment, compute_placement_adjustment, should_damaten,
)


# ============================================================
# Point calculation tests
# ============================================================

class TestCalculatePoints:
    """Test _calculate_points against known point values."""

    def test_1han_30fu_non_dealer_ron(self):
        assert _calculate_points(1, 30, False, False) == 1000

    def test_1han_30fu_dealer_ron(self):
        assert _calculate_points(1, 30, True, False) == 1500

    def test_2han_30fu_non_dealer_ron(self):
        assert _calculate_points(2, 30, False, False) == 2000

    def test_3han_30fu_non_dealer_ron(self):
        assert _calculate_points(3, 30, False, False) == 3900

    def test_4han_30fu_non_dealer_ron(self):
        assert _calculate_points(4, 30, False, False) == 7700

    def test_mangan_non_dealer(self):
        assert _calculate_points(5, 30, False, False) == 8000

    def test_mangan_dealer(self):
        assert _calculate_points(5, 30, True, False) == 12000

    def test_haneman_non_dealer(self):
        assert _calculate_points(6, 30, False, False) == 12000

    def test_baiman_non_dealer(self):
        assert _calculate_points(8, 30, False, False) == 16000

    def test_sanbaiman_non_dealer(self):
        assert _calculate_points(11, 30, False, False) == 24000

    def test_yakuman_non_dealer(self):
        assert _calculate_points(13, 30, False, False) == 32000

    def test_yakuman_dealer(self):
        assert _calculate_points(13, 30, True, False) == 48000

    def test_chiitoitsu_25fu(self):
        assert _calculate_points(2, 25, False, False) == 1600

    def test_pinfu_tsumo_20fu(self):
        # Pinfu tsumo is actually 2han 20fu (平和+門前清自摸和)
        # basic = 20 * 2^4 = 320
        # ko = ceil100(320) = 400, oya = ceil100(640) = 700
        # total = 400 + 400 + 700 = 1500
        pts = _calculate_points(2, 20, False, True)
        assert pts == 1500

    def test_3han_30fu_dealer_ron(self):
        assert _calculate_points(3, 30, True, False) == 5800

    def test_mangan_cap_4han_40fu(self):
        assert _calculate_points(4, 40, False, False) == 8000


# ============================================================
# Indicator to dora tests
# ============================================================

class TestIndicatorToDora:
    def test_number_tile(self):
        assert indicator_to_dora("1m") == "2m"
        assert indicator_to_dora("8p") == "9p"
        assert indicator_to_dora("9s") == "1s"

    def test_wind(self):
        assert indicator_to_dora("E") == "S"
        assert indicator_to_dora("N") == "E"

    def test_dragon(self):
        assert indicator_to_dora("P") == "F"
        assert indicator_to_dora("C") == "P"


# ============================================================
# Game state tests
# ============================================================

class TestGameState:
    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 0)
        gs.round_wind = kwargs.get("round_wind", "E")
        gs.round_number = kwargs.get("round_number", 1)
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        if "my_hand" in kwargs:
            gs.my_hand = list(kwargs["my_hand"])
        return gs

    def test_my_placement_first(self):
        gs = self._make_gs(scores=[30000, 25000, 20000, 15000])
        assert gs.my_placement == 1

    def test_my_placement_last(self):
        gs = self._make_gs(scores=[15000, 25000, 20000, 30000])
        assert gs.my_placement == 4

    def test_my_placement_tiebreak(self):
        gs = self._make_gs(scores=[25000, 25000, 25000, 25000])
        assert gs.my_placement == 1

    def test_is_all_last_south4(self):
        gs = self._make_gs(round_wind="S", round_number=4)
        assert gs.is_all_last is True

    def test_is_all_last_south3(self):
        gs = self._make_gs(round_wind="S", round_number=3)
        assert gs.is_all_last is False

    def test_is_all_last_tonpu(self):
        gs = self._make_gs(round_wind="E", round_number=4)
        gs._is_tonpu = True
        assert gs.is_all_last is True

    def test_diff_to_above(self):
        gs = self._make_gs(scores=[20000, 30000, 25000, 15000])
        assert gs.diff_to_above == 5000

    def test_diff_to_below(self):
        gs = self._make_gs(scores=[20000, 30000, 25000, 15000])
        assert gs.diff_to_below == 5000

    def test_count_dora_in_hand(self):
        gs = self._make_gs(my_hand=["5mr", "5m", "1p", "2p"])
        gs.dora_indicators = ["4m"]  # dora = 5m
        count = gs.count_dora_in_hand()
        assert count == 3  # 5mr matches + red, 5m matches

    def test_my_wind(self):
        gs = self._make_gs(player_id=0, dealer=0)
        assert gs.my_wind() == "E"
        gs2 = self._make_gs(player_id=1, dealer=0)
        assert gs2.my_wind() == "S"

    def test_process_event_start_game(self):
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 2})
        assert gs.player_id == 2
        assert gs._initialized is True

    def test_process_event_start_kyoku(self):
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
        gs.process_event({
            "type": "start_kyoku",
            "bakaze": "S",
            "kyoku": 3,
            "honba": 1,
            "kyotaku": 2,
            "oya": 1,
            "scores": [20000, 30000, 25000, 15000],
            "dora_marker": "5m",
            "tehais": [
                ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13
            ]
        })
        assert gs.round_wind == "S"
        assert gs.round_number == 3
        assert gs.honba == 1
        assert gs.kyotaku == 2
        assert gs.dealer == 1
        assert len(gs.my_hand) == 13


# ============================================================
# Acceptance count / deficiency tests
# ============================================================

class TestAcceptanceCount:
    def test_tenpai_hand_low_deficiency(self):
        hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"]
        counts = _hand_to_34(hand)
        d = _estimate_deficiency(counts)
        assert d <= 1

    def test_far_hand_high_deficiency(self):
        hand = ["1m", "3m", "5m", "7m", "9m", "1p", "3p", "5p", "7p", "9p", "1s", "3s", "5s"]
        counts = _hand_to_34(hand)
        d = _estimate_deficiency(counts)
        assert d >= 4

    def test_complete_hand_zero_deficiency(self):
        # Complete hand: 123m 456p 789s EEE PP
        hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "E", "P", "P"]
        counts = _hand_to_34(hand)
        d = _estimate_deficiency(counts)
        assert d == 0


# ============================================================
# Push/fold tests
# ============================================================

class TestPushFold:
    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = kwargs.get("round_wind", "E")
        gs.round_number = kwargs.get("round_number", 1)
        gs.turn = kwargs.get("turn", 20)
        gs.my_hand = kwargs.get("my_hand", ["1m", "2m", "3m"] * 4 + ["E"])
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return gs

    def test_tenpai_always_push(self):
        gs = self._make_gs()
        result = evaluate_push_fold(gs, shanten=0, acceptance_count=8)
        assert result.decision == Decision.PUSH

    def test_far_from_tenpai_with_riichi_folds(self):
        gs = self._make_gs(turn=40)
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 4
        result = evaluate_push_fold(gs, shanten=3, acceptance_count=4)
        assert result.decision == Decision.FOLD

    def test_iishanten_early_no_threat_pushes(self):
        gs = self._make_gs(turn=16)
        result = evaluate_push_fold(gs, shanten=1, acceptance_count=10)
        assert result.decision == Decision.PUSH

    def test_ryanshanten_riichi_folds_or_mawashi(self):
        gs = self._make_gs(turn=40)
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 6
        result = evaluate_push_fold(gs, shanten=2, acceptance_count=6)
        assert result.decision in (Decision.FOLD, Decision.MAWASHI)


# ============================================================
# Placement adjustment tests
# ============================================================

class TestPlacementAdjustment:
    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = kwargs.get("round_wind", "S")
        gs.round_number = kwargs.get("round_number", 4)
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return gs

    def test_all_last_1st_prefers_damaten(self):
        gs = self._make_gs(scores=[35000, 25000, 20000, 20000])
        adj = compute_placement_adjustment(gs)
        assert adj.prefer_damaten is True
        assert adj.riichi_multiplier < 1.0

    def test_all_last_4th_dealer_encourages_melds(self):
        gs = self._make_gs(
            scores=[15000, 30000, 25000, 30000],
            player_id=0, dealer=0
        )
        adj = compute_placement_adjustment(gs)
        assert adj.meld_multiplier >= 1.1

    def test_all_last_4th_non_dealer_close(self):
        # Player 0 is 4th, only 1000 pts behind 3rd → any agari works
        gs = self._make_gs(scores=[20000, 30000, 25000, 21000])
        adj = compute_placement_adjustment(gs)
        assert adj.meld_multiplier >= 1.0

    def test_adjust_for_placement_all_last_1st_push_to_mawashi(self):
        gs = self._make_gs(scores=[35000, 25000, 20000, 20000])
        original = PushFoldResult(Decision.PUSH, 0.9, "tenpai")
        adjusted = adjust_for_placement(original, gs)
        assert adjusted.decision == Decision.MAWASHI

    def test_adjust_for_placement_all_last_4th_fold_to_push(self):
        # Player 0 is 4th, 1000 pts behind 3rd (reachable)
        gs = self._make_gs(scores=[22000, 30000, 25000, 23000])
        original = PushFoldResult(Decision.FOLD, 0.7, "far from tenpai")
        adjusted = adjust_for_placement(original, gs)
        assert adjusted.decision in (Decision.PUSH, Decision.MAWASHI)


# ============================================================
# Open hand point estimation tests
# ============================================================

class TestOpenHandPoints:
    def test_single_yakuhai_pon(self):
        p = PlayerInfo()
        p.melds = [MeldInfo(meld_type="pon", tiles=["P", "P", "P"])]
        pts = p.estimate_open_hand_points("E", "S", [])
        assert pts >= 1000

    def test_double_dragon_pon(self):
        p = PlayerInfo()
        p.melds = [
            MeldInfo(meld_type="pon", tiles=["P", "P", "P"]),
            MeldInfo(meld_type="pon", tiles=["F", "F", "F"]),
        ]
        pts = p.estimate_open_hand_points("E", "S", [])
        assert pts >= 7700

    def test_no_yaku_returns_zero(self):
        p = PlayerInfo()
        p.melds = [MeldInfo(meld_type="chi", tiles=["1m", "2m", "3m"])]
        pts = p.estimate_open_hand_points("E", "S", [])
        assert pts == 0

    def test_dora_pon(self):
        p = PlayerInfo()
        p.melds = [MeldInfo(meld_type="pon", tiles=["P", "P", "P"])]
        # P is yakuhai (1 han). Dora is "F", so no dora bonus.
        pts_no_dora = p.estimate_open_hand_points("E", "S", [])
        # Now with dora matching a meld tile
        pts_with_dora = p.estimate_open_hand_points("E", "S", ["P"])
        assert pts_with_dora > pts_no_dora


# ============================================================
# Damaten tests
# ============================================================

class TestDamaten:
    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = kwargs.get("round_wind", "S")
        gs.round_number = kwargs.get("round_number", 4)
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        gs.my_hand = kwargs.get("my_hand", ["1m"] * 13)
        gs.turn = kwargs.get("turn", 20)
        return gs

    def test_all_last_1st_big_lead_damaten(self):
        gs = self._make_gs(scores=[40000, 25000, 20000, 15000])
        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=5000, acceptance_count=8)
        assert result is True

    def test_all_last_1st_thin_lead_cheap_hand_damaten(self):
        gs = self._make_gs(scores=[25500, 25000, 24500, 25000])
        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=2000, acceptance_count=8)
        assert result is True

    def test_no_prefer_damaten_returns_false(self):
        gs = self._make_gs(
            scores=[25000, 25000, 25000, 25000],
            round_wind="E", round_number=1
        )
        adj = PlacementAdjustment(prefer_damaten=False)
        result = should_damaten(gs, adj, hand_value=8000, acceptance_count=8)
        assert result is False

    def test_bad_wait_cheap_hand_no_damaten(self):
        gs = self._make_gs(scores=[30000, 25000, 23000, 22000])
        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=2000, acceptance_count=2)
        assert result is False


# ============================================================
# Mentsu extraction order fix tests
# ============================================================

class TestMentsuExtractionOrder:
    """Test that _count_mentsu_and_partial handles both extraction orders."""

    def test_kotsu_then_shuntsu_better(self):
        """1m1m1m 2m3m4m: kotsu-first gives 2 mentsu, shuntsu-first gives 1."""
        from mjai_bot.akagi_supreme.game_state import _count_mentsu_and_partial, _hand_to_34
        hand = ["1m", "1m", "1m", "2m", "3m", "4m"]
        counts = _hand_to_34(hand)
        mentsu, partial = _count_mentsu_and_partial(counts)
        assert mentsu == 2  # 111m + 234m

    def test_shuntsu_first_better(self):
        """1m2m3m 4m5m6m: shuntsu-first gives 2, kotsu-first might miss."""
        from mjai_bot.akagi_supreme.game_state import _count_mentsu_and_partial, _hand_to_34
        hand = ["1m", "2m", "3m", "4m", "5m", "6m"]
        counts = _hand_to_34(hand)
        mentsu, partial = _count_mentsu_and_partial(counts)
        assert mentsu == 2  # 123m + 456m

    def test_mixed_optimal(self):
        """2m2m2m 3m4m5m 6m6m6m: both orders should find 3 mentsu."""
        from mjai_bot.akagi_supreme.game_state import _count_mentsu_and_partial, _hand_to_34
        hand = ["2m", "2m", "2m", "3m", "4m", "5m", "6m", "6m", "6m"]
        counts = _hand_to_34(hand)
        mentsu, partial = _count_mentsu_and_partial(counts)
        assert mentsu == 3


# ============================================================
# Post-riichi genbutsu tracking tests
# ============================================================

class TestPostRiichiGenbutsu:
    """Test that tiles passed on by riichi players are tracked as safe."""

    def test_post_riichi_safe_tiles(self):
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
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
        # Player 1 declares riichi
        gs.process_event({"type": "tsumo", "actor": 1, "pai": "?"})
        gs.process_event({"type": "reach", "actor": 1})
        gs.process_event({"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False})
        gs.process_event({"type": "reach_accepted", "actor": 1})

        # Player 2 discards 5s — riichi player 1 doesn't call ron
        gs.process_event({"type": "tsumo", "actor": 2, "pai": "?"})
        gs.process_event({"type": "dahai", "actor": 2, "pai": "5s", "tsumogiri": True})

        # 5s should be tracked as safe against player 1
        assert "5s" in gs.players[1].post_riichi_safe

    def test_post_riichi_safe_red_normalized(self):
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
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
        gs.process_event({"type": "tsumo", "actor": 1, "pai": "?"})
        gs.process_event({"type": "reach", "actor": 1})
        gs.process_event({"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": False})
        gs.process_event({"type": "reach_accepted", "actor": 1})

        # Player 2 discards 5mr — should be normalized to "5m"
        gs.process_event({"type": "tsumo", "actor": 2, "pai": "?"})
        gs.process_event({"type": "dahai", "actor": 2, "pai": "5mr", "tsumogiri": True})

        assert "5m" in gs.players[1].post_riichi_safe


# ============================================================
# Last discard tracking tests
# ============================================================

class TestLastDiscardTracking:
    def test_last_discard_tracked(self):
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
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
        gs.process_event({"type": "tsumo", "actor": 1, "pai": "?"})
        gs.process_event({"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True})
        assert gs._last_discard_tile == "9m"
        assert gs._last_discard_actor == 1

        gs.process_event({"type": "tsumo", "actor": 2, "pai": "?"})
        gs.process_event({"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False})
        assert gs._last_discard_tile == "E"
        assert gs._last_discard_actor == 2

    def test_own_discard_not_tracked(self):
        """Own discards should not update _last_discard_tile."""
        gs = GameState()
        gs.process_event({"type": "start_game", "id": 0})
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
        gs.process_event({"type": "tsumo", "actor": 1, "pai": "?"})
        gs.process_event({"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": True})
        # Our discard shouldn't overwrite
        gs.process_event({"type": "tsumo", "actor": 0, "pai": "N"})
        gs.process_event({"type": "dahai", "actor": 0, "pai": "N", "tsumogiri": True})
        assert gs._last_discard_tile == "W"  # still player 1's discard


# ============================================================
# All-last placement adjustment fix tests
# ============================================================

class TestAllLastPlacementFixes:
    def _make_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = kwargs.get("round_wind", "S")
        gs.round_number = kwargs.get("round_number", 4)
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return gs

    def test_all_last_1st_thin_lead_keeps_push(self):
        """All-last 1st with thin lead (<4000): keep pushing to secure 1st."""
        gs = self._make_gs(scores=[26000, 25000, 24000, 25000])
        original = PushFoldResult(Decision.PUSH, 0.9, "tenpai")
        adjusted = adjust_for_placement(original, gs)
        # Thin lead — should keep PUSH, not downgrade to MAWASHI
        assert adjusted.decision == Decision.PUSH

    def test_all_last_1st_big_lead_mawashi(self):
        """All-last 1st with big lead: convert to MAWASHI for safety."""
        gs = self._make_gs(scores=[40000, 25000, 20000, 15000])
        original = PushFoldResult(Decision.PUSH, 0.9, "tenpai")
        adjusted = adjust_for_placement(original, gs)
        assert adjusted.decision == Decision.MAWASHI

    def test_all_last_4th_fold_upgraded(self):
        """All-last 4th should upgrade FOLD — nothing to lose."""
        gs = self._make_gs(scores=[15000, 30000, 25000, 30000])
        original = PushFoldResult(Decision.FOLD, 0.7, "far from tenpai")
        adjusted = adjust_for_placement(original, gs)
        assert adjusted.decision in (Decision.PUSH, Decision.MAWASHI)

    def test_all_last_4th_big_deficit_still_upgrades(self):
        """All-last 4th with huge deficit still upgrades — 4th is worst outcome."""
        gs = self._make_gs(scores=[10000, 35000, 30000, 25000])
        original = PushFoldResult(Decision.FOLD, 0.7, "desperate")
        adjusted = adjust_for_placement(original, gs)
        # Should at least upgrade to MAWASHI (even with huge deficit)
        assert adjusted.decision != Decision.FOLD


# ============================================================
# Tanyao detection test
# ============================================================

class TestTanyaoDetection:
    def test_tanyao_hand_higher_value(self):
        """Hand with all 2-8 tiles should estimate higher due to tanyao.

        Add 1 dora so the extra han from tanyao crosses a point threshold:
        - Tanyao hand: tanyao(1) + dora(1) + riichi(1) + ura(0.3) = 3.3 → 3han → 3900
        - No-tanyao hand: dora(1) + riichi(1) + ura(0.3) = 2.3 → 2han → 2000
        """
        gs_tanyao = GameState()
        gs_tanyao._initialized = True
        gs_tanyao.player_id = 0
        gs_tanyao.dealer = 1
        gs_tanyao.my_hand = ["2m", "3m", "4m", "5p", "6p", "7p", "2s", "3s", "4s", "5s", "6s", "7s", "8s"]
        gs_tanyao.dora_indicators = ["4p"]  # dora = 5p, in both hands

        gs_no_tanyao = GameState()
        gs_no_tanyao._initialized = True
        gs_no_tanyao.player_id = 0
        gs_no_tanyao.dealer = 1
        gs_no_tanyao.my_hand = ["1m", "2m", "3m", "5p", "6p", "7p", "1s", "2s", "3s", "5s", "6s", "7s", "9s"]
        gs_no_tanyao.dora_indicators = ["4p"]  # dora = 5p, in both hands

        val_tanyao = estimate_hand_value(gs_tanyao)
        val_no_tanyao = estimate_hand_value(gs_no_tanyao)
        assert val_tanyao > val_no_tanyao
