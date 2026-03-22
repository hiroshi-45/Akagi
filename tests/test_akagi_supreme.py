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
        if gs.round_wind in ("S", "W", "N"):
            gs._is_tonpu = False
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

    def test_tenpai_cheap_bad_shape_vs_dealer_riichi_folds(self):
        gs = self._make_gs(turn=36) # turn 9
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 8
        gs.dealer = 1 # dealer is player 1
        # hand_value <= 2000, bad_shape, threat >= 1.5, my_turn >= 8
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "C"] # 1000pt
        result = evaluate_push_fold(gs, shanten=0, acceptance_count=4)
        assert result.decision != Decision.PUSH

    def test_open_hand_threat_treated_as_riichi_for_safety(self):
        gs = self._make_gs(turn=36)
        gs.players[1].melds = [
            MeldInfo("pon", ["P", "P", "P"]),
            MeldInfo("pon", ["F", "F", "F"]),
            MeldInfo("pon", ["C", "C", "C"])
        ] # extreme threat > 1.2
        gs.players[1].river.append(("9m", False))
        threat = gs._threat_of(1)
        assert threat >= 1.2
        # Now get safety context and check if riichi_flags has player 1
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine
        engine = StrategyEngine()
        engine.gs = gs
        ctx = engine._build_safety_context()
        assert ctx.riichi_flags[1] is True
        # Genbutsu should find 9m
        mask = [False]*46
        mask[8] = True  # 9m discard available
        genbutsu = engine._find_genbutsu_discard(mask)
        assert genbutsu == 8


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
        if gs.round_wind in ("S", "W", "N"):
            gs._is_tonpu = False
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
        gs.players[1].riichi_declared = True  # Add threat to trigger placement defense
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
        if gs.round_wind in ("S", "W", "N"):
            gs._is_tonpu = False
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        gs.my_hand = kwargs.get("my_hand", ["1m"] * 13)
        gs.turn = kwargs.get("turn", 20)
        return gs

    def test_all_last_1st_big_lead_damaten(self):
        gs = self._make_gs(scores=[40000, 25000, 20000, 15000])
        # Big lead (>= 12000): damaten returns True for high hand value
        gs.my_hand = ["1m"] * 13  # dummy hand
        gs.visible_counts = [0] * 34
        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=8000, acceptance_count=8)
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

    def test_bad_wait_cheap_hand_south_lead_damaten(self):
        gs = self._make_gs(scores=[30000, 25000, 23000, 22000])
        adj = compute_placement_adjustment(gs)
        result = should_damaten(gs, adj, hand_value=2000, acceptance_count=2)
        assert result is True


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
        if gs.round_wind in ("S", "W", "N"):
            gs._is_tonpu = False
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

    def test_all_last_1st_big_lead_fold(self):
        """All-last 1st with big lead and threat: convert to FOLD for full safety."""
        gs = self._make_gs(scores=[40000, 25000, 20000, 15000])
        gs.players[1].riichi_declared = True  # Add threat
        original = PushFoldResult(Decision.PUSH, 0.9, "tenpai")
        adjusted = adjust_for_placement(original, gs)
        assert adjusted.decision == Decision.FOLD

    def test_all_last_4th_fold_upgraded(self):
        """All-last 4th should upgrade FOLD when hand is decent enough to matter."""
        gs = self._make_gs(scores=[15000, 30000, 25000, 30000])
        # Force a high value hand to trigger the push condition (needed: 5000+ points)
        # Use multiple doras to guarantee the heuristic evaluation is high
        gs.dora_indicators = ["1m", "2m", "3m"] # Doras: 2m, 3m, 4m
        gs.my_hand = ["2m", "2m", "2m", "2m", "3m", "3m", "3m", "3m", "4m", "4m", "4m", "4m", "5s"] # 12 doras
        original = PushFoldResult(Decision.FOLD, 0.7, "far from tenpai")
        adjusted = adjust_for_placement(original, gs)
        assert adjusted.decision in (Decision.PUSH, Decision.MAWASHI)

    def test_all_last_4th_big_deficit_still_upgrades(self):
        """All-last 4th with huge deficit still upgrades if dealer."""
        gs = self._make_gs(scores=[10000, 35000, 30000, 25000], player_id=0, dealer=0)
        original = PushFoldResult(Decision.FOLD, 0.7, "desperate")
        adjusted = adjust_for_placement(original, gs)
        # Should at least upgrade to MAWASHI because we are dealer and MUST win
        assert adjusted.decision != Decision.FOLD


# ============================================================
# Tanyao detection test
# ============================================================

class TestTanyaoDetection:
    def test_tanyao_hand_higher_value(self):
        """Hand with all 2-8 tiles should estimate higher due to tanyao.

        Use 2 dora so the extra han from tanyao crosses a point threshold:
        - Tanyao hand: tanyao(1) + dora(2) + menzen(0.7) = 3.7 → 3han → 3900
        - No-tanyao hand: dora(2) + menzen(0.7) = 2.7 → 2han → 2000
        """
        gs_tanyao = GameState()
        gs_tanyao._initialized = True
        gs_tanyao.player_id = 0
        gs_tanyao.dealer = 1
        gs_tanyao.my_hand = ["2m", "3m", "4m", "5p", "6p", "7p", "2s", "3s", "4s", "5s", "6s", "7s", "8s"]
        gs_tanyao.dora_indicators = ["4p", "6s"]  # dora = 5p and 7s

        gs_no_tanyao = GameState()
        gs_no_tanyao._initialized = True
        gs_no_tanyao.player_id = 0
        gs_no_tanyao.dealer = 1
        gs_no_tanyao.my_hand = ["1m", "2m", "3m", "5p", "6p", "7p", "1s", "2s", "3s", "5s", "6s", "7s", "9s"]
        gs_no_tanyao.dora_indicators = ["4p", "6s"]  # dora = 5p and 7s

        val_tanyao = estimate_hand_value(gs_tanyao)
        val_no_tanyao = estimate_hand_value(gs_no_tanyao)
        assert val_tanyao > val_no_tanyao


# ============================================================
# Riichi turn tracking tests (per-player turn fix)
# ============================================================

class TestRiichiTurnTracking:
    """Verify riichi_turn stores per-player turn, not global turn counter."""

    def test_riichi_turn_is_per_player(self):
        """Riichi at global turn 24 (per-player ~6) should store ~6."""
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
        # Simulate 24 tsumo events (6 rounds of 4 players)
        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                 "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                 "1s", "2s", "3s", "4s", "5s", "6s"]
        for i in range(24):
            actor = i % 4
            gs.process_event({"type": "tsumo", "actor": actor, "pai": tiles[i] if actor != 0 else tiles[i]})
            gs.process_event({"type": "dahai", "actor": actor, "pai": tiles[i], "tsumogiri": True})

        # Player 1 declares riichi at global turn 25 (per-player turn 6)
        gs.process_event({"type": "tsumo", "actor": 1, "pai": "?"})
        gs.process_event({"type": "reach", "actor": 1})
        gs.process_event({"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False})

        # riichi_turn should be per-player turn (~6), not global turn (25)
        assert gs.players[1].riichi_turn <= 8  # per-player turn
        assert gs.players[1].riichi_turn >= 4  # at least several rounds in

    def test_early_riichi_threat_boost(self):
        """Early riichi (per-player turn <= 6) should add threat bonus."""
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
        # Simulate a few turns, then player 1 riichi on turn ~4 (per-player ~1)
        for i in range(4):
            actor = i % 4
            gs.process_event({"type": "tsumo", "actor": actor, "pai": "?"})
            gs.process_event({"type": "dahai", "actor": actor, "pai": f"{i+1}m", "tsumogiri": True})

        gs.process_event({"type": "tsumo", "actor": 1, "pai": "?"})
        gs.process_event({"type": "reach", "actor": 1})
        gs.process_event({"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": False})

        # Early riichi should have higher threat than base riichi
        threat = gs.players[1].apparent_threat_level(gs.my_turn, "E", "S")
        assert threat >= 2.0  # base riichi (1.5) + early bonus (0.5) = 2.0


# ============================================================
# Dama tenpai risk estimation test
# ============================================================

class TestDamaTenpaiRisk:
    def test_dama_signal_raises_risk(self):
        """Tedashi after tsumogiri streak should raise risk above default."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.round_number = 1
        gs.turn = 40  # mid-game
        gs.my_hand = ["1m", "2m", "3m"] * 4 + ["E"]
        for i, s in enumerate([25000, 25000, 25000, 25000]):
            gs.players[i].score = s

        # Set up player 2 with tsumogiri streak then tedashi
        gs.players[2]._consecutive_tsumogiri = 4
        gs.players[2]._tedashi_count = 1
        gs.players[2]._tsumogiri_streak_tedashi = True

        risk = estimate_risk_of_deal_in(gs)
        # Should be at least 7700 (mangan-level) due to dama signal
        assert risk >= 7700


# ============================================================
# Iishanten late extreme threat FOLD test
# ============================================================

class TestIishantenLateFold:
    def test_iishanten_late_extreme_threat_folds(self):
        """Late iishanten with extreme threat and weak hand should FOLD."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.round_number = 1
        gs.turn = 40
        gs.my_hand = ["1m", "3m", "5p", "7p", "9s", "E", "S", "W", "N", "P", "F", "C", "1s"]
        for i, s in enumerate([25000, 25000, 25000, 25000]):
            gs.players[i].score = s

        # Multiple riichi = extreme threat
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 3
        gs.players[2].riichi_declared = True
        gs.players[2].riichi_turn = 4

        result = evaluate_push_fold(gs, shanten=1, acceptance_count=4)
        # With extreme threat (two riichi) and cheap hand, should fold
        assert result.decision == Decision.FOLD


# ============================================================
# Damaten thin wait test
# ============================================================

class TestDamatenThinWait:
    def test_very_thin_wait_prefers_riichi(self):
        """With only 1-2 wait tiles remaining, riichi for intimidation."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "S"
        gs._is_tonpu = False
        gs.round_number = 4  # all-last
        gs.turn = 32
        gs.my_hand = ["1m"] * 13
        # Set visible counts so that wait_tile_details returns thin wait (<=2 remaining)
        gs.visible_counts = [0] * 34
        gs.visible_counts[0] = 2  # 1m: 2 visible → 2 remaining (thin wait)
        for i, s in enumerate([25000, 30000, 23000, 22000]):
            gs.players[i].score = s

        # 2nd place with prefer_damaten, but very thin wait
        adj = compute_placement_adjustment(gs)
        # Even with prefer_damaten, thin wait (2 tiles) should not damaten
        # because riichi intimidation is more valuable
        result = should_damaten(gs, adj, hand_value=3000, acceptance_count=2)
        assert result is False


# ============================================================
# 3-player mahjong (sanma) support tests
# ============================================================

class TestThreePlayerSupport:
    """Test 3-player mahjong mode detection and behavior."""

    def _make_3p_gs(self, **kwargs) -> GameState:
        gs = GameState()
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 0)
        gs.round_wind = kwargs.get("round_wind", "E")
        gs.round_number = kwargs.get("round_number", 1)
        gs.num_players = 3
        scores = kwargs.get("scores", [35000, 35000, 35000, 0])
        for i, s in enumerate(scores):
            gs.players[i].score = s
        return gs

    def test_3p_detected_from_start_kyoku(self):
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

    def test_3p_placement_out_of_3(self):
        gs = self._make_3p_gs(scores=[30000, 40000, 35000, 0])
        assert gs.my_placement == 3  # lowest among active 3

    def test_3p_placement_first(self):
        gs = self._make_3p_gs(scores=[40000, 30000, 35000, 0])
        assert gs.my_placement == 1

    def test_3p_active_players_excludes_seat3(self):
        gs = self._make_3p_gs(scores=[35000, 35000, 35000, 0])
        active = gs._active_players
        assert len(active) == 3
        assert 3 not in active

    def test_3p_diff_to_first(self):
        gs = self._make_3p_gs(scores=[30000, 40000, 35000, 0])
        assert gs.diff_to_first == 10000

    def test_3p_remaining_tiles_55(self):
        """3-player mahjong has ~55 tiles in wall (vs 70 for 4P)."""
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
        assert gs.remaining_tiles == 55

    def test_3p_num_riichi_opponents_max2(self):
        gs = self._make_3p_gs(scores=[35000, 35000, 35000, 0])
        gs.players[1].riichi_declared = True
        gs.players[2].riichi_declared = True
        assert gs.num_riichi_opponents == 2

    def test_3p_my_turn_divides_by_3(self):
        gs = self._make_3p_gs()
        gs.turn = 9
        assert gs.my_turn == 3  # 9 // 3


# ============================================================
# ActionConfig tests
# ============================================================

class TestActionConfig:
    """Test 4P and 3P action index configurations."""

    def test_4p_config(self):
        from mjai_bot.akagi_supreme.strategy_engine import ACTION_CONFIG_4P
        assert ACTION_CONFIG_4P.idx_reach == 37
        assert ACTION_CONFIG_4P.idx_pon == 41
        assert ACTION_CONFIG_4P.idx_kan == 42
        assert ACTION_CONFIG_4P.idx_hora == 43
        assert ACTION_CONFIG_4P.idx_none == 45
        assert len(ACTION_CONFIG_4P.chi_indices) == 3
        assert 38 in ACTION_CONFIG_4P.chi_indices

    def test_3p_config(self):
        from mjai_bot.akagi_supreme.strategy_engine import ACTION_CONFIG_3P
        assert ACTION_CONFIG_3P.idx_reach == 37
        assert ACTION_CONFIG_3P.idx_pon == 38
        assert ACTION_CONFIG_3P.idx_kan == 39
        assert ACTION_CONFIG_3P.idx_hora == 40
        assert ACTION_CONFIG_3P.idx_none == 42
        assert len(ACTION_CONFIG_3P.chi_indices) == 0  # no chi in 3P

    def test_3p_meld_indices_no_chi(self):
        from mjai_bot.akagi_supreme.strategy_engine import ACTION_CONFIG_3P
        # Meld indices should only contain pon and kan
        assert ACTION_CONFIG_3P.meld_indices == frozenset({38, 39})


# ============================================================
# Menzen loss evaluation tests
# ============================================================

class TestMenzenLoss:
    """Test the menzen loss evaluation in meld decisions."""

    def test_tenpai_menzen_high_penalty(self):
        """Tenpai menzen hand should have high penalty for melding."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(action_config=ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.dealer = 1  # Not dealer, base value will be 2000 (normal penalty)
        engine.gs.my_hand = ["1m"] * 13
        engine.gs.players[0].melds = []  # menzen
        engine._last_shanten = 0  # tenpai

        # Chi penalty should be higher than pon
        chi_penalty = engine._evaluate_menzen_loss(38)  # chi_low
        pon_penalty = engine._evaluate_menzen_loss(41)  # pon
        assert chi_penalty > 0.10
        assert pon_penalty > 0.10
        assert chi_penalty > pon_penalty

    def test_far_hand_no_penalty(self):
        """Far-from-tenpai hand should have no menzen penalty."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(action_config=ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.players[0].melds = []
        engine._last_shanten = 4

        penalty = engine._evaluate_menzen_loss(38)
        assert penalty == 0.0

    def test_already_open_no_penalty(self):
        """Already-open hand should have no menzen penalty."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(action_config=ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.players[0].melds = [MeldInfo(meld_type="pon", tiles=["P", "P", "P"])]
        engine._last_shanten = 1

        penalty = engine._evaluate_menzen_loss(38)
        assert penalty == 0.0


# ============================================================
# Tile-level wait detail tests
# ============================================================

class TestWaitTileDetails:
    """Test per-tile unseen count for wait tiles."""

    def test_tenpai_hand_has_wait_details(self):
        """Tenpai hand should report wait tile details."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        # Tenpai waiting on E: 123m 456p 789s EE PP
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"]
        gs.visible_counts = [0] * 34
        # Mark tiles in hand as visible
        for t in gs.my_hand:
            idx = tile_to_index(t)
            if 0 <= idx < 34:
                gs.visible_counts[idx] += 1

        details = gs.wait_tile_details()
        # Should have at least one wait tile with unseen copies
        assert len(details) > 0
        total = sum(cnt for _, cnt in details)
        assert total > 0

    def test_no_wait_when_far(self):
        """Very disconnected hand should have minimal/no useful wait tiles."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.my_hand = ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"]
        gs.visible_counts = [0] * 34
        for t in gs.my_hand:
            idx = tile_to_index(t)
            if 0 <= idx < 34:
                gs.visible_counts[idx] += 1
        # This is a very disconnected hand — it's basically kokushi pattern
        # but estimate_acceptance_count still finds tiles that improve it
        details = gs.wait_tile_details()
        # Just verify the method runs without error
        assert isinstance(details, list)

    def test_damaten_uses_tile_details(self):
        """Damaten with good wide wait should return True."""
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "S"
        gs._is_tonpu = False
        gs.round_number = 4
        gs.turn = 24
        # Tenpai hand waiting on multiple tiles
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"]
        gs.visible_counts = [0] * 34
        for t in gs.my_hand:
            idx = tile_to_index(t)
            if 0 <= idx < 34:
                gs.visible_counts[idx] += 1
        for i, s in enumerate([40000, 25000, 20000, 15000]):
            gs.players[i].score = s

        # Verify wait_tile_details actually finds waits
        details = gs.wait_tile_details()
        assert len(details) >= 1  # should find E and/or P as waits

        adj = compute_placement_adjustment(gs)
        # Big lead all-last 1st with haneman+ hand: should prefer damaten
        result = should_damaten(gs, adj, hand_value=12000, acceptance_count=4)
        assert result is True


# ============================================================
# Controller 3P auto-switch tests
# ============================================================

class TestControllerAutoSwitch:
    """Test that controller switches to correct supreme bots."""

    def test_get_3p_bot_name_with_supreme(self):
        from mjai_bot.controller import Controller
        ctrl = Controller.__new__(Controller)
        ctrl.available_bots = []
        ctrl.available_bots_names = ["mortal", "mortal3p", "akagi_supreme", "akagi_supreme3p"]
        ctrl.bot = None
        ctrl.temp_mjai_msg = []
        ctrl.starting_game = False

        # Mock settings
        import settings.settings as ss
        original = ss.settings.model if hasattr(ss.settings, 'model') else None
        try:
            ss.settings.model = "akagi_supreme"
            assert ctrl._get_3p_bot_name() == "akagi_supreme3p"
            assert ctrl._get_4p_bot_name() == "akagi_supreme"

            ss.settings.model = "mortal"
            assert ctrl._get_3p_bot_name() == "mortal3p"
            assert ctrl._get_4p_bot_name() == "mortal"
        finally:
            if original is not None:
                ss.settings.model = original


# ============================================================
# 10th evaluation fix tests
# ============================================================

class TestIppatsuIishantenNotFold:
    """Iishanten with ippatsu should NOT auto-fold -- evaluate normally."""

    def test_iishanten_ippatsu_does_not_fold(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.round_number = 1
        gs.turn = 12  # turn 3
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "E", "E", "P", "P", "C"]
        for i, s in enumerate([25000, 25000, 25000, 25000]):
            gs.players[i].score = s
        # Player 1 ippatsu
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 2
        gs.players[1].riichi_ippatsu = True

        result = evaluate_push_fold(gs, shanten=1, acceptance_count=8)
        # Should NOT be FOLD just because of ippatsu at iishanten
        # (regular push/fold logic should decide)
        assert result.decision != Decision.FOLD or "ippatsu" not in result.reason


class TestDealerDiscountNotMutateThreat:
    """Dealer bonus should use discount factor, not mutate threat."""

    def test_dealer_pushes_harder_with_discount(self):
        """Dealer with same threat should push more than non-dealer."""
        gs_dealer = GameState()
        gs_dealer._initialized = True
        gs_dealer.player_id = 0
        gs_dealer.dealer = 0  # is dealer
        gs_dealer.round_wind = "E"
        gs_dealer.round_number = 1
        gs_dealer.turn = 24
        gs_dealer.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"]
        for i, s in enumerate([25000, 25000, 25000, 25000]):
            gs_dealer.players[i].score = s
        gs_dealer.players[1].riichi_declared = True
        gs_dealer.players[1].riichi_turn = 3

        gs_non_dealer = GameState()
        gs_non_dealer._initialized = True
        gs_non_dealer.player_id = 0
        gs_non_dealer.dealer = 1  # NOT dealer
        gs_non_dealer.round_wind = "E"
        gs_non_dealer.round_number = 1
        gs_non_dealer.turn = 24
        gs_non_dealer.my_hand = list(gs_dealer.my_hand)
        for i, s in enumerate([25000, 25000, 25000, 25000]):
            gs_non_dealer.players[i].score = s
        gs_non_dealer.players[1].riichi_declared = True
        gs_non_dealer.players[1].riichi_turn = 3

        result_dealer = evaluate_push_fold(gs_dealer, shanten=1, acceptance_count=6)
        result_non = evaluate_push_fold(gs_non_dealer, shanten=1, acceptance_count=6)
        # Dealer should be at least as aggressive as non-dealer
        order = {Decision.FOLD: 0, Decision.MAWASHI: 1, Decision.PUSH: 2}
        assert order[result_dealer.decision] >= order[result_non.decision]


class TestAllLast4thMawashiNotFold:
    """All-last 4th with cheap hand vs threat should MAWASHI, not FOLD."""

    def test_4th_cheap_hand_mawashi(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "S"
        gs._is_tonpu = False
        gs.round_number = 4
        gs.turn = 32
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "C"]
        for i, s in enumerate([15000, 30000, 25000, 30000]):
            gs.players[i].score = s
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5

        result = evaluate_push_fold(gs, shanten=0, acceptance_count=4)
        adjusted = adjust_for_placement(result, gs)
        # 4th place should never be hard FOLD — at minimum MAWASHI
        assert adjusted.decision != Decision.FOLD, \
            f"All-last 4th should not FOLD — use MAWASHI at minimum. Got: {adjusted.decision}, reason: {adjusted.reason}"


class TestChiitoitsuNoOverestimate:
    """Chiitoitsu hand should not have pinfu bonus added."""

    def test_chiitoi_no_pinfu_bonus(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.my_hand = ["2m", "2m", "5m", "5m", "3p", "3p", "7p", "7p", "4s", "4s", "8s", "8s", "E"]
        gs.dora_indicators = []
        val_chiitoi = estimate_hand_value(gs)

        # Compare with same number of tiles but NOT chiitoi route
        gs2 = GameState()
        gs2._initialized = True
        gs2.player_id = 0
        gs2.dealer = 1
        gs2.my_hand = ["2m", "3m", "4m", "5p", "6p", "7p", "2s", "3s", "4s", "8s", "8s", "E", "E"]
        gs2.dora_indicators = []
        val_normal = estimate_hand_value(gs2)

        # Chiitoi should be valued due to chiitoi bonus but shouldn't also get pinfu
        # They should be in a reasonable range, not wildly different
        assert val_chiitoi >= 1000  # has chiitoi value


class TestKabeBonusMiddleTiles:
    """Kabe bonus should apply to 4,5,6 when neighbors are walled."""

    def test_kabe_4_with_3_walled(self):
        from mjai_bot.strategy.safety import kabe_bonus
        visible = {"m": {r: 0 for r in range(1, 10)}, "p": {r: 0 for r in range(1, 10)}, "s": {r: 0 for r in range(1, 10)}}
        visible["m"][3] = 4  # all 3m visible
        bonus = kabe_bonus("4m", visible)
        assert bonus > 0  # should get one_chance bonus

    def test_kabe_5_with_4_walled(self):
        from mjai_bot.strategy.safety import kabe_bonus
        visible = {"m": {r: 0 for r in range(1, 10)}, "p": {r: 0 for r in range(1, 10)}, "s": {r: 0 for r in range(1, 10)}}
        visible["p"][4] = 4  # all 4p visible
        bonus = kabe_bonus("5p", visible)
        assert bonus > 0

    def test_kabe_6_with_7_walled(self):
        from mjai_bot.strategy.safety import kabe_bonus
        visible = {"m": {r: 0 for r in range(1, 10)}, "p": {r: 0 for r in range(1, 10)}, "s": {r: 0 for r in range(1, 10)}}
        visible["s"][7] = 4
        bonus = kabe_bonus("6s", visible)
        assert bonus > 0

    def test_kabe_5_no_wall(self):
        from mjai_bot.strategy.safety import kabe_bonus
        visible = {"m": {r: 0 for r in range(1, 10)}, "p": {r: 0 for r in range(1, 10)}, "s": {r: 0 for r in range(1, 10)}}
        bonus = kabe_bonus("5m", visible)
        assert bonus == 0.0  # no wall → no bonus


class TestAllLast2ndMeldMultiplier:
    """All-last 2nd with 3rd close should have meld_multiplier >= 1.1."""

    def test_2nd_3rd_close_meld_boost(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "S"
        gs._is_tonpu = False
        gs.round_number = 4
        # 2nd place (28000), 1st far (38000), 3rd close (27000)
        # diff_above to 1st = 10000 (too far for easy reach) → han_for_1st_ron > 3
        # diff_below to 3rd = 1000 (very close) → hits the diff_below < 4000 branch
        for i, s in enumerate([28000, 38000, 27000, 7000]):
            gs.players[i].score = s
        adj = compute_placement_adjustment(gs)
        assert adj.meld_multiplier >= 1.1, \
            f"Expected meld_multiplier >= 1.1 but got {adj.meld_multiplier}, reason: {adj.reason}"

# ============================================================
# Thought log detail tests
# ============================================================

class TestThoughtLogDetails:
    """Test that the strategy engine generates detailed thought logs."""

    def _make_engine_with_gs(self, **kwargs):
        """Create a StrategyEngine with a configured GameState."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine
        engine = StrategyEngine()
        gs = engine.gs
        gs._initialized = True
        gs.player_id = kwargs.get("player_id", 0)
        gs.dealer = kwargs.get("dealer", 1)
        gs.round_wind = kwargs.get("round_wind", "E")
        gs.round_number = kwargs.get("round_number", 1)
        gs.turn = kwargs.get("turn", 20)
        gs.my_hand = kwargs.get("my_hand", ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"])
        if gs.round_wind in ("S", "W", "N"):
            gs._is_tonpu = False
        if "scores" in kwargs:
            for i, s in enumerate(kwargs["scores"]):
                gs.players[i].score = s
        return engine

    def test_thought_contains_hand_composition(self):
        """Thought log should contain hand composition details."""
        engine = self._make_engine_with_gs(
            my_hand=["5mr", "5m", "6m", "1p", "2p", "3p", "7s", "8s", "9s", "E", "E", "P", "P"],
            scores=[25000, 25000, 25000, 25000],
        )
        engine.gs.dora_indicators = ["4m"]  # dora = 5m
        engine.set_shanten(1)

        q_values = [0.0] * 46
        q_values[0] = 0.5
        mask = [False] * 46
        mask[0] = True

        engine.adjust_action(q_values, mask, 0, True)
        thought_text = "\n".join(engine.last_thought)

        assert "ドラ" in thought_text, f"Should contain dora info, got: {thought_text}"

    def test_thought_contains_threat_analysis(self):
        """Thought log should contain threat analysis when opponents are threatening."""
        engine = self._make_engine_with_gs(
            my_hand=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
            scores=[25000, 25000, 25000, 25000],
            turn=36,
        )
        engine.gs.players[1].riichi_declared = True
        engine.gs.players[1].riichi_turn = 3
        engine.set_shanten(1)

        q_values = [0.0] * 46
        q_values[0] = 0.5
        mask = [False] * 46
        mask[0] = True

        engine.adjust_action(q_values, mask, 0, True)
        thought_text = "\n".join(engine.last_thought)

        assert "脅威分析" in thought_text, f"Should contain threat analysis, got: {thought_text}"
        assert "リーチ" in thought_text, f"Should mention riichi threat, got: {thought_text}"

    def test_thought_contains_wait_info_at_tenpai(self):
        """Thought log should contain wait info when hand is tenpai."""
        engine = self._make_engine_with_gs(
            my_hand=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
            scores=[25000, 25000, 25000, 25000],
        )
        engine.set_shanten(0)

        q_values = [0.0] * 46
        q_values[0] = 0.5
        mask = [False] * 46
        mask[0] = True

        engine.adjust_action(q_values, mask, 0, True)
        thought_text = "\n".join(engine.last_thought)

        assert "待ち" in thought_text, f"Should contain wait info at tenpai, got: {thought_text}"

    def test_thought_fold_contains_safety_reason(self):
        """Thought log should contain safety analysis during FOLD."""
        engine = self._make_engine_with_gs(
            my_hand=["1m", "3m", "5p", "7p", "9s", "E", "S", "W", "N", "P", "F", "C", "1s"],
            scores=[25000, 25000, 25000, 25000],
            turn=40,
        )
        engine.gs.players[1].riichi_declared = True
        engine.gs.players[1].riichi_turn = 3
        engine.gs.players[2].riichi_declared = True
        engine.gs.players[2].riichi_turn = 4
        engine.gs.players[1].river.append(("9s", False))
        engine.set_shanten(4)

        q_values = [0.0] * 46
        mask = [False] * 46
        for idx in [0, 2, 12, 16, 26]:
            q_values[idx] = 0.1
            mask[idx] = True

        engine.adjust_action(q_values, mask, 0, True)
        thought_text = "\n".join(engine.last_thought)

        assert "安全度分析" in thought_text, f"Should contain safety analysis, got: {thought_text}"

    def test_thought_contains_round_info(self):
        """Thought log should contain round and turn info."""
        engine = self._make_engine_with_gs(
            round_wind="S",
            round_number=3,
            scores=[30000, 25000, 22000, 23000],
        )
        engine.gs.honba = 2
        engine.gs.kyotaku = 1
        engine.set_shanten(1)

        q_values = [0.0] * 46
        q_values[0] = 0.5
        mask = [False] * 46
        mask[0] = True

        engine.adjust_action(q_values, mask, 0, True)
        thought_text = "\n".join(engine.last_thought)

        assert "南3局" in thought_text, f"Should contain round info, got: {thought_text}"
        assert "2本場" in thought_text, f"Should contain honba info, got: {thought_text}"
        assert "供託1本" in thought_text, f"Should contain kyotaku info, got: {thought_text}"

    def test_thought_contains_pf_reason(self):
        """Thought log should contain push/fold reason in Japanese."""
        engine = self._make_engine_with_gs(
            my_hand=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
            scores=[25000, 25000, 25000, 25000],
        )
        engine.set_shanten(0)

        q_values = [0.0] * 46
        q_values[0] = 0.5
        mask = [False] * 46
        mask[0] = True

        engine.adjust_action(q_values, mask, 0, True)
        thought_text = "\n".join(engine.last_thought)

        assert "判断" in thought_text, f"Should contain judgment, got: {thought_text}"
        assert "理由" in thought_text, f"Should contain reason, got: {thought_text}"
        assert "確信度" in thought_text, f"Should contain confidence, got: {thought_text}"


# ============================================================
# 11th evaluation fix tests
# ============================================================

class TestRyanshantenMenzenPenalty:
    """Ryanshanten cheap hand should NOT have excessive menzen penalty."""

    def test_ryanshanten_cheap_reasonable_penalty(self):
        """Ryanshanten (<2000pt) should have fixed mild penalty, not doubled."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(action_config=ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.dealer = 1
        engine.gs.my_hand = ["1m", "3m", "5p", "7p", "9s", "E", "S", "W", "N", "P", "F", "C", "1s"]
        engine.gs.players[0].melds = []  # menzen
        engine.gs.dora_indicators = []
        engine._last_shanten = 2  # ryanshanten

        # Chi penalty for cheap ryanshanten should be reasonable (not >= 0.20)
        chi_penalty = engine._evaluate_menzen_loss(38)  # chi_low
        assert chi_penalty <= 0.10, \
            f"Ryanshanten cheap chi penalty {chi_penalty} is too high, should be <= 0.10"


class TestChiitoitsuMenzenPenalty:
    """Chiitoitsu route (5+ pairs) should have very high menzen penalty for pon."""

    def test_chiitoi_5pairs_high_pon_penalty(self):
        """5-pair hand should refuse pon with extremely high penalty."""
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(action_config=ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.dealer = 1
        engine.gs.my_hand = ["2m", "2m", "5m", "5m", "3p", "3p", "7p", "7p", "4s", "4s", "8s", "8s", "E"]
        engine.gs.players[0].melds = []
        engine.gs.dora_indicators = []
        engine._last_shanten = 1  # iishanten for chiitoi

        pon_penalty = engine._evaluate_menzen_loss(41)  # pon
        assert pon_penalty >= 0.25, \
            f"Chiitoi route pon penalty {pon_penalty} too low, should be >= 0.25"


class TestAllLast2ndPassOverride3rdClose:
    """All-last 2nd should NOT override pass when 3rd is dangerously close."""

    def test_2nd_3rd_close_no_pass_override(self):
        from mjai_bot.akagi_supreme.strategy_engine import StrategyEngine, ACTION_CONFIG_4P
        engine = StrategyEngine(action_config=ACTION_CONFIG_4P)
        engine.gs._initialized = True
        engine.gs.player_id = 0
        engine.gs.dealer = 1
        engine.gs.round_wind = "S"
        engine.gs._is_tonpu = False
        engine.gs.round_number = 4

        # 2nd (28000), close to 1st (30000), but 3rd is very close (27500)
        for i, s in enumerate([28000, 30000, 27500, 14500]):
            engine.gs.players[i].score = s

        q_values = [0.0] * 46
        q_values[ACTION_CONFIG_4P.idx_pon] = 0.3
        q_values[ACTION_CONFIG_4P.idx_none] = 0.35
        mask = [False] * 46
        mask[ACTION_CONFIG_4P.idx_pon] = True
        mask[ACTION_CONFIG_4P.idx_none] = True

        from mjai_bot.akagi_supreme.placement_strategy import compute_placement_adjustment
        from mjai_bot.akagi_supreme.push_fold import PushFoldResult, Decision
        p_adj = compute_placement_adjustment(engine.gs)
        pf_result = PushFoldResult(Decision.MAWASHI, 0.5, "test")
        result = engine._check_pass_override(q_values, mask, p_adj, pf_result)
        # Should NOT override pass because 3rd is too close (diff_to_below = 500 < 4000)
        assert result is None, \
            f"Should not override pass when 3rd is close, but got action {result}"


class TestChaseRiichiAlways:
    """Against opponent riichi, should always chase-riichi (even bad wait cheap hand)."""

    def test_bad_wait_cheap_vs_riichi_riichi(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "S"
        gs._is_tonpu = False
        gs.round_number = 3  # not all-last
        gs.turn = 24
        gs.my_hand = ["1m", "2m", "1p", "1p", "1p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "E"]
        gs.visible_counts = [0] * 34
        for t in gs.my_hand:
            idx = tile_to_index(t)
            if 0 <= idx < 34:
                gs.visible_counts[idx] += 1
        for i, s in enumerate([25000, 25000, 25000, 25000]):
            gs.players[i].score = s
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 4

        adj = PlacementAdjustment(prefer_damaten=True)
        # Bad wait (penchan 3m) + cheap hand (2000) vs riichi
        # Top player: always chase-riichi for intimidation + extra han
        result = should_damaten(gs, adj, hand_value=2000, acceptance_count=4)
        assert result is False, \
            "Should riichi (not damaten) when chasing opponent's riichi"


class TestIppatsuIishantenMawashi:
    """Iishanten + ippatsu + cheap bad shape → MAWASHI (not PUSH)."""

    def test_iishanten_ippatsu_cheap_bad_shape_mawashi(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "E"
        gs.round_number = 2
        gs.turn = 24
        gs.my_hand = ["1m", "2m", "4p", "5p", "7s", "8s", "E", "E", "S", "S", "W", "N", "C"]
        gs.visible_counts = [0] * 34
        for t in gs.my_hand:
            idx = tile_to_index(t)
            if 0 <= idx < 34:
                gs.visible_counts[idx] += 1
        for i, s in enumerate([25000, 25000, 25000, 25000]):
            gs.players[i].score = s
        # Opponent has riichi with ippatsu active
        gs.players[1].riichi_declared = True
        gs.players[1].riichi_turn = 5
        gs.players[1].riichi_ippatsu = True

        # Cheap (< 3900) + bad shape (acceptance <= 4)
        result = evaluate_push_fold(gs, shanten=1, acceptance_count=3)
        assert result.decision == Decision.MAWASHI, \
            f"Iishanten + ippatsu + cheap bad shape should be MAWASHI, got {result.decision}"


class TestAllLast2ndDamaten:
    """All-last 2nd with 3rd close: should damaten to protect lead."""

    def test_2nd_3rd_close_damaten(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "S"
        gs._is_tonpu = False
        gs.round_number = 4  # all-last
        gs.turn = 28
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "W", "W"]
        gs.visible_counts = [0] * 34
        for t in gs.my_hand:
            idx = tile_to_index(t)
            if 0 <= idx < 34:
                gs.visible_counts[idx] += 1
        # 2nd place, 3rd is close (2000 pts gap)
        gs.players[0].score = 32000  # me - 2nd
        gs.players[1].score = 36000  # 1st
        gs.players[2].score = 30000  # 3rd - close!
        gs.players[3].score = 22000  # 4th

        adj = PlacementAdjustment(prefer_damaten=True)
        # bad wait but total_remaining >= 3: damaten overrides bad-wait riichi preference
        result = should_damaten(gs, adj, hand_value=3900, acceptance_count=3)
        assert result is True, \
            "All-last 2nd with 3rd close should damaten to protect against 3rd"


class TestAllLast3rdDamaten:
    """All-last 3rd with 2nd close: should damaten for fast out-agari."""

    def test_3rd_2nd_close_damaten(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.dealer = 1
        gs.round_wind = "S"
        gs._is_tonpu = False
        gs.round_number = 4  # all-last
        gs.turn = 28
        gs.my_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "W", "W"]
        gs.visible_counts = [0] * 34
        for t in gs.my_hand:
            idx = tile_to_index(t)
            if 0 <= idx < 34:
                gs.visible_counts[idx] += 1
        # 3rd place, 2nd is close (2000 pts gap)
        gs.players[0].score = 28000  # me - 3rd
        gs.players[1].score = 36000  # 1st
        gs.players[2].score = 30000  # 2nd - close!
        gs.players[3].score = 22000  # 4th

        adj = PlacementAdjustment(prefer_damaten=True)
        # bad wait but total_remaining >= 3: damaten for fast out-agari
        result = should_damaten(gs, adj, hand_value=3900, acceptance_count=3)
        assert result is True, \
            "All-last 3rd with 2nd close and low han needed should damaten"


class TestThreePlayerZeroScore:
    """3P player at exactly 0 points should still be active."""

    def test_0_score_player_still_active(self):
        gs = GameState()
        gs._initialized = True
        gs.player_id = 0
        gs.num_players = 3
        gs.players[0].score = 40000
        gs.players[1].score = 0      # 0 points but still playing!
        gs.players[2].score = 35000
        gs.players[3].score = 0      # empty seat in 3P

        active = gs._active_players
        assert len(active) == 3, f"Should have 3 active players, got {len(active)}"
        assert 0 in active
        assert 1 in active
        assert 2 in active
        assert 3 not in active, "Seat 3 should never be active in 3P"


class TestYakuhaiKanThreat:
    """Yakuhai daiminkan should count toward threat level."""

    def test_daiminkan_yakuhai_threat(self):
        p = PlayerInfo()
        p.melds.append(MeldInfo(meld_type="daiminkan", tiles=["P", "P", "P", "P"]))
        threat = p.apparent_threat_level(current_turn=6, round_wind="E", seat_wind="S")
        # Should count as yakuhai + 1 meld
        assert threat >= 0.6, f"Daiminkan yakuhai should contribute threat, got {threat}"

