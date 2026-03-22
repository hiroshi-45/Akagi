# -*- coding: utf-8 -*-
"""Placement-aware strategy adjustments.

In competitive mahjong (especially ranked/tournament play), final placement
matters more than raw point accumulation. This module adjusts action selection
to optimize for placement rather than just expected value.

Key concepts:
- ラス回避 (last-place avoidance): The #1 priority in ranked mahjong
- トップ取り (first-place pursuit): Important but secondary to avoiding last
- 順位点 (placement bonus): +30/+10/-10/-30 or similar uma structure
- オーラス判断 (all-last decisions): Special endgame strategy

Enhanced with:
- Precise reversal condition calculation (ツモ/ロン/直撃)
- Damaten in broader situations (not just all-last 1st)
- Supply stick (kyotaku) influence on strategy
- Dealer repeat (連荘) strategy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .game_state import GameState


@dataclass
class PlacementAdjustment:
    """Adjustment to apply to Q-value-based action selection."""
    # Multiplier for riichi Q-value bonus (< 1.0 = discourage riichi)
    riichi_multiplier: float = 1.0
    # Multiplier for meld (chi/pon) Q-value bonus (< 1.0 = discourage melding)
    meld_multiplier: float = 1.0
    # Extra safety weight to add (0.0 = no change)
    extra_safety: float = 0.0
    # Whether to consider damaten over riichi
    prefer_damaten: bool = False
    # Minimum hand value threshold to push (in estimated points)
    min_push_value: float = 0.0
    reason: str = ""


def compute_placement_adjustment(gs: GameState) -> PlacementAdjustment:
    """Compute strategic adjustments based on placement context."""
    placement = gs.my_placement
    diff_above = gs.diff_to_above
    diff_below = gs.diff_to_below
    is_dealer = gs.is_dealer_me

    # === All Last (オーラス) ===
    if gs.is_all_last:
        return _all_last_strategy(gs, placement, diff_above, diff_below, is_dealer)

    # === South round (南場) ===
    if gs.is_south:
        return _south_strategy(gs, placement, diff_above, diff_below, is_dealer)

    # === East round (東場) — mostly standard play ===
    return _east_strategy(gs, placement, diff_above, diff_below, is_dealer)


def _all_last_strategy(gs: GameState, placement: int,
                        diff_above: int, diff_below: int,
                        is_dealer: bool) -> PlacementAdjustment:
    """Special strategy for the final round.

    Enhanced with:
    - Precise reversal calculations and supply stick awareness
    - Direct hit (直撃) conditions for targeted reversal
    - Noten penalty (ノーテン罰符) awareness
    """
    kyotaku_bonus = gs.kyotaku * 1000

    if placement == 1:
        # Leading in all-last: protect the lead
        # Consider noten penalty: if lead <= 3000, being noten at ryukyoku
        # could cost us 1st place (worst case -3000 from noten penalty)
        noten_risk = diff_below <= 3000
        if noten_risk:
            # Thin lead + noten penalty risk: need to stay tenpai or win
            # Push slightly harder to maintain tenpai
            return PlacementAdjustment(
                riichi_multiplier=0.8,
                meld_multiplier=0.9,
                extra_safety=0.08,
                prefer_damaten=True,
                reason="all-last 1st, thin lead, noten penalty risk"
            )

        # Key question: can opponents overtake with a tsumo?
        if diff_below >= 12000:
            return PlacementAdjustment(
                riichi_multiplier=0.5,
                meld_multiplier=0.7,
                extra_safety=0.25,
                prefer_damaten=True,
                reason="all-last 1st, big lead - protect"
            )
        if diff_below >= 4000:
            return PlacementAdjustment(
                riichi_multiplier=0.7,
                meld_multiplier=0.8,
                extra_safety=0.15,
                prefer_damaten=True,
                reason="all-last 1st, moderate lead"
            )
        # Thin lead: be careful but can push decent hands
        return PlacementAdjustment(
            riichi_multiplier=0.85,
            meld_multiplier=0.9,
            extra_safety=0.10,
            prefer_damaten=True,
            reason="all-last 1st, thin lead"
        )

    if placement == 2:
        # Calculate what's needed to reach 1st
        pts_for_1st = gs.points_needed_for_placement(1)
        han_for_1st_ron = gs.min_han_for_points(pts_for_1st, is_tsumo=False)
        han_for_1st_tsumo = gs.min_han_for_points(pts_for_1st, is_tsumo=True)

        # Check direct hit on 1st place player
        first_seat = None
        for i, p in enumerate(gs.players):
            if i != gs.player_id and p.score >= max(pp.score for pp in gs.players):
                first_seat = i
                break
        if first_seat is not None:
            direct_pts = gs.points_needed_direct_hit(first_seat, 1)
            han_direct = gs.min_han_for_points(direct_pts, is_tsumo=False)
            # Direct hit is easier than normal ron (opponent loses what we gain)
            if han_direct < han_for_1st_ron and han_direct <= 2:
                return PlacementAdjustment(
                    riichi_multiplier=1.1,
                    meld_multiplier=1.0,
                    extra_safety=0.0,
                    min_push_value=max(1000, direct_pts),
                    reason=f"all-last 2nd, direct hit on 1st ({han_direct}han)"
                )

        if han_for_1st_ron <= 3:
            # Can overtake 1st with a small hand direct hit
            return PlacementAdjustment(
                riichi_multiplier=1.1,
                meld_multiplier=0.9,
                extra_safety=0.05,
                min_push_value=max(2000, pts_for_1st),
                reason=f"all-last 2nd, 1st reachable ({han_for_1st_ron}han ron)"
            )
        # Focus on defending 2nd
        if diff_below < 4000:
            # 3rd is close: push to extend lead
            return PlacementAdjustment(
                riichi_multiplier=1.0,
                meld_multiplier=1.0,
                extra_safety=0.05,
                prefer_damaten=False,
                reason="all-last 2nd, 3rd is close - push"
            )
        if diff_below < 8000:
            return PlacementAdjustment(
                riichi_multiplier=0.9,
                meld_multiplier=0.9,
                extra_safety=0.10,
                prefer_damaten=True,
                reason="all-last 2nd, moderate gap to 3rd"
            )
        # Safe lead over 3rd
        if han_for_1st_tsumo <= 4:
            # Can still reach 1st with tsumo, worth a shot
            return PlacementAdjustment(
                riichi_multiplier=0.95,
                meld_multiplier=0.85,
                extra_safety=0.10,
                prefer_damaten=False,
                reason="all-last 2nd, safe over 3rd, 1st reachable by tsumo"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.8,
            meld_multiplier=0.8,
            extra_safety=0.15,
            prefer_damaten=True,
            reason="all-last 2nd, safe lead over 3rd"
        )

    if placement == 3:
        pts_for_2nd = gs.points_needed_for_placement(2)
        han_for_2nd = gs.min_han_for_points(pts_for_2nd, is_tsumo=False)

        if diff_below < 4000:
            # 4th is close: also need to avoid dropping
            if han_for_2nd <= 2:
                return PlacementAdjustment(
                    riichi_multiplier=1.0,
                    meld_multiplier=1.1,
                    extra_safety=0.05,
                    reason="all-last 3rd, 2nd close, 4th near - fast agari"
                )
            return PlacementAdjustment(
                riichi_multiplier=0.9,
                meld_multiplier=1.0,
                extra_safety=0.10,
                reason="all-last 3rd, 4th near - careful"
            )
        if han_for_2nd <= 2:
            return PlacementAdjustment(
                riichi_multiplier=1.0,
                meld_multiplier=1.1,
                extra_safety=0.0,
                reason="all-last 3rd, close to 2nd"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.9,
            meld_multiplier=1.0,
            extra_safety=0.10,
            reason="all-last 3rd, steady play"
        )

    # placement == 4
    pts_for_3rd = gs.points_needed_for_placement(3)
    han_for_3rd_ron = gs.min_han_for_points(pts_for_3rd, is_tsumo=False)
    han_for_3rd_tsumo = gs.min_han_for_points(pts_for_3rd, is_tsumo=True)

    if is_dealer:
        # Dealer 4th in all-last: keep dealing (連荘) is the priority
        # Any agari works for renchan
        return PlacementAdjustment(
            riichi_multiplier=1.1,
            meld_multiplier=1.2,  # fast melds for any agari
            extra_safety=-0.15,
            reason="all-last 4th dealer, any agari for renchan"
        )

    if han_for_3rd_ron <= 1:
        # Very close to 3rd: any agari works
        return PlacementAdjustment(
            riichi_multiplier=1.0,
            meld_multiplier=1.3,  # encourage fast melds
            extra_safety=-0.10,
            reason=f"all-last 4th, very close ({pts_for_3rd}pts needed)"
        )
    if han_for_3rd_tsumo <= 3:
        return PlacementAdjustment(
            riichi_multiplier=1.2,
            meld_multiplier=1.1,
            extra_safety=-0.05,
            min_push_value=max(2000, pts_for_3rd),
            reason=f"all-last 4th, reachable ({han_for_3rd_tsumo}han tsumo)"
        )
    if han_for_3rd_ron <= 5:
        # Need mangan
        return PlacementAdjustment(
            riichi_multiplier=1.3,
            meld_multiplier=0.8,  # discourage cheap melds
            extra_safety=-0.10,
            min_push_value=5200,
            reason="all-last 4th, need mangan"
        )
    # Desperate
    return PlacementAdjustment(
        riichi_multiplier=1.1,
        meld_multiplier=0.7,
        extra_safety=-0.10,
        min_push_value=7700,
        reason="all-last 4th, desperate"
    )


def _south_strategy(gs: GameState, placement: int,
                     diff_above: int, diff_below: int,
                     is_dealer: bool) -> PlacementAdjustment:
    """Strategy adjustments for south round (not all-last).

    Enhanced with kyotaku awareness and dealer repeat value.
    """
    kyotaku_bonus = gs.kyotaku * 1000

    if placement == 1:
        if diff_below >= 20000:
            return PlacementAdjustment(
                riichi_multiplier=0.7,
                meld_multiplier=0.8,
                extra_safety=0.15,
                prefer_damaten=True,
                reason="south 1st, huge lead"
            )
        if diff_below >= 8000:
            # Comfortable: damaten if hand is already strong
            return PlacementAdjustment(
                riichi_multiplier=0.85,
                meld_multiplier=0.9,
                extra_safety=0.08,
                prefer_damaten=diff_below <= 12000,  # damaten only if not super safe
                reason="south 1st, comfortable lead"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.95,
            extra_safety=0.05,
            reason="south 1st, thin lead"
        )

    if placement == 4:
        if diff_above >= 30000:
            return PlacementAdjustment(
                riichi_multiplier=1.15,
                meld_multiplier=0.9,
                extra_safety=-0.10,
                reason="south 4th, far behind"
            )
        if is_dealer:
            # Dealer in 4th: renchan is valuable
            return PlacementAdjustment(
                riichi_multiplier=1.1,
                meld_multiplier=1.1,
                extra_safety=-0.08,
                reason="south 4th dealer, renchan value"
            )
        return PlacementAdjustment(
            riichi_multiplier=1.05,
            extra_safety=-0.05,
            reason="south 4th, moderate deficit"
        )

    # 2nd or 3rd
    if placement == 2 and diff_below < 4000:
        # 3rd is close in south: be careful not to drop
        return PlacementAdjustment(
            extra_safety=0.05,
            reason="south 2nd, 3rd is close"
        )
    if placement == 3 and diff_above < 4000:
        # Close to 2nd: push slightly
        return PlacementAdjustment(
            riichi_multiplier=1.05,
            extra_safety=-0.03,
            reason="south 3rd, close to 2nd"
        )

    return PlacementAdjustment(reason=f"south {placement}th, standard play")


def _east_strategy(gs: GameState, placement: int,
                    diff_above: int, diff_below: int,
                    is_dealer: bool) -> PlacementAdjustment:
    """Strategy for east round — mostly standard but with dealer awareness."""

    if is_dealer:
        return PlacementAdjustment(
            riichi_multiplier=1.05,
            meld_multiplier=1.05,
            extra_safety=-0.03,
            reason="east dealer, slight aggression"
        )

    return PlacementAdjustment(reason="east round, standard play")


def should_damaten(gs: GameState, adj: PlacementAdjustment,
                   hand_value: float = 0.0,
                   acceptance_count: int = 0) -> bool:
    """Whether to consider damaten (hidden tenpai) over riichi.

    Enhanced with:
    - Wait shape quality: bad shape (few tiles) favors riichi (need ura dora),
      good shape favors damaten (high win rate without revealing)
    - Turn awareness: early damaten is more valuable (opponents push more),
      late damaten is less valuable (fewer turns to win)
    - Acceptance count integration

    Still prefer riichi when:
    - Hand is cheap and needs riichi to be meaningful
    - Large kyotaku makes winning more rewarding
    - Bad wait shape where riichi ura dora is the value source
    - Very late game where intimidation matters less
    """
    if not adj.prefer_damaten:
        return False

    my_turn = gs.my_turn
    bad_wait = acceptance_count > 0 and acceptance_count <= 4

    # === Very late game: damaten loses value (fewer chances to win) ===
    # Exception: all-last 1st place where any agari wins
    if my_turn >= 14 and not (gs.is_all_last and gs.my_placement == 1):
        return False

    # === Bad wait shape: riichi adds value via ura dora and intimidation ===
    # Unless the hand is already very expensive or placement demands it
    if bad_wait and hand_value < 8000:
        # Bad wait + cheap hand: riichi is almost always better
        if not (gs.is_all_last and gs.my_placement == 1 and gs.diff_to_below <= 1000):
            return False

    # === All-last 1st place ===
    if gs.is_all_last and gs.my_placement == 1:
        lead = gs.diff_to_below

        # Mangan+ hand: riichi adds little value
        if hand_value >= 8000:
            return True

        # Very thin lead: riichi stick cost could flip placement
        if lead <= 1000:
            return True

        if lead <= 4000:
            kyotaku_bonus = gs.kyotaku * 1000
            if kyotaku_bonus >= 2000:
                return False  # Big pot justifies riichi stick
            return hand_value >= 3000

        if lead >= 12000:
            return True  # Safe enough to always damaten

        kyotaku_bonus = gs.kyotaku * 1000
        if kyotaku_bonus >= 2000:
            return False
        return hand_value >= 3000

    # === South round 1st place with prefer_damaten ===
    if gs.is_south and gs.my_placement == 1:
        lead = gs.diff_to_below
        if lead <= 1000:
            return True  # Extremely thin lead
        # Mangan+ hand in south 1st: damaten is reasonable
        if hand_value >= 8000 and lead <= 8000:
            return True

    # === Any placement: haneman+ closed hand with GOOD wait ===
    # When hand is already haneman+, riichi only adds ura dora potential
    # but reveals tenpai. Good-wait damaten has high win rate.
    # Bad-wait haneman+ still prefers riichi for intimidation.
    if hand_value >= 12000 and not gs.my_info.is_open() and not bad_wait:
        return True

    return False
