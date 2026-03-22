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
    """Special strategy for the final round."""

    if placement == 1:
        # Leading: protect the lead
        if diff_below >= 12000:
            # Very safe lead: ultra-conservative
            return PlacementAdjustment(
                riichi_multiplier=0.5,
                meld_multiplier=0.7,
                extra_safety=0.25,
                prefer_damaten=True,
                reason="all-last 1st, big lead - protect"
            )
        if diff_below >= 4000:
            # Decent lead: moderate defense
            return PlacementAdjustment(
                riichi_multiplier=0.7,
                meld_multiplier=0.8,
                extra_safety=0.15,
                prefer_damaten=True,
                reason="all-last 1st, moderate lead"
            )
        # Thin lead: careful but can push with good hands
        return PlacementAdjustment(
            riichi_multiplier=0.85,
            meld_multiplier=0.9,
            extra_safety=0.10,
            prefer_damaten=True,
            reason="all-last 1st, thin lead"
        )

    if placement == 2:
        if diff_above <= 7700:
            # Can overtake 1st with a direct hit: push for top
            return PlacementAdjustment(
                riichi_multiplier=1.1,
                meld_multiplier=0.9,
                extra_safety=0.05,
                min_push_value=3900,
                reason="all-last 2nd, 1st reachable"
            )
        # Focus on defending 2nd — but adjust based on gap to 3rd.
        # If 3rd is close (< 4000), damaten risks giving away information
        # and we should play more aggressively to secure 2nd.
        if diff_below < 4000:
            # 3rd is breathing down our neck: push aggressively to extend the lead
            return PlacementAdjustment(
                riichi_multiplier=1.0,
                meld_multiplier=1.0,
                extra_safety=0.05,
                prefer_damaten=False,
                reason="all-last 2nd, 3rd is close - push to extend lead"
            )
        if diff_below < 8000:
            # Moderate gap to 3rd: balanced approach
            return PlacementAdjustment(
                riichi_multiplier=0.9,
                meld_multiplier=0.9,
                extra_safety=0.10,
                prefer_damaten=True,
                reason="all-last 2nd, moderate gap to 3rd"
            )
        # Safe lead over 3rd: defend 2nd comfortably
        return PlacementAdjustment(
            riichi_multiplier=0.8,
            meld_multiplier=0.8,
            extra_safety=0.15,
            prefer_damaten=True,
            reason="all-last 2nd, safe lead over 3rd"
        )

    if placement == 3:
        if diff_above <= 4000:
            # Can move up to 2nd with small hand
            return PlacementAdjustment(
                riichi_multiplier=1.0,
                meld_multiplier=1.1,
                extra_safety=0.0,
                reason="all-last 3rd, close to 2nd"
            )
        # Standard play, avoid dropping to 4th
        return PlacementAdjustment(
            riichi_multiplier=0.9,
            meld_multiplier=1.0,
            extra_safety=0.10,
            reason="all-last 3rd, steady play"
        )

    # placement == 4
    if diff_above <= 2000:
        # Very close to 3rd: any agari works
        return PlacementAdjustment(
            riichi_multiplier=1.0,
            meld_multiplier=1.3,  # encourage fast melds
            extra_safety=-0.10,   # push harder
            reason="all-last 4th, very close"
        )
    if diff_above <= 8000:
        # Reachable with a decent hand
        return PlacementAdjustment(
            riichi_multiplier=1.2,
            meld_multiplier=1.1,
            extra_safety=-0.05,
            min_push_value=2000,
            reason="all-last 4th, reachable"
        )
    if diff_above <= 16000:
        # Need a big hand
        return PlacementAdjustment(
            riichi_multiplier=1.3,
            meld_multiplier=0.8,  # discourage cheap melds
            extra_safety=-0.10,
            min_push_value=5200,
            reason="all-last 4th, need big hand"
        )
    # Desperate: need mangan+ or dealer repeat
    if is_dealer:
        return PlacementAdjustment(
            riichi_multiplier=1.2,
            meld_multiplier=1.0,
            extra_safety=-0.15,
            reason="all-last 4th dealer, keep dealing"
        )
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
    """Strategy adjustments for south round (not all-last)."""

    if placement == 1:
        if diff_below >= 20000:
            # Huge lead: very conservative
            return PlacementAdjustment(
                riichi_multiplier=0.7,
                meld_multiplier=0.8,
                extra_safety=0.15,
                prefer_damaten=True,
                reason="south 1st, huge lead"
            )
        if diff_below >= 8000:
            return PlacementAdjustment(
                riichi_multiplier=0.85,
                meld_multiplier=0.9,
                extra_safety=0.08,
                reason="south 1st, comfortable lead"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.95,
            extra_safety=0.05,
            reason="south 1st, thin lead"
        )

    if placement == 4:
        if diff_above >= 30000:
            # Very far behind: need to take risks
            return PlacementAdjustment(
                riichi_multiplier=1.15,
                meld_multiplier=0.9,
                extra_safety=-0.10,
                reason="south 4th, far behind"
            )
        return PlacementAdjustment(
            riichi_multiplier=1.05,
            extra_safety=-0.05,
            reason="south 4th, moderate deficit"
        )

    # 2nd or 3rd: mostly standard
    return PlacementAdjustment(reason=f"south {placement}th, standard play")


def _east_strategy(gs: GameState, placement: int,
                    diff_above: int, diff_below: int,
                    is_dealer: bool) -> PlacementAdjustment:
    """Strategy for east round — mostly standard but with dealer awareness."""

    if is_dealer:
        # Dealers benefit more from winning: encourage slightly more aggression
        return PlacementAdjustment(
            riichi_multiplier=1.05,
            meld_multiplier=1.05,
            extra_safety=-0.03,
            reason="east dealer, slight aggression"
        )

    return PlacementAdjustment(reason="east round, standard play")


def should_damaten(gs: GameState, adj: PlacementAdjustment,
                   hand_value: float = 0.0) -> bool:
    """Whether to consider damaten (hidden tenpai) over riichi.

    Damaten is preferred when:
    - Already leading and riichi stick loss is costly relative to the lead
    - Hand is already very expensive (mangan+) and riichi stick is not needed
    - Thin lead where losing 1000 pts for riichi stick would drop placement

    Riichi is still preferred when:
    - Hand is cheap (1 han) and needs riichi + ura dora to be meaningful
    - Kyotaku on table makes winning more rewarding (offsets the riichi stick cost)
    - Leading comfortably and the riichi stick cost is negligible vs lead
    """
    if not adj.prefer_damaten:
        return False

    # === All-last 1st place ===
    if gs.is_all_last and gs.my_placement == 1:
        lead = gs.diff_to_below

        # Mangan+ hand: already expensive enough, damaten avoids riichi stick risk
        if hand_value >= 8000:
            return True

        # Very thin lead (riichi stick would erase it or flip placement)
        if lead <= 1000:
            return True

        # Small lead: damaten unless hand is cheap and needs riichi to matter,
        # or unless there's a big kyotaku making riichi stick cost worthwhile
        if lead <= 4000:
            kyotaku_bonus = gs.kyotaku * 1000
            if kyotaku_bonus >= 2000:
                return False  # Big pot: riichi stick cost offset by kyotaku reward
            # If hand is cheap (< 3000 pts), riichi is needed to make it count
            # If hand is decent (3000+), damaten is safer
            return hand_value >= 3000

        # Comfortable lead: damaten generally fine to protect it,
        # but don't damaten a super cheap hand that needs riichi to win at all
        if lead >= 12000:
            return True  # safe enough to always damaten

        # Between 4000 and 12000: damaten if kyotaku doesn't justify the risk
        kyotaku_bonus = gs.kyotaku * 1000
        # If winning already nets enough extra from kyotaku, riichi stick is less costly
        if kyotaku_bonus >= 2000:
            return False  # riichi: big pot on table makes it worth it
        return hand_value >= 3000

    # === Non all-last 1st with prefer_damaten flag ===
    if gs.my_placement == 1:
        if gs.diff_to_below <= 1000:
            return True  # Extremely thin lead
        # Otherwise let riichi through — damaten in mid-game costs too much tempo

    return False
