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
- Wait tile remaining count (山残り) for damaten decisions
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
    kyotaku_bonus = gs.kyotaku * 1000

    if placement == 1:
        noten_risk = diff_below <= 3000
        if noten_risk:
            return PlacementAdjustment(
                riichi_multiplier=0.8,
                meld_multiplier=0.9,
                prefer_damaten=True,
                reason="all-last 1st, thin lead, noten penalty risk"
            )
        if diff_below >= 12000:
            return PlacementAdjustment(
                riichi_multiplier=0.5,
                meld_multiplier=0.7,
                prefer_damaten=True,
                reason="all-last 1st, big lead - protect"
            )
        if diff_below >= 4000:
            return PlacementAdjustment(
                riichi_multiplier=0.7,
                meld_multiplier=0.8,
                prefer_damaten=True,
                reason="all-last 1st, moderate lead"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.85,
            meld_multiplier=0.9,
            prefer_damaten=True,
            reason="all-last 1st, thin lead"
        )

    if placement == 2:
        pts_for_1st = gs.points_needed_for_placement(1)
        han_for_1st_ron = gs.min_han_for_points(pts_for_1st, is_tsumo=False)
        han_for_1st_tsumo = gs.min_han_for_points(pts_for_1st, is_tsumo=True)

        first_seat = None
        for i, p in enumerate(gs.players):
            if i != gs.player_id and p.score >= max(pp.score for pp in gs.players):
                first_seat = i
                break
        if first_seat is not None:
            direct_pts = gs.points_needed_direct_hit(first_seat, 1)
            han_direct = gs.min_han_for_points(direct_pts, is_tsumo=False)
            if han_direct < han_for_1st_ron and han_direct <= 2:
                return PlacementAdjustment(
                    riichi_multiplier=1.1,
                    meld_multiplier=1.0,
                    min_push_value=max(1000, direct_pts),
                    reason=f"all-last 2nd, direct hit on 1st ({han_direct}han)"
                )

        if han_for_1st_ron <= 3:
            return PlacementAdjustment(
                riichi_multiplier=1.1,
                meld_multiplier=0.9,
                min_push_value=max(2000, pts_for_1st),
                reason=f"all-last 2nd, 1st reachable ({han_for_1st_ron}han ron)"
            )
        if diff_below < 4000:
            return PlacementAdjustment(
                riichi_multiplier=1.0,
                meld_multiplier=1.0,
                prefer_damaten=False,
                reason="all-last 2nd, 3rd is close - push"
            )
        if diff_below < 8000:
            return PlacementAdjustment(
                riichi_multiplier=0.9,
                meld_multiplier=0.9,
                prefer_damaten=True,
                reason="all-last 2nd, moderate gap to 3rd"
            )
        if han_for_1st_tsumo <= 4:
            return PlacementAdjustment(
                riichi_multiplier=0.95,
                meld_multiplier=0.85,
                prefer_damaten=False,
                reason="all-last 2nd, safe over 3rd, 1st reachable by tsumo"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.8,
            meld_multiplier=0.8,
            prefer_damaten=True,
            reason="all-last 2nd, safe lead over 3rd"
        )

    if placement == 3:
        pts_for_2nd = gs.points_needed_for_placement(2)
        han_for_2nd = gs.min_han_for_points(pts_for_2nd, is_tsumo=False)

        if diff_below < 4000:
            if han_for_2nd <= 2:
                return PlacementAdjustment(
                    riichi_multiplier=1.0,
                    meld_multiplier=1.1,
                    reason="all-last 3rd, 2nd close, 4th near - fast agari"
                )
            return PlacementAdjustment(
                riichi_multiplier=0.9,
                meld_multiplier=1.0,
                reason="all-last 3rd, 4th near - careful"
            )
        if han_for_2nd <= 2:
            return PlacementAdjustment(
                riichi_multiplier=1.0,
                meld_multiplier=1.1,
                reason="all-last 3rd, close to 2nd"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.9,
            meld_multiplier=1.0,
            reason="all-last 3rd, steady play"
        )

    # placement == 4
    pts_for_3rd = gs.points_needed_for_placement(3)
    han_for_3rd_ron = gs.min_han_for_points(pts_for_3rd, is_tsumo=False)
    han_for_3rd_tsumo = gs.min_han_for_points(pts_for_3rd, is_tsumo=True)

    if is_dealer:
        return PlacementAdjustment(
            riichi_multiplier=1.1,
            meld_multiplier=1.2,
            reason="all-last 4th dealer, any agari for renchan"
        )

    if han_for_3rd_ron <= 1:
        return PlacementAdjustment(
            riichi_multiplier=1.0,
            meld_multiplier=1.3,
            reason=f"all-last 4th, very close ({pts_for_3rd}pts needed)"
        )
    if han_for_3rd_tsumo <= 3:
        return PlacementAdjustment(
            riichi_multiplier=1.2,
            meld_multiplier=1.1,
            min_push_value=max(2000, pts_for_3rd),
            reason=f"all-last 4th, reachable ({han_for_3rd_tsumo}han tsumo)"
        )
    if han_for_3rd_ron <= 5:
        return PlacementAdjustment(
            riichi_multiplier=1.3,
            meld_multiplier=0.8,
            min_push_value=5200,
            reason="all-last 4th, need mangan"
        )
    return PlacementAdjustment(
        riichi_multiplier=1.1,
        meld_multiplier=0.7,
        min_push_value=7700,
        reason="all-last 4th, desperate"
    )


def _south_strategy(gs: GameState, placement: int,
                     diff_above: int, diff_below: int,
                     is_dealer: bool) -> PlacementAdjustment:
    """Strategy adjustments for south round (not all-last)."""
    if placement == 1:
        if diff_below >= 20000:
            return PlacementAdjustment(
                riichi_multiplier=0.7,
                meld_multiplier=0.8,
                prefer_damaten=True,
                reason="south 1st, huge lead"
            )
        if diff_below >= 8000:
            return PlacementAdjustment(
                riichi_multiplier=0.85,
                meld_multiplier=0.9,
                prefer_damaten=diff_below <= 12000,
                reason="south 1st, comfortable lead"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.95,
            reason="south 1st, thin lead"
        )

    if placement == 4:
        if diff_above >= 30000:
            return PlacementAdjustment(
                riichi_multiplier=1.15,
                meld_multiplier=0.9,
                reason="south 4th, far behind"
            )
        if is_dealer:
            return PlacementAdjustment(
                riichi_multiplier=1.1,
                meld_multiplier=1.1,
                reason="south 4th dealer, renchan value"
            )
        return PlacementAdjustment(
            riichi_multiplier=1.05,
            reason="south 4th, moderate deficit"
        )

    if placement == 2 and diff_below < 4000:
        return PlacementAdjustment(
            reason="south 2nd, 3rd is close"
        )
    if placement == 3 and diff_above < 4000:
        return PlacementAdjustment(
            riichi_multiplier=1.05,
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
            reason="east dealer, slight aggression"
        )

    return PlacementAdjustment(reason="east round, standard play")


def should_damaten(gs: GameState, adj: PlacementAdjustment,
                   hand_value: float = 0.0,
                   acceptance_count: int = 0) -> bool:
    """Whether to consider damaten (hidden tenpai) over riichi.

    Enhanced with tile-level wait analysis (山残り):
    - Per-tile unseen counts determine wait quality more precisely
    - Single-tile waits with 1 remaining copy → riichi for intimidation
    - Multi-tile waits with good mountain → damaten viable
    - Concentration ratio: if most acceptance is on 1 tile, it's fragile

    Top player thinking on riichi vs damaten:
    - Riichi: +1han, ura dora chance, intimidation, but costs 1000pts and locks hand
    - Damaten: flexibility, can change wait, no information leak, can dodge
    - Mountain remaining (山残り) is critical: few tiles left = riichi for intimidation
    """
    if not adj.prefer_damaten:
        return False

    my_turn = gs.my_turn

    # === Get tile-level wait information ===
    wait_details = gs.wait_tile_details()
    num_wait_kinds = len(wait_details)  # how many different tiles we're waiting on
    max_single_tile = max((cnt for _, cnt in wait_details), default=0) if wait_details else 0
    total_remaining = sum(cnt for _, cnt in wait_details) if wait_details else acceptance_count

    # Use tile-level data if available, fall back to acceptance_count
    if total_remaining == 0 and acceptance_count > 0:
        total_remaining = acceptance_count

    bad_wait = total_remaining > 0 and total_remaining <= 4

    # === Very late game: damaten loses value ===
    if my_turn >= 14 and not (gs.is_all_last and gs.my_placement == 1):
        return False

    # === Tile-level mountain analysis (山残り) ===
    if total_remaining > 0:
        # Single wait kind with only 1 copy left: riichi for intimidation
        # (can't realistically tsumo, so threaten opponents into folding)
        if num_wait_kinds == 1 and max_single_tile <= 1:
            if not (gs.is_all_last and gs.my_placement == 1):
                return False

        # Multiple wait kinds but total <= 2: very thin overall
        if total_remaining <= 2 and num_wait_kinds <= 2:
            if not (gs.is_all_last and gs.my_placement == 1):
                return False

        # Concentrated wait: if one tile has 3+ copies but others have 0-1,
        # the wait looks wide but is actually fragile (opponent might hold it)
        if num_wait_kinds >= 2 and max_single_tile >= 3:
            # Good concentration — most copies on one tile, damaten is viable
            # since we have a realistic tsumo target
            pass
        elif num_wait_kinds >= 2 and total_remaining >= 6:
            # Wide wait with decent mountain — damaten is strong
            pass

    # === Bad wait shape: riichi adds value via ura dora and intimidation ===
    if bad_wait and hand_value < 8000:
        if not (gs.is_all_last and gs.my_placement == 1 and gs.diff_to_below <= 1000):
            return False

    # === All-last 1st place ===
    if gs.is_all_last and gs.my_placement == 1:
        lead = gs.diff_to_below

        if hand_value >= 8000:
            return True

        # Very thin lead: riichi stick cost could flip placement
        if lead <= 1000:
            return True

        if lead <= 4000:
            kyotaku_bonus = gs.kyotaku * 1000
            if kyotaku_bonus >= 2000:
                return False
            # With good wait (tile-level): damaten to protect lead
            if total_remaining >= 6 and num_wait_kinds >= 2:
                return True
            return hand_value >= 3000

        if lead >= 12000:
            return True

        kyotaku_bonus = gs.kyotaku * 1000
        if kyotaku_bonus >= 2000:
            return False
        # Good mountain = damaten viable even with moderate hand value
        if total_remaining >= 8 and hand_value >= 2000:
            return True
        return hand_value >= 3000

    # === South round 1st place with prefer_damaten ===
    if gs.is_south and gs.my_placement == 1:
        lead = gs.diff_to_below
        if lead <= 1000:
            return True
        if hand_value >= 8000 and lead <= 8000:
            return True
        # Good wide wait — damaten to maintain lead
        if total_remaining >= 8 and num_wait_kinds >= 2 and lead >= 4000:
            return True

    # === Any placement: haneman+ closed hand with GOOD wait ===
    if hand_value >= 12000 and not gs.my_info.is_open() and not bad_wait:
        return True

    return False
