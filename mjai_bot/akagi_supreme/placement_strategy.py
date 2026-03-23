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
        # Noten penalty depends on how many players are tenpai/noten.
        # Use the actual worst-case effect from GameState rather than
        # a hardcoded 3000, which underestimates the risk.
        noten_penalty = abs(gs.noten_penalty_effect())
        noten_risk = diff_below <= noten_penalty
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
                    riichi_multiplier=0.8,
                    meld_multiplier=1.0,
                    prefer_damaten=True,
                    min_push_value=max(1000, direct_pts),
                    reason=f"all-last 2nd, direct hit on 1st ({han_direct}han), prefer damaten"
                )

        if han_for_1st_ron <= 3:
            return PlacementAdjustment(
                riichi_multiplier=1.1,
                meld_multiplier=0.9,
                min_push_value=max(2000, pts_for_1st),
                reason=f"all-last 2nd, 1st reachable ({han_for_1st_ron}han ron)"
            )
        if diff_below < 4000:
            # 3rd is close — riichi stick cost (1000pts) could narrow the gap
            # further, and riichi locks the hand. Damaten preserves flexibility
            # to dodge dangerous tiles and protect 2nd place.
            return PlacementAdjustment(
                riichi_multiplier=0.9,
                meld_multiplier=1.1,
                prefer_damaten=True,
                reason="all-last 2nd, 3rd is close - damaten to protect, meld for speed"
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
            # 4th is close: riichi stick (1000pt) directly risks dropping
            # to 4th (ラス). Top players prioritize ラス回避 above all else.
            # When gap < 2000, the 1000pt riichi deposit narrows it to < 1000,
            # making placement flip very likely on any mishap.
            if diff_below < 2000:
                return PlacementAdjustment(
                    riichi_multiplier=0.7,
                    meld_multiplier=1.1,
                    prefer_damaten=True,
                    reason="all-last 3rd, 4th dangerously close (<2000pts), protect against ラス"
                )
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
        if diff_below >= 8000 and han_for_2nd >= 4:
            return PlacementAdjustment(
                riichi_multiplier=0.8,
                meld_multiplier=1.3,
                prefer_damaten=True,
                reason="all-last 3rd, 4th far, 2nd far - fast agari to secure 3rd"
            )
        return PlacementAdjustment(
            riichi_multiplier=0.9,
            meld_multiplier=1.1,
            reason="all-last 3rd, steady play to secure placement"
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

    required_val = max(2000, int(pts_for_3rd * 0.8))

    if han_for_3rd_ron <= 1:
        return PlacementAdjustment(
            riichi_multiplier=1.0,
            meld_multiplier=1.3,
            min_push_value=required_val,
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
            min_push_value=required_val,
            reason="all-last 4th, need mangan"
        )
    return PlacementAdjustment(
        riichi_multiplier=1.1,
        meld_multiplier=0.7,
        min_push_value=required_val,
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
    """Strategy for east round — mostly standard but with dealer awareness.

    Top players value dealer position highly (renchan = extra scoring
    opportunity). Dealer should be noticeably more aggressive.
    """
    if is_dealer:
        return PlacementAdjustment(
            riichi_multiplier=1.1,
            meld_multiplier=1.1,
            reason="east dealer, aggression for renchan value"
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

    # === Opponent riichi: riichi for intimidation and ura dora ===
    # When opponents have declared riichi, our riichi adds:
    # 1. Intimidation: non-riichi opponents fold harder → fewer deal-ins to us
    # 2. Ura dora: free value since hand is already locked
    # 3. Ippatsu chance: opponents in riichi can't dodge
    # Top players prefer riichi over damaten when opponents are in riichi,
    # UNLESS we're all-last 1st protecting a lead (ending game > extra points),
    # or our hand is too cheap to justify the risk of oi-riichi.
    if gs.num_riichi_opponents >= 1:
        if not (gs.is_all_last and gs.my_placement == 1):
            # Exception: when riichi stick cost (1000pt) directly threatens
            # placement, top players damaten to protect. Paying 1000pt when
            # the gap to the player below is ≤1000 can flip placement.
            # ラス回避 (avoid last) is the #1 priority in competitive mahjong.
            if (gs.is_all_last and gs.my_placement in (2, 3)
                    and gs.diff_to_below <= 1000):
                pass  # Let damaten logic continue; placement protection > chase riichi
            else:
                # トッププレイヤーは対リーチ時、PUSHと判断された以上は攻めるべき。
                # 悪形安手でもリーチで威嚇して周りを降ろすのが正しい判断。
                # ダマテンで構えても悪形では出ない/ツモれないので、リーチの+1翻と
                # 威嚇効果（他家ベタオリ誘発）の方が期待値が高い。
                return False  # 追っかけリーチ (oi-riichi) — always riichi vs riichi

    # === Very late game: damaten loses value ===
    # BUGFIX: In very late game (turn 14+), riichi is bad because of the 1000pt risk
    # and few draws left. Top players prefer damaten to maintain flexibility and save points.
    # Exceptions:
    # - All-last 1st: handled by later logic (protect lead vs riichi stick cost)
    # - All-last 4th: riichi for +1 han; 1000pt cost is irrelevant when already
    #   last, and damaten's "flexibility" is worthless with so few draws
    if my_turn >= 14 and not (gs.is_all_last and gs.my_placement in (1, 4)):
        return True

    # === Tile-level mountain analysis (山残り) ===
    if total_remaining > 0:
        # Single wait kind with only 1 copy left: extremely low tsumo expectations.
        # Top players usually avoid committing 1000 points to a dead wait.
        # Prefer damaten to keep fold options open and avoid point loss.
        if num_wait_kinds == 1 and max_single_tile <= 1:
            if not (gs.is_all_last and gs.my_placement == 1):
                return True

        # Multiple wait kinds but total <= 2: very thin overall.
        # Top players do not riichi a nearly dead wait to avoid getting locked into a bad situation.
        # EXCEPTION: All-last 2nd/3rd — with ultra-thin wait, riichi intimidation is
        # more valuable than damaten because tsumo chance is near zero anyway.
        # Let the bad_wait section below handle placement-specific logic.
        if total_remaining <= 2:
            if not (gs.is_all_last and gs.my_placement == 1):
                if not (gs.is_all_last and gs.my_placement in (2, 3)):
                    return True

            # Good concentration on one tile or wide wait — damaten strength
            # is already handled by the general control flow below.
            # (No special action needed here; continue to placement checks.)

    # === Bad wait shape: riichi adds value via ura dora and intimidation ===
    # Exception: all-last 1st where riichi deposit (1000pts) threatens our lead.
    # Exception 2: South round 1st place with a lead. Damaten preserves safety.
    if bad_wait and hand_value < 8000:
        if gs.is_all_last and gs.my_placement == 1 and gs.diff_to_below <= abs(gs.noten_penalty_effect()):
            return True
        if gs.is_south and gs.my_placement == 1 and gs.diff_to_below >= 4000:
            return True
        # All-last 2nd/3rd: placement protection overrides bad-wait riichi preference.
        # Riichi stick cost (1000pts) can flip placement, so damaten despite bad wait.
        # EXCEPTION: extremely thin wait (≤2 tiles) — riichi intimidation is more
        # valuable because tsumo chance is near zero anyway.
        if total_remaining >= 3:
            if gs.is_all_last and gs.my_placement == 2 and gs.diff_to_below < 4000:
                return True
            if gs.is_all_last and gs.my_placement == 3 and gs.diff_to_below < 2000:
                return True
            # All-last 3rd: if climbing to 2nd requires low han and gap is small,
            # damaten to maximize chance of out-agari even with bad wait
            if gs.is_all_last and gs.my_placement == 3 and gs.diff_to_above <= 4000:
                pts_for_2nd = gs.points_needed_for_placement(2)
                han_for_2nd = gs.min_han_for_points(pts_for_2nd, is_tsumo=False)
                if han_for_2nd <= 2:
                    return True
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
                return True
            # With good wait (tile-level): damaten to protect lead
            if total_remaining >= 6 and num_wait_kinds >= 2:
                return True
            return hand_value >= 3000

        if lead >= 12000:
            return True

        kyotaku_bonus = gs.kyotaku * 1000
        if kyotaku_bonus >= 2000:
            return True
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

    # === All-last 2nd place: protect against 3rd ===
    # When 3rd is close, riichi stick (1000pts) narrows the gap dangerously.
    # Top players prefer damaten to preserve the safety margin.
    if gs.is_all_last and gs.my_placement == 2:
        lead_to_3rd = gs.diff_to_below
        if lead_to_3rd < 4000:
            # Thin lead: damaten to avoid 1000pt riichi stick loss
            return True
        pts_for_1st = gs.points_needed_for_placement(1)
        han_direct = 99
        for i, p in enumerate(gs.players):
            if i != gs.player_id and p.score >= max(pp.score for pp in gs.players):
                dp = gs.points_needed_direct_hit(i, 1)
                han_direct = gs.min_han_for_points(dp, is_tsumo=False)
                break
        # Direct hit 1st with low han: damaten for surprise (hide tenpai)
        if han_direct <= 2 and hand_value >= 2000:
            return True

    # === All-last 3rd place: speed agari to escape 4th ===
    # When 2nd is close, damaten for fast out-agari is viable.
    # Top players damaten when the point condition is simple (1-2 han enough).
    if gs.is_all_last and gs.my_placement == 3:
        pts_for_2nd = gs.points_needed_for_placement(2)
        han_for_2nd = gs.min_han_for_points(pts_for_2nd, is_tsumo=False)
        if han_for_2nd <= 2 and gs.diff_to_above <= 4000:
            return True
        # 4th is close: damaten to avoid riichi stick narrowing the gap
        if gs.diff_to_below < 2000:
            return True

    # === Any placement: haneman+ closed hand (almost always damaten) ===
    if hand_value >= 12000 and not gs.my_info.is_open():
        return True

    # === Any placement: mangan closed hand ===
    if hand_value >= 8000 and not gs.my_info.is_open():
        # Top players damaten Mangan if the wait is bad (avoid 1000pt risk / preserve flexibility)
        if bad_wait:
            return True
        # If good wait, damaten is still strong to ensure win (minimize opponents' fold rate)
        # especially when we are 1st or 2nd and want to protect our position without risking 1000pts.
        if not bad_wait and gs.my_placement <= 2:
            return True

    return False
