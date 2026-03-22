# -*- coding: utf-8 -*-
"""Push/fold decision engine.

Determines whether the bot should push (attack), fold (defend),
or play a balanced strategy based on game context.

Key factors:
- Shanten count (distance to tenpai)
- Opponent threat level (riichi, open hands, behavioral patterns)
- Hand value potential (dora, yakuhai, suit composition)
- Placement and point situation
- Turn number (not just remaining tiles)
- Acceptance count (how live is the hand)

Design principle: Trust Mortal's Q-values as the primary signal.
Only intervene when strategic context clearly demands it.
Top players almost never fold from tenpai, and push aggressively
from iishanten with decent hands.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .game_state import (
    GameState, Tile, parse_tile, is_honor, tile_base,
    YAKUHAI_HONORS, SUITS, HONORS,
)


class Decision(Enum):
    PUSH = 0       # Full attack - trust Mortal's Q-values
    BALANCED = 1   # Slight safety bias on dangerous tiles
    CAUTIOUS = 2   # Moderate safety adjustment
    FOLD = 3       # Maximum defense - prioritize safe tiles


@dataclass
class PushFoldResult:
    decision: Decision
    confidence: float  # 0.0 to 1.0
    reason: str
    # How much to weight safety vs Q-values (0.0 = pure Q-value, 1.0 = pure safety)
    safety_weight: float


def estimate_hand_value(gs: GameState) -> float:
    """Estimate potential hand value in points.

    Enhanced with:
    - Yakuhai detection (round wind, seat wind, sangenpai)
    - Suit composition analysis (honitsu/chinitsu potential)
    - Better dora counting (no double-count)
    - Menzen bonus
    - Kyotaku and honba
    """
    if gs.is_dealer_me:
        base = 3900.0  # dealer 1 han 30 fu
    else:
        base = 2000.0  # non-dealer 1 han 30 fu

    han_estimate = 0.0

    # === Dora count ===
    dora_count = gs.count_dora_in_hand()
    han_estimate += dora_count

    # === Yakuhai in hand ===
    hand = gs.my_hand
    for tile in set(hand):
        if gs.is_my_yakuhai(tile):
            count = sum(1 for t in hand if tile_base(t) == tile)
            if count >= 3:
                han_estimate += 1.0  # confirmed yakuhai
            elif count >= 2:
                han_estimate += 0.4  # possible pon

    # === Suit composition: honitsu/chinitsu potential ===
    suit_counts = {"m": 0, "p": 0, "s": 0, "z": 0}
    for t in hand:
        if t in HONORS:
            suit_counts["z"] += 1
        else:
            s, _, _ = parse_tile(t)
            if s in suit_counts:
                suit_counts[s] += 1
    total_number = suit_counts["m"] + suit_counts["p"] + suit_counts["s"]
    if total_number > 0:
        dominant_suit_count = max(suit_counts["m"], suit_counts["p"], suit_counts["s"])
        # Honitsu: one suit + honors
        if dominant_suit_count + suit_counts["z"] >= len(hand) - 1:
            han_estimate += 1.5  # likely honitsu (2 han open, 3 closed)
        elif dominant_suit_count + suit_counts["z"] >= len(hand) - 2:
            han_estimate += 0.5  # possible honitsu

    # === Riichi potential for closed hand ===
    if not gs.my_info.is_open():
        han_estimate += 1.0  # riichi
        han_estimate += 0.3  # ippatsu/ura dora average contribution

    # === Convert han estimate to points ===
    # Rough point table (30fu baseline)
    if han_estimate >= 11:
        value = 32000.0 if gs.is_dealer_me else 24000.0  # sanbaiman
    elif han_estimate >= 8:
        value = 24000.0 if gs.is_dealer_me else 16000.0  # baiman
    elif han_estimate >= 6:
        value = 18000.0 if gs.is_dealer_me else 12000.0  # haneman
    elif han_estimate >= 5:
        value = 12000.0 if gs.is_dealer_me else 8000.0  # mangan
    elif han_estimate >= 4:
        value = 11600.0 if gs.is_dealer_me else 7700.0
    elif han_estimate >= 3:
        value = 5800.0 if gs.is_dealer_me else 3900.0
    elif han_estimate >= 2:
        value = 2900.0 if gs.is_dealer_me else 2000.0
    else:
        value = base

    # Kyotaku on table: winning gains all accumulated riichi sticks
    value += gs.kyotaku * 1000

    # Honba bonus
    value += gs.honba * 300

    return value


def estimate_risk_of_deal_in(gs: GameState) -> float:
    """Estimate expected cost if we deal into an opponent.

    Enhanced with turn-aware risk and behavioral threat reading.
    """
    base_risk = 5200.0  # average deal-in cost

    # Riichi opponents
    n_riichi = gs.num_riichi_opponents
    if n_riichi >= 2:
        base_risk *= 1.5  # double riichi
    elif n_riichi == 1:
        for i, p in enumerate(gs.players):
            if i != gs.player_id and p.riichi_declared:
                if p.riichi_turn <= 6:
                    base_risk *= 1.3  # early riichi = likely good hand
                if p.is_dealer:
                    base_risk *= 1.5  # dealer riichi is expensive

    # Open hands with strong threat signals
    for i, p in enumerate(gs.players):
        if i == gs.player_id:
            continue
        if p.is_open():
            threat = p.apparent_threat_level(gs.turn)
            if threat >= 1.5:
                # High-threat open hand (multiple yakuhai, honitsu, toitoi)
                base_risk = max(base_risk, 8000 * (1 + (threat - 1.5) * 0.3))
            elif threat >= 0.8:
                base_risk = max(base_risk, 5200 * (1 + threat * 0.3))

    # Dama tenpai signal from tedashi patterns (non-riichi threat)
    for i, p in enumerate(gs.players):
        if i == gs.player_id or p.riichi_declared:
            continue
        if p.tedashi_after_tsumogiri_streak():
            base_risk = max(base_risk, 5200)

    # Honba bonus
    base_risk += gs.honba * 300

    return base_risk


def evaluate_push_fold(gs: GameState, shanten: int,
                       acceptance_count: int = 0) -> PushFoldResult:
    """Main push/fold evaluation.

    Design: Top players almost never fold from tenpai or good iishanten.
    We use turn number as a continuous variable, not just remaining tiles.
    acceptance_count: number of useful remaining tiles (from estimate_acceptance_count).
    """
    hand_value = estimate_hand_value(gs)
    risk = estimate_risk_of_deal_in(gs)
    threat = gs.max_opponent_threat()
    my_turn = gs.my_turn  # per-player turn (0-based)

    # Classify shape quality based on acceptance count
    # Good shape: >= 8 useful tiles, Bad shape: <= 4
    good_shape = acceptance_count >= 8
    bad_shape = acceptance_count > 0 and acceptance_count <= 4

    # === Tenpai: almost always push ===
    # Top players virtually never fold from tenpai.
    # But bad-shape tenpai (e.g. penchan 4 tiles) is less pushable than
    # good-shape tenpai (e.g. ryanmen 8 tiles) in extreme situations.
    if shanten <= 0:
        if threat >= 3.5 and hand_value < 2000 and my_turn >= 12 and bad_shape:
            # Extreme case: triple riichi, cheap hand, very late, bad wait
            return PushFoldResult(
                Decision.BALANCED, 0.6,
                "tenpai but cheap bad-shape vs extreme threat, very late",
                safety_weight=0.10
            )
        if threat >= 2.5 and hand_value < 2000 and my_turn >= 14 and bad_shape:
            # Very late, cheap bad-shape vs strong threats
            return PushFoldResult(
                Decision.BALANCED, 0.55,
                "tenpai cheap bad-shape, very late, strong threats",
                safety_weight=0.06
            )
        # Standard: push from tenpai
        return PushFoldResult(
            Decision.PUSH, 0.95,
            "tenpai - push",
            safety_weight=0.0
        )

    # === Iishanten (1-away): mostly push, context-dependent ===
    if shanten == 1:
        if threat <= 0.5:
            return PushFoldResult(
                Decision.PUSH, 0.85,
                "iishanten, low threat",
                safety_weight=0.03
            )
        # Turn-aware: early iishanten is much more pushable
        if my_turn <= 8:
            # Still early/mid game
            if hand_value >= risk * 0.4:
                return PushFoldResult(
                    Decision.PUSH, 0.75,
                    "iishanten, early-mid game, decent value",
                    safety_weight=0.05
                )
            if threat >= 1.5:
                sw = 0.08 if good_shape else 0.12
                return PushFoldResult(
                    Decision.BALANCED, 0.7,
                    "iishanten, riichi opponent, early-mid",
                    safety_weight=sw
                )
            return PushFoldResult(
                Decision.PUSH, 0.7,
                "iishanten, early-mid game",
                safety_weight=0.05
            )
        else:
            # Late game iishanten
            if hand_value >= risk * 0.6:
                sw = 0.08 if good_shape else 0.15
                return PushFoldResult(
                    Decision.BALANCED, 0.65,
                    "iishanten, late but valuable hand",
                    safety_weight=sw
                )
            if threat >= 2.0:
                sw = 0.15 if good_shape else 0.22
                return PushFoldResult(
                    Decision.BALANCED, 0.6,
                    "iishanten, late, multiple threats",
                    safety_weight=sw
                )
            if threat >= 1.0:
                sw = 0.12 if good_shape else 0.18
                return PushFoldResult(
                    Decision.BALANCED, 0.65,
                    "iishanten, late, riichi opponent",
                    safety_weight=sw
                )
            return PushFoldResult(
                Decision.BALANCED, 0.65,
                "iishanten, late game",
                safety_weight=0.10
            )

    # === Ryanshanten (2-away): turn-aware ===
    if shanten == 2:
        if my_turn <= 6:
            # Early game: still developing, light defense
            if threat <= 0.5:
                return PushFoldResult(
                    Decision.PUSH, 0.6,
                    "ryanshanten, early round, low threat",
                    safety_weight=0.05
                )
            if threat >= 1.5:
                return PushFoldResult(
                    Decision.BALANCED, 0.6,
                    "ryanshanten, early, riichi opponent",
                    safety_weight=0.15
                )
            return PushFoldResult(
                Decision.BALANCED, 0.6,
                "ryanshanten, early round",
                safety_weight=0.10
            )
        elif my_turn <= 12:
            # Mid game
            if threat >= 2.0:
                return PushFoldResult(
                    Decision.CAUTIOUS, 0.65,
                    "ryanshanten, mid-game, multiple threats",
                    safety_weight=0.30
                )
            if threat >= 1.0:
                if hand_value >= risk * 0.5 and good_shape:
                    return PushFoldResult(
                        Decision.BALANCED, 0.6,
                        "ryanshanten, mid-game, riichi but valuable+connected",
                        safety_weight=0.18
                    )
                if hand_value >= risk * 0.5:
                    return PushFoldResult(
                        Decision.BALANCED, 0.6,
                        "ryanshanten, mid-game, riichi but valuable",
                        safety_weight=0.20
                    )
                return PushFoldResult(
                    Decision.CAUTIOUS, 0.6,
                    "ryanshanten, mid-game, riichi",
                    safety_weight=0.25
                )
            return PushFoldResult(
                Decision.BALANCED, 0.55,
                "ryanshanten, mid-game, no threat",
                safety_weight=0.10
            )
        else:
            # Late game ryanshanten: mostly defensive
            if threat >= 1.5:
                return PushFoldResult(
                    Decision.CAUTIOUS, 0.7,
                    "ryanshanten, late, threats present",
                    safety_weight=0.40
                )
            if threat >= 0.5:
                return PushFoldResult(
                    Decision.CAUTIOUS, 0.6,
                    "ryanshanten, late game",
                    safety_weight=0.30
                )
            return PushFoldResult(
                Decision.BALANCED, 0.55,
                "ryanshanten, late, no threat",
                safety_weight=0.18
            )

    # === 3+ away: heavily defensive, but turn-aware ===
    if my_turn <= 6 and threat <= 0.5:
        # Very early, no threat: can still develop
        return PushFoldResult(
            Decision.BALANCED, 0.5,
            f"shanten={shanten}, very early, no threat",
            safety_weight=0.15
        )
    if threat >= 2.0:
        return PushFoldResult(
            Decision.FOLD, 0.85,
            f"shanten={shanten} vs multiple threats, full fold",
            safety_weight=0.65
        )
    if threat >= 1.0:
        return PushFoldResult(
            Decision.CAUTIOUS, 0.75,
            f"shanten={shanten} with riichi opponent",
            safety_weight=0.50
        )
    if my_turn >= 12:
        return PushFoldResult(
            Decision.CAUTIOUS, 0.65,
            f"shanten={shanten}, very late",
            safety_weight=0.45
        )
    return PushFoldResult(
        Decision.CAUTIOUS, 0.55,
        f"shanten={shanten}, cautious play",
        safety_weight=0.30
    )


def adjust_for_placement(result: PushFoldResult, gs: GameState) -> PushFoldResult:
    """Adjust push/fold based on placement context.

    - 4th place with big deficit: push harder (need to recover)
    - 1st place with big lead: play safer (protect lead)
    - All last special handling
    """
    placement = gs.my_placement
    diff_above = gs.diff_to_above
    diff_below = gs.diff_to_below

    # === All Last (オーラス) special logic ===
    if gs.is_all_last:
        if placement == 1:
            more_defensive = result.decision if result.decision.value >= Decision.CAUTIOUS.value else Decision.CAUTIOUS
            return PushFoldResult(
                more_defensive,
                result.confidence,
                f"all-last 1st place: {result.reason}",
                safety_weight=max(result.safety_weight, 0.35)
            )
        if placement == 4:
            if diff_above <= 8000:
                return PushFoldResult(
                    Decision.PUSH,
                    0.8,
                    "all-last 4th, reachable deficit",
                    safety_weight=max(0.0, result.safety_weight - 0.20)
                )
            else:
                return PushFoldResult(
                    Decision.BALANCED,
                    0.6,
                    "all-last 4th, large deficit",
                    safety_weight=max(0.0, result.safety_weight - 0.10)
                )

    # === South round general adjustments ===
    if gs.is_south:
        if placement == 1 and diff_below >= 12000:
            return PushFoldResult(
                result.decision,
                result.confidence,
                f"south, leading comfortably: {result.reason}",
                safety_weight=min(1.0, result.safety_weight + 0.08)
            )
        if placement == 4 and diff_above >= 20000:
            return PushFoldResult(
                result.decision,
                result.confidence,
                f"south 4th, desperate: {result.reason}",
                safety_weight=max(0.0, result.safety_weight - 0.12)
            )

    # === General placement adjustments (small) ===
    if placement == 1:
        result.safety_weight = min(1.0, result.safety_weight + 0.03)
    elif placement == 4:
        result.safety_weight = max(0.0, result.safety_weight - 0.03)

    return result
