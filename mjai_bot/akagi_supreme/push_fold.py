# -*- coding: utf-8 -*-
"""Push/fold decision engine.

Determines whether the bot should push (attack), fold (defend),
or play a balanced strategy based on game context.

Key factors:
- Shanten count (distance to tenpai)
- Opponent threat level (riichi, open hands)
- Hand value potential (dora, yakuhai)
- Placement and point situation
- Remaining tiles in wall
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .game_state import GameState, Tile, parse_tile, is_honor, YAKUHAI_HONORS


class Decision(Enum):
    PUSH = "push"          # Full attack - trust Mortal's Q-values
    BALANCED = "balanced"  # Slight safety bias on dangerous tiles
    CAUTIOUS = "cautious"  # Moderate safety adjustment
    FOLD = "fold"          # Maximum defense - prioritize safe tiles


@dataclass
class PushFoldResult:
    decision: Decision
    confidence: float  # 0.0 to 1.0
    reason: str
    # How much to weight safety vs Q-values (0.0 = pure Q-value, 1.0 = pure safety)
    safety_weight: float


def estimate_hand_value(gs: GameState) -> float:
    """Estimate potential hand value in points (rough).

    Returns a multiplier representing hand quality.
    Higher = more valuable hand worth pushing for.
    """
    value = 1000.0  # base hand value (1 han 30 fu)

    # Dora count
    dora_count = gs.count_dora_in_hand()
    if dora_count >= 3:
        value *= 4.0
    elif dora_count >= 2:
        value *= 2.5
    elif dora_count >= 1:
        value *= 1.5

    # Yakuhai in hand (potential pon or head)
    hand_tiles = gs.my_hand
    for tile in hand_tiles:
        if tile in YAKUHAI_HONORS:
            count = sum(1 for t in hand_tiles if t == tile)
            if count >= 3:
                value *= 1.5
            elif count >= 2:
                value *= 1.2

    # Riichi potential (menzen)
    if not gs.my_info.is_open():
        value *= 1.3  # riichi bonus potential

    # Dealer bonus
    if gs.is_dealer_me:
        value *= 1.5

    return value


def estimate_risk_of_deal_in(gs: GameState) -> float:
    """Estimate expected cost if we deal into an opponent.

    Returns estimated point loss.
    """
    base_risk = 5200.0  # average deal-in cost

    # Riichi opponents
    n_riichi = gs.num_riichi_opponents
    if n_riichi >= 2:
        base_risk *= 1.4  # more likely to hit someone
    elif n_riichi == 1:
        # Early riichi tends to be higher value
        for i, p in enumerate(gs.players):
            if i != gs.player_id and p.riichi_declared:
                if p.riichi_turn <= 6:
                    base_risk *= 1.3  # early riichi = likely good hand
                if p.is_dealer:
                    base_risk *= 1.5  # dealer riichi is expensive

    # Open hands with yakuhai
    for i, p in enumerate(gs.players):
        if i == gs.player_id:
            continue
        if p.is_open():
            threat = p.apparent_threat_level()
            if threat > 0.5:
                base_risk = max(base_risk, 3900 * (1 + threat * 0.5))

    # Honba bonus
    base_risk += gs.honba * 300

    return base_risk


def evaluate_push_fold(gs: GameState, shanten: int) -> PushFoldResult:
    """Main push/fold evaluation.

    Args:
        gs: Current game state
        shanten: Current shanten count (-1 = tenpai or agari, 0 = tenpai, 1+ = iishanten etc.)

    Returns:
        PushFoldResult with decision and safety weight
    """
    hand_value = estimate_hand_value(gs)
    risk = estimate_risk_of_deal_in(gs)
    threat = gs.max_opponent_threat()
    placement = gs.my_placement

    # === Tenpai: almost always push ===
    if shanten <= 0:
        if threat >= 2.5 and hand_value < 3000:
            # Tenpai but cheap hand vs very dangerous opponent
            return PushFoldResult(
                Decision.CAUTIOUS, 0.6,
                "tenpai but cheap hand vs high threat",
                safety_weight=0.25
            )
        return PushFoldResult(
            Decision.PUSH, 0.9,
            "tenpai - push",
            safety_weight=0.0
        )

    # === Iishanten (1-away): context-dependent ===
    if shanten == 1:
        if threat <= 0.5:
            # No serious threat
            return PushFoldResult(
                Decision.PUSH, 0.8,
                "iishanten, low threat",
                safety_weight=0.05
            )
        if hand_value >= risk * 0.6:
            # Good hand value relative to risk
            return PushFoldResult(
                Decision.BALANCED, 0.7,
                "iishanten, decent value vs risk",
                safety_weight=0.15
            )
        if threat >= 1.5:
            return PushFoldResult(
                Decision.CAUTIOUS, 0.65,
                "iishanten but high threat",
                safety_weight=0.35
            )
        return PushFoldResult(
            Decision.BALANCED, 0.7,
            "iishanten, moderate situation",
            safety_weight=0.20
        )

    # === Ryanshanten (2-away): lean defensive ===
    if shanten == 2:
        if threat <= 0.3 and gs.remaining_tiles > 40:
            # Early in round, no threat
            return PushFoldResult(
                Decision.BALANCED, 0.6,
                "ryanshanten, early round low threat",
                safety_weight=0.15
            )
        if threat >= 1.0:
            return PushFoldResult(
                Decision.CAUTIOUS, 0.7,
                "ryanshanten with threat",
                safety_weight=0.45
            )
        if hand_value >= risk * 0.8 and gs.remaining_tiles > 30:
            return PushFoldResult(
                Decision.BALANCED, 0.55,
                "ryanshanten but high value hand",
                safety_weight=0.25
            )
        return PushFoldResult(
            Decision.CAUTIOUS, 0.6,
            "ryanshanten, moderate defense",
            safety_weight=0.35
        )

    # === 3+ away: heavily defensive ===
    if threat >= 0.5:
        return PushFoldResult(
            Decision.FOLD, 0.8,
            f"shanten={shanten} with threat, full fold",
            safety_weight=0.70
        )
    if gs.remaining_tiles <= 30:
        return PushFoldResult(
            Decision.CAUTIOUS, 0.65,
            f"shanten={shanten}, late in round",
            safety_weight=0.50
        )
    return PushFoldResult(
        Decision.CAUTIOUS, 0.55,
        f"shanten={shanten}, cautious play",
        safety_weight=0.40
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
            # Leading in all-last: play very safe
            return PushFoldResult(
                max(result.decision, Decision.CAUTIOUS, key=lambda d: d.value),
                result.confidence,
                f"all-last 1st place: {result.reason}",
                safety_weight=max(result.safety_weight, 0.40)
            )
        if placement == 4:
            if diff_above <= 8000:
                # Deficit is reachable: push hard
                return PushFoldResult(
                    Decision.PUSH,
                    0.8,
                    "all-last 4th, reachable deficit",
                    safety_weight=max(0.0, result.safety_weight - 0.20)
                )
            else:
                # Need a big hand: moderate push
                return PushFoldResult(
                    Decision.BALANCED,
                    0.6,
                    "all-last 4th, large deficit",
                    safety_weight=max(0.0, result.safety_weight - 0.10)
                )

    # === South round general adjustments ===
    if gs.is_south:
        if placement == 1 and diff_below >= 12000:
            # Comfortable lead in south: protect it
            return PushFoldResult(
                result.decision,
                result.confidence,
                f"south, leading comfortably: {result.reason}",
                safety_weight=min(1.0, result.safety_weight + 0.10)
            )
        if placement == 4 and diff_above >= 20000:
            # Desperate 4th: push harder
            return PushFoldResult(
                result.decision,
                result.confidence,
                f"south 4th, desperate: {result.reason}",
                safety_weight=max(0.0, result.safety_weight - 0.15)
            )

    # === General placement adjustments ===
    if placement == 1:
        # Leading: slightly more conservative
        result.safety_weight = min(1.0, result.safety_weight + 0.05)
    elif placement == 4:
        # Last: slightly more aggressive
        result.safety_weight = max(0.0, result.safety_weight - 0.05)

    return result
