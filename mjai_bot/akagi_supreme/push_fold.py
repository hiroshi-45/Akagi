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
    """Estimate potential hand value in points (rough).

    Returns estimated point value (tsumo/ron average) as an actual point figure,
    comparable to estimate_risk_of_deal_in().
    """
    # Base: menzen riichi tsumo ~ 2000 non-dealer, 3900 dealer (1 han 30 fu average)
    if gs.is_dealer_me:
        value = 3900.0
    else:
        value = 2000.0

    # Dora count — each dora roughly doubles/adds a han
    dora_count = gs.count_dora_in_hand()
    if dora_count >= 4:
        value *= 6.0   # haneman+
    elif dora_count >= 3:
        value *= 4.0   # mangan~haneman
    elif dora_count >= 2:
        value *= 2.5   # ~ 3 han
    elif dora_count >= 1:
        value *= 1.6   # ~ 2 han

    # Yakuhai in hand (potential pon or head)
    hand_tiles = gs.my_hand
    for tile in hand_tiles:
        if tile in YAKUHAI_HONORS:
            count = sum(1 for t in hand_tiles if t == tile)
            if count >= 3:
                value *= 1.5  # confirmed yakuhai pon
            elif count >= 2:
                value *= 1.15  # possible pon

    # Riichi bonus for closed hand (ippatsu/ura potential)
    if not gs.my_info.is_open():
        value *= 1.2

    # Kyotaku on table: winning gains all accumulated riichi sticks
    value += gs.kyotaku * 1000

    # Honba bonus
    value += gs.honba * 300

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
    # Top players virtually never fold from tenpai. The only exception is when
    # facing double/triple riichi with a cheap hand and real safe tiles available.
    if shanten <= 0:
        if threat >= 3.0 and hand_value < 2000 and gs.remaining_tiles <= 30:
            # Double+ riichi, late in round, cheap hand: slight safety bias on tile choice
            return PushFoldResult(
                Decision.BALANCED, 0.6,
                "tenpai but cheap hand vs double riichi late game",
                safety_weight=0.10
            )
        return PushFoldResult(
            Decision.PUSH, 0.95,
            "tenpai - push",
            safety_weight=0.0
        )

    # === Iishanten (1-away): context-dependent ===
    if shanten == 1:
        if threat <= 0.5:
            # No serious threat: proceed toward tenpai
            return PushFoldResult(
                Decision.PUSH, 0.8,
                "iishanten, low threat",
                safety_weight=0.05
            )
        # Value/risk ratio: both in the same point unit now
        if hand_value >= risk * 0.5:
            # Hand value justifies the risk
            return PushFoldResult(
                Decision.BALANCED, 0.7,
                "iishanten, decent value vs risk",
                safety_weight=0.12
            )
        if threat >= 1.5:
            # Single riichi or dangerous open hand: moderate safety bias
            # but do NOT heavily fold from iishanten — top players keep advancing
            return PushFoldResult(
                Decision.BALANCED, 0.65,
                "iishanten, riichi opponent",
                safety_weight=0.20
            )
        return PushFoldResult(
            Decision.BALANCED, 0.7,
            "iishanten, moderate situation",
            safety_weight=0.15
        )

    # === Ryanshanten (2-away): lean defensive ===
    if shanten == 2:
        if threat <= 0.3 and gs.remaining_tiles > 40:
            # Early in round, no threat
            return PushFoldResult(
                Decision.BALANCED, 0.6,
                "ryanshanten, early round low threat",
                safety_weight=0.12
            )
        if threat >= 2.0:
            # Multiple or very dangerous threats: cautious play
            return PushFoldResult(
                Decision.CAUTIOUS, 0.7,
                "ryanshanten vs multiple threats",
                safety_weight=0.40
            )
        if threat >= 1.0:
            return PushFoldResult(
                Decision.CAUTIOUS, 0.65,
                "ryanshanten with single riichi",
                safety_weight=0.30
            )
        if hand_value >= risk * 0.6 and gs.remaining_tiles > 30:
            return PushFoldResult(
                Decision.BALANCED, 0.55,
                "ryanshanten but high value hand",
                safety_weight=0.20
            )
        return PushFoldResult(
            Decision.CAUTIOUS, 0.6,
            "ryanshanten, moderate defense",
            safety_weight=0.28
        )

    # === 3+ away: heavily defensive ===
    if threat >= 2.0:
        # Multiple threats: full fold
        return PushFoldResult(
            Decision.FOLD, 0.85,
            f"shanten={shanten} vs multiple threats, full fold",
            safety_weight=0.70
        )
    if threat >= 1.0:
        return PushFoldResult(
            Decision.CAUTIOUS, 0.75,
            f"shanten={shanten} with riichi opponent",
            safety_weight=0.55
        )
    if gs.remaining_tiles <= 30:
        return PushFoldResult(
            Decision.CAUTIOUS, 0.65,
            f"shanten={shanten}, late in round",
            safety_weight=0.45
        )
    return PushFoldResult(
        Decision.CAUTIOUS, 0.55,
        f"shanten={shanten}, cautious play",
        safety_weight=0.35
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
            # Use integer-valued enum so max() picks the more defensive option
            more_defensive = result.decision if result.decision.value >= Decision.CAUTIOUS.value else Decision.CAUTIOUS
            return PushFoldResult(
                more_defensive,
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
