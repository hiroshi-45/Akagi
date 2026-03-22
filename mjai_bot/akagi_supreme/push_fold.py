# -*- coding: utf-8 -*-
"""Push/fold decision engine.

Determines whether the bot should push (attack), fold (defend),
or play mawashi (defend while maintaining hand shape).

Key design principle: **Binary decision, not blended.**
Top players think in terms of "push or fold", not "70% push / 30% fold".
When they push, they trust their tile efficiency fully.
When they fold, they go full betaori.
Mawashi is the middle ground: prioritize safety but pick the least
damaging safe tile for hand progression.

This avoids double-counting safety that's already in Mortal's Q-values.

Factors:
- Shanten count (distance to tenpai)
- Opponent threat level (riichi, open hands, behavioral patterns)
- Hand value potential (dora, yakuhai, suit composition)
- Placement and point situation
- Turn number
- Acceptance count (how live is the hand)
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
    PUSH = 0       # Full attack - trust Mortal's action completely
    MAWASHI = 1    # Defend but maintain hand - safe tiles first, Q-value tiebreak
    FOLD = 2       # Full betaori - safest tile regardless of hand value


@dataclass
class PushFoldResult:
    decision: Decision
    confidence: float  # 0.0 to 1.0
    reason: str


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

    # === Tanyao potential (断么九) ===
    # All tiles are 2-8 numbered tiles (no terminals or honors)
    all_tanyao = True
    for t in hand:
        if t in HONORS:
            all_tanyao = False
            break
        s, r, _ = parse_tile(t)
        if r is not None and (r == 1 or r == 9):
            all_tanyao = False
            break
    if all_tanyao and len(hand) >= 4:
        han_estimate += 1.0  # tanyao

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
        if dominant_suit_count + suit_counts["z"] >= len(hand) - 1:
            han_estimate += 1.5  # likely honitsu (2 han open, 3 closed)
        elif dominant_suit_count + suit_counts["z"] >= len(hand) - 2:
            han_estimate += 0.5  # possible honitsu

    # === Menzen bonus for closed hand ===
    # Closed hands CAN declare riichi (+1 han + ura dora), but we don't
    # know at estimation time whether we actually will. Use a discounted
    # value: menzen tsumo (0.5 probability-weighted) + riichi option value.
    # Full riichi value (1.0 + 0.3 ura) was too aggressive — it inflated
    # hand value for dama/fold situations where riichi won't happen.
    if not gs.my_info.is_open():
        han_estimate += 0.7  # discounted riichi option value

    # === Convert han estimate to points ===
    if han_estimate >= 11:
        value = 36000.0 if gs.is_dealer_me else 24000.0
    elif han_estimate >= 8:
        value = 24000.0 if gs.is_dealer_me else 16000.0
    elif han_estimate >= 6:
        value = 18000.0 if gs.is_dealer_me else 12000.0
    elif han_estimate >= 5:
        value = 12000.0 if gs.is_dealer_me else 8000.0
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

    Enhanced with open hand point estimation.
    """
    base_risk = 5200.0  # average deal-in cost

    # Riichi opponents — double riichi is extremely dangerous because
    # two independent threats means many more tiles are dangerous.
    n_riichi = gs.num_riichi_opponents
    if n_riichi >= 2:
        base_risk *= 1.8
    elif n_riichi == 1:
        for i, p in enumerate(gs.players):
            if i != gs.player_id and p.riichi_declared:
                if p.is_dealer:
                    # Dealer riichi: base cost is much higher (dealer mangan
                    # = 12000). Top players treat dealer riichi as 1.5-2x
                    # the danger of non-dealer riichi.
                    base_risk *= 1.6
                if p.riichi_turn <= 6:
                    # Early riichi = likely good hand (good shape, dora, etc.)
                    base_risk *= 1.3

    # Ippatsu bonus: during ippatsu window, deal-in cost is +1 han.
    # Top players are extra cautious with their first discard after
    # someone's riichi — ippatsu ura dora stack can create haneman+.
    has_ippatsu = any(
        p.riichi_ippatsu for i, p in enumerate(gs.players)
        if i != gs.player_id
    )
    if has_ippatsu:
        base_risk *= 1.25  # +1 han ≈ roughly doubles low-han hands

    # Open hands with estimated point values
    for i, p in enumerate(gs.players):
        if i == gs.player_id:
            continue
        if p.is_open():
            estimated_pts = p.estimate_open_hand_points(
                gs.round_wind, gs._opponent_wind(i), gs.doras)
            if estimated_pts > 0:
                base_risk = max(base_risk, float(estimated_pts))

    # Dama tenpai signal from tedashi patterns.
    # A closed hand showing tsumogiri streak → tedashi pattern suggests
    # tenpai with a potentially high-value damaten hand. Top players treat
    # this as a serious threat (could be mangan+ closed hand).
    for i, p in enumerate(gs.players):
        if i == gs.player_id or p.riichi_declared:
            continue
        if p.tedashi_after_tsumogiri_streak():
            base_risk = max(base_risk, 7700)  # assume at least mangan potential

    # Honba bonus
    base_risk += gs.honba * 300

    return base_risk


def evaluate_push_fold(gs: GameState, shanten: int,
                       acceptance_count: int = 0) -> PushFoldResult:
    """Main push/fold evaluation.

    Returns a binary decision: PUSH, MAWASHI, or FOLD.
    No safety_weight — the caller uses different tile selection
    strategies based on the decision.

    Design: Top players think "push or fold", not "partially push".
    - Tenpai: almost always PUSH
    - Good iishanten: PUSH unless extreme threat
    - Bad iishanten / ryanshanten with threat: MAWASHI
    - Far from tenpai with threat: FOLD
    """
    hand_value = estimate_hand_value(gs)
    risk = estimate_risk_of_deal_in(gs)
    threat = gs.max_opponent_threat()
    my_turn = gs.my_turn

    # === Dealer bonus (親の打点1.5倍 + 連荘価値) ===
    # Dealers get 1.5x payout AND keep dealership on win (renchan).
    # Top players push significantly harder as dealer because:
    # 1. Hand value is 1.5x (already reflected in estimate_hand_value)
    # 2. Winning maintains oya = another chance to score
    # 3. Not winning passes oya (opportunity cost)
    # Reduce effective threat when we're dealer to model this aggression.
    if gs.is_dealer_me:
        threat *= 0.75  # dealer should push through moderate threats

    good_shape = acceptance_count >= 8
    bad_shape = acceptance_count > 0 and acceptance_count <= 4

    # === Tenpai: almost always push ===
    if shanten <= 0:
        # Only exception: extremely cheap bad-shape tenpai vs extreme threat
        # in very late game. Top players mawashi when the expected gain
        # from pushing is far less than the expected loss from deal-in.
        if threat >= 3.0 and hand_value < 3900 and my_turn >= 14 and bad_shape:
            return PushFoldResult(
                Decision.MAWASHI, 0.6,
                "tenpai but cheap bad-shape vs extreme threat, very late"
            )
        return PushFoldResult(
            Decision.PUSH, 0.95,
            "tenpai - push"
        )

    # === Iishanten (1-away) ===
    if shanten == 1:
        if threat <= 0.5:
            return PushFoldResult(Decision.PUSH, 0.85, "iishanten, low threat")

        if my_turn <= 8:
            # Early/mid game iishanten: push unless hand is garbage
            if hand_value >= risk * 0.3:
                return PushFoldResult(
                    Decision.PUSH, 0.75,
                    "iishanten, early-mid game, decent value"
                )
            if threat >= 2.0:
                return PushFoldResult(
                    Decision.MAWASHI, 0.7,
                    "iishanten, early-mid, high threat, weak hand"
                )
            return PushFoldResult(
                Decision.PUSH, 0.7,
                "iishanten, early-mid game"
            )
        else:
            # Late game iishanten: top players are much more cautious here.
            # Multiple riichi or extreme threat with cheap hand → fold.
            # Two+ riichi opponents means both threat AND risk are elevated;
            # top players fold iishanten here unless the hand is very strong.
            n_riichi = gs.num_riichi_opponents
            if n_riichi >= 2 and hand_value < risk * 0.5:
                return PushFoldResult(
                    Decision.FOLD, 0.7,
                    "iishanten, late, double riichi, weak hand"
                )
            if threat >= 2.5 and hand_value < risk * 0.4:
                return PushFoldResult(
                    Decision.FOLD, 0.7,
                    "iishanten, late, extreme threat, weak hand"
                )
            if hand_value >= risk * 0.5 and good_shape:
                return PushFoldResult(
                    Decision.PUSH, 0.65,
                    "iishanten, late but valuable good-shape hand"
                )
            if hand_value >= risk * 0.5:
                return PushFoldResult(
                    Decision.MAWASHI, 0.65,
                    "iishanten, late, valuable but uncertain shape"
                )
            if threat >= 2.0:
                return PushFoldResult(
                    Decision.MAWASHI, 0.6,
                    "iishanten, late, multiple threats"
                )
            return PushFoldResult(
                Decision.MAWASHI, 0.6,
                "iishanten, late game"
            )

    # === Ryanshanten (2-away) ===
    if shanten == 2:
        if my_turn <= 6 and threat <= 0.5:
            return PushFoldResult(
                Decision.PUSH, 0.6,
                "ryanshanten, early round, low threat"
            )
        if my_turn <= 6 and threat < 1.5:
            return PushFoldResult(
                Decision.MAWASHI, 0.6,
                "ryanshanten, early, moderate threat"
            )
        if threat >= 2.0:
            return PushFoldResult(
                Decision.FOLD, 0.7,
                "ryanshanten, strong threats"
            )
        if threat >= 1.0:
            if hand_value >= risk * 0.5 and good_shape and my_turn <= 10:
                return PushFoldResult(
                    Decision.MAWASHI, 0.6,
                    "ryanshanten, riichi but valuable+connected, mid-game"
                )
            return PushFoldResult(
                Decision.FOLD, 0.65,
                "ryanshanten, riichi opponent"
            )
        if my_turn >= 12:
            return PushFoldResult(
                Decision.MAWASHI, 0.6,
                "ryanshanten, late, no threat"
            )
        return PushFoldResult(
            Decision.MAWASHI, 0.55,
            "ryanshanten, cautious play"
        )

    # === 3+ away: fold or mawashi ===
    if my_turn <= 6 and threat <= 0.5:
        return PushFoldResult(
            Decision.MAWASHI, 0.5,
            f"shanten={shanten}, very early, no threat"
        )
    if threat >= 1.5:
        return PushFoldResult(
            Decision.FOLD, 0.85,
            f"shanten={shanten} vs threats, full fold"
        )
    if threat >= 0.5:
        return PushFoldResult(
            Decision.FOLD, 0.7,
            f"shanten={shanten} with some threat"
        )
    if my_turn >= 10:
        return PushFoldResult(
            Decision.FOLD, 0.6,
            f"shanten={shanten}, late game"
        )
    return PushFoldResult(
        Decision.MAWASHI, 0.55,
        f"shanten={shanten}, early but far from tenpai"
    )


def adjust_for_placement(result: PushFoldResult, gs: GameState) -> PushFoldResult:
    """Adjust push/fold based on placement context.

    - 4th place: push harder (need to recover)
    - 1st place with big lead: play safer (protect lead)
    - All last special handling
    """
    placement = gs.my_placement
    diff_above = gs.diff_to_above
    diff_below = gs.diff_to_below

    # === All Last (オーラス) special logic ===
    if gs.is_all_last:
        if placement == 1:
            # Thin lead where noten penalty could flip placement:
            # -3000 noten penalty at ryukyoku can drop 1st to 2nd/3rd.
            # Top players push tenpai/good iishanten to avoid this.
            noten_danger = diff_below <= abs(gs.noten_penalty_effect())
            if diff_below < 4000:
                # Upgrade FOLD/MAWASHI to at least MAWASHI when noten
                # penalty could cost us 1st place. Tenpai is defensive here.
                if noten_danger and result.decision == Decision.FOLD:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        result.confidence,
                        f"all-last 1st, noten penalty risk, can't fully fold: {result.reason}"
                    )
                return result  # keep original decision, don't weaken to mawashi
            # Comfortable lead: top players fully fold to protect placement.
            # MAWASHI still risks deal-in; FOLD is the safe choice.
            if diff_below >= 12000:
                if result.decision == Decision.PUSH:
                    return PushFoldResult(
                        Decision.FOLD,
                        result.confidence,
                        f"all-last 1st, big lead, full fold: {result.reason}"
                    )
                return result
            # Moderate lead (4000-12000): mawashi to avoid unnecessary risk
            if result.decision == Decision.PUSH:
                return PushFoldResult(
                    Decision.MAWASHI,
                    result.confidence,
                    f"all-last 1st place, moderate lead, careful: {result.reason}"
                )
            return result

        if placement == 2:
            # All-last 2nd: priority depends on gap to 3rd.
            # If 3rd is close, protect 2nd (避ラス > トップ取り).
            # If 3rd is far, can afford to push for 1st.
            if diff_below < 4000:
                # 3rd is close — protect 2nd place.
                # Deal-in could drop us to 3rd or worse.
                if result.decision == Decision.PUSH:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        result.confidence,
                        f"all-last 2nd, 3rd is close ({diff_below}pts gap): {result.reason}"
                    )
            return result

        if placement == 3:
            # All-last 3rd: ラス回避が最優先.
            # If 4th is close behind, push harder to secure agari and avoid 4th.
            # If 4th is far behind, can play more carefully.
            if diff_below < 4000:
                # 4th is close — need to win to avoid dropping to last.
                if result.decision == Decision.FOLD:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        result.confidence,
                        f"all-last 3rd, 4th close ({diff_below}pts), can't fully fold: {result.reason}"
                    )
                if result.decision == Decision.MAWASHI:
                    return PushFoldResult(
                        Decision.PUSH,
                        result.confidence,
                        f"all-last 3rd, 4th close ({diff_below}pts), push to avoid ラス: {result.reason}"
                    )
            return result

        if placement == 4:
            # All-last 4th: 4th is already the worst outcome in ranked
            # mahjong (-30 uma). Any chance to climb is worth taking.
            # Top players push with almost any hand here — even with huge
            # deficits, because dealing in doesn't worsen 4th place.
            if result.decision in (Decision.FOLD, Decision.MAWASHI):
                return PushFoldResult(
                    Decision.PUSH,
                    0.85,
                    "all-last 4th, must push regardless of deficit"
                )
            return result

    # === South round general adjustments ===
    if gs.is_south:
        if placement == 1 and diff_below >= 12000:
            # Comfortable lead: upgrade push to mawashi
            if result.decision == Decision.PUSH:
                return PushFoldResult(
                    Decision.MAWASHI,
                    result.confidence,
                    f"south, leading comfortably: {result.reason}"
                )
        if placement == 4:
            # South 4th: need to recover before all-last.
            # Top players push more aggressively but not recklessly —
            # unlike all-last, there are still rounds left to recover.
            # Jumping from FOLD straight to PUSH risks unnecessary deal-ins.
            if diff_above >= 20000:
                # Very desperate: upgrade fold to mawashi (not push —
                # still rounds left, and a big deal-in makes it worse)
                if result.decision == Decision.FOLD:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        result.confidence,
                        f"south 4th, big deficit, at least mawashi: {result.reason}"
                    )
                if result.decision == Decision.MAWASHI:
                    return PushFoldResult(
                        Decision.PUSH,
                        result.confidence,
                        f"south 4th, desperate push: {result.reason}"
                    )
            elif diff_above >= 8000:
                # Moderate deficit: at least mawashi
                if result.decision == Decision.FOLD:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        result.confidence,
                        f"south 4th, moderate deficit: {result.reason}"
                    )

    return result
