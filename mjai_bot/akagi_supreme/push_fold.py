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
    melds = gs.my_melds
    
    all_tiles = list(hand)
    for m in melds:
        all_tiles.extend(m.tiles)

    for tile in set(all_tiles):
        if gs.is_my_yakuhai(tile):
            count = sum(1 for t in all_tiles if tile_base(t) == tile)
            if count >= 3:
                han_estimate += 1.0  # confirmed yakuhai
            elif count >= 2:
                if gs.unseen_count(tile) > 0 and tile in set(hand):
                    han_estimate += 0.4  # possible pon

    # === Tanyao potential (断么九) ===
    # All tiles in hand AND melds must be 2-8
    all_tanyao = True
    for t in all_tiles:
        if t in HONORS:
            all_tanyao = False
            break
        s, r, _ = parse_tile(t)
        if r is not None and (r == 1 or r == 9):
            all_tanyao = False
            break
    if all_tanyao and len(all_tiles) >= 4:
        han_estimate += 1.0  # tanyao

    # === Chiitoitsu potential (七対子) ===
    pairs = 0
    counts = {}
    for t in hand:
        tb = tile_base(t)
        counts[tb] = counts.get(tb, 0) + 1
    for count in counts.values():
        if count >= 2:
            pairs += 1
    is_chiitoi_route = pairs >= 5
    if is_chiitoi_route:
        han_estimate += 2.0  # chiitoitsu

    # === Suit composition: honitsu/chinitsu potential ===
    suit_counts = {"m": 0, "p": 0, "s": 0, "z": 0}
    for t in all_tiles:
        if t in HONORS:
            suit_counts["z"] += 1
        else:
            s, _, _ = parse_tile(t)
            if s in suit_counts:
                suit_counts[s] += 1
    total_number = suit_counts["m"] + suit_counts["p"] + suit_counts["s"]
    if total_number > 0:
        dominant_suit_count = max(suit_counts["m"], suit_counts["p"], suit_counts["s"])
        if dominant_suit_count + suit_counts["z"] >= len(all_tiles) - 1:
            han_estimate += 1.5  # likely honitsu (2 han open, 3 closed)
        elif dominant_suit_count + suit_counts["z"] >= len(all_tiles) - 2:
            han_estimate += 0.5  # possible honitsu

    # === Toitoi potential ===
    pon_like_melds = sum(1 for m in melds if m.meld_type in ("pon", "daiminkan", "kakan", "ankan"))
    pairs_in_hand = 0
    counts = {}
    for t in hand:
        tb = tile_base(t)
        counts[tb] = counts.get(tb, 0) + 1
    for count in counts.values():
        if count >= 2:
            pairs_in_hand += 1
    if pon_like_melds + pairs_in_hand >= 3:
        han_estimate += 1.0  # strong toitoi potential

    # === Menzen bonus for closed hand ===
    if not gs.my_info.is_open():
        if gs.my_info.riichi_declared:
            # Already declared riichi: 100% guarantee of riichi (+1 han) + ura dora chance (~0.3 han)
            han_estimate += 1.3
        else:
            han_estimate += 0.7  # discounted riichi option value

        # Pinfu (平和) potential: mostly numbered tiles, little/no honors, and sequence-based
        # Skip pinfu potential for chiitoitsu route — chiitoi is pair-based, not sequence-based.
        # Adding pinfu bonus on top of chiitoi causes point overestimation.
        if not is_chiitoi_route and total_number >= 10 and suit_counts["z"] <= 1:
            shuntsu_potential = 0
            for suit in ["m", "p", "s"]:
                ranks = sorted(list(set([parse_tile(t)[1] for t in hand if parse_tile(t)[0] == suit and parse_tile(t)[1] is not None])))
                for i in range(len(ranks) - 1):
                    if ranks[i+1] - ranks[i] <= 2:
                        shuntsu_potential += 1
            if shuntsu_potential >= 3:
                han_estimate += 0.5

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
            if p.is_dealer:
                base_risk = max(base_risk, 11600)  # dealer damaten is extremely scary
            else:
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
    # === Already in riichi ===
    # A hand in riichi cannot fold or mawashi. The libriichi engine enforces this via action masks,
    # but logging FOLD for a riichi hand is semantically confusing. Return PUSH immediately.
    if gs.my_info.riichi_declared:
        return PushFoldResult(Decision.PUSH, 1.0, "already in riichi - locked")

    # === Ippatsu Avoidance ===
    # Top players avoid dealing into Ippatsu when far from tenpai.
    # However, iishanten with good shape should NOT auto-fold just for ippatsu —
    # top players evaluate the actual push/fold context (hand value, shape, threat)
    # and only fold ippatsu when they're 2+ away from tenpai.
    has_ippatsu = any(p.riichi_ippatsu for i, p in enumerate(gs.players) if i != gs.player_id)
    if has_ippatsu and shanten >= 2:
        return PushFoldResult(
            Decision.FOLD, 0.9,
            "opponent ippatsu turn, far from tenpai - strict fold"
        )
    # Iishanten + ippatsu: top players still mawashi if cheap bad-shape
    # Ippatsu ura dora stack can create haneman+, so avoid reckless push
    # with a hand that wouldn't justify the risk.
    if has_ippatsu and shanten == 1:
        hand_value = estimate_hand_value(gs)
        if hand_value < 3900 and (acceptance_count > 0 and acceptance_count <= 4):
            return PushFoldResult(
                Decision.MAWASHI, 0.75,
                "opponent ippatsu turn, iishanten but cheap bad-shape - mawashi"
            )

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
    # Use a discount factor on threat thresholds instead of mutating threat directly.
    # This avoids inconsistency with adjust_for_placement which re-reads raw threat.
    dealer_discount = 0.75 if gs.is_dealer_me else 1.0

    good_shape = acceptance_count >= 8
    bad_shape = acceptance_count > 0 and acceptance_count <= 4

    # === Tenpai: almost always push ===
    if shanten <= 0:
        # Cheap tenpai vs riichi/strong threat (dealer-adjusted)
        if hand_value < 3900 and threat >= 1.5 * dealer_discount:
            if bad_shape:
                return PushFoldResult(
                    Decision.FOLD, 0.8,
                    "cheap bad-shape tenpai vs strong threat - complete fold"
                )
            if my_turn >= 12 and hand_value <= 2000:
                return PushFoldResult(
                    Decision.MAWASHI, 0.65,
                    "cheap good-shape tenpai vs high threat late, avoiding risky deal-in"
                )
        
        # 2. Original extreme late game exception for moderate threat
        if threat >= 2.0 * dealer_discount and hand_value < 3900 and my_turn >= 14 and bad_shape:
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
        effective_threat = threat * dealer_discount
        if effective_threat <= 0.5:
            return PushFoldResult(Decision.PUSH, 0.85, "iishanten, low threat")

        # Top player logic: facing an attack, bad-shape + low-value = strictly FOLD/MAWASHI
        if effective_threat >= 1.0:
            if effective_threat >= 1.8:
                if hand_value >= 16000:
                    return PushFoldResult(Decision.MAWASHI, 0.6, "iishanten vs multiple threats, baiman+ hand, mawashi")
                return PushFoldResult(Decision.FOLD, 0.8, "iishanten vs multiple threats (>= 1.8), complete fold")
                
            if hand_value < risk * 0.8:
                if effective_threat >= 1.5:
                    return PushFoldResult(
                        Decision.FOLD, 0.8,
                        "iishanten, strong threat, not enough value - complete fold"
                    )
                if my_turn >= 9:
                    return PushFoldResult(
                        Decision.FOLD, 0.75,
                        "iishanten, threat, not enough value, late - fold"
                    )
                if bad_shape:
                    return PushFoldResult(
                        Decision.FOLD, 0.7,
                        "iishanten, early but bad shape vs threat - fold"
                    )
                return PushFoldResult(
                    Decision.MAWASHI, 0.7,
                    "iishanten, early vs threat"
                )

            # Good shape or decent value: evaluate late-game multi-threats
            if my_turn >= 12 or effective_threat >= 2.0:
                n_riichi = gs.num_riichi_opponents
                if n_riichi >= 2 and hand_value < risk * 0.5:
                    return PushFoldResult(Decision.FOLD, 0.7, "iishanten, late, double riichi, weak hand")
                if effective_threat >= 2.5 and hand_value < risk * 0.4:
                    return PushFoldResult(Decision.FOLD, 0.7, "iishanten, late, extreme threat, weak hand")
                
                # Top players leading will strictly fold against high threat when not tenpai
                if effective_threat >= 2.0 and gs.my_placement <= 2:
                    return PushFoldResult(Decision.FOLD, 0.75, "iishanten, 1st/2nd place vs extreme threat - complete fold")
                if my_turn >= 12 and effective_threat >= 1.5 and gs.my_placement <= 2 and hand_value < 8000:
                    return PushFoldResult(Decision.FOLD, 0.75, "iishanten, late vs threat, 1st/2nd place without mangan - complete fold")

                if hand_value >= risk * 0.5 and good_shape:
                    if effective_threat >= 2.0 or (my_turn >= 12 and effective_threat >= 1.5):
                        return PushFoldResult(Decision.MAWASHI, 0.65, "iishanten, late but valuable good-shape hand, forced to mawashi vs high threat")
                    return PushFoldResult(Decision.PUSH, 0.65, "iishanten, late but valuable good-shape hand")
                if hand_value >= risk * 0.5:
                    return PushFoldResult(Decision.MAWASHI, 0.65, "iishanten, late, valuable but uncertain shape")
                return PushFoldResult(Decision.FOLD, 0.6, "iishanten, late vs threat, weak hand")

        # Early-mid game with moderate/low threat
        if my_turn <= 8:
            if hand_value >= risk * 0.3:
                return PushFoldResult(Decision.PUSH, 0.75, "iishanten, early-mid game, decent value")
            return PushFoldResult(Decision.PUSH, 0.7, "iishanten, early-mid game")
        else:
            return PushFoldResult(Decision.MAWASHI, 0.6, "iishanten, late game")

    # === Ryanshanten (2-away) ===
    if shanten == 2:
        effective_threat = threat * dealer_discount
        if my_turn <= 6 and effective_threat < 0.8:
            return PushFoldResult(
                Decision.PUSH, 0.6,
                "ryanshanten, early round, low threat"
            )
        if my_turn <= 6 and effective_threat < 1.0:
            return PushFoldResult(
                Decision.MAWASHI, 0.6,
                "ryanshanten, early, moderate threat"
            )
        if effective_threat >= 1.5:
            return PushFoldResult(
                Decision.FOLD, 0.75,
                "ryanshanten, strong threats"
            )
        if effective_threat >= 1.0:
            # Top player logic: 2-shanten vs clear threat is an instant FOLD.
            return PushFoldResult(
                Decision.FOLD, 0.7,
                "ryanshanten vs threat >= 1.0, complete fold"
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
    effective_threat = threat * dealer_discount
    if my_turn <= 6 and effective_threat <= 0.5:
        return PushFoldResult(
            Decision.MAWASHI, 0.5,
            f"shanten={shanten}, very early, no threat"
        )
    if effective_threat >= 1.5:
        return PushFoldResult(
            Decision.FOLD, 0.85,
            f"shanten={shanten} vs threats, full fold"
        )
    if effective_threat >= 0.5:
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


def adjust_for_placement(result: PushFoldResult, gs: GameState, shanten: int = 6) -> PushFoldResult:
    """Adjust push/fold based on placement context.

    - 4th place: push harder (need to recover)
    - 1st place with big lead: play safer (protect lead)
    - All last special handling
    """
    placement = gs.my_placement
    diff_above = gs.diff_to_above
    diff_below = gs.diff_to_below
    threat = gs.max_opponent_threat()

    # If there is no real threat from opponents, do not force FOLD/MAWASHI purely due to placement.
    # Maintaining our hand structure against no threat is strictly superior.
    if threat < 0.5 and result.decision != Decision.FOLD:
        return result

    # === All Last (オーラス) special logic ===
    if gs.is_all_last:
        if placement == 1:
            # Thin lead where noten penalty could flip placement:
            # -3000 noten penalty at ryukyoku can drop 1st to 2nd/3rd.
            # Top players push tenpai/good iishanten to avoid this.
            noten_danger = diff_below <= abs(gs.noten_penalty_effect())
            if diff_below < 4000:
                # Upgrade FOLD/MAWASHI to at least MAWASHI when noten
                # penalty could cost us 1st place. Tenpai is defensive here,
                # but only if we are actually close to tenpai.
                if noten_danger and result.decision == Decision.FOLD and shanten <= 1:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        result.confidence,
                        f"all-last 1st, noten penalty risk, can't fully fold: {result.reason}"
                    )
                return result  # keep original decision, don't weaken to mawashi
            # Comfortable lead: top players fully fold to protect placement.
            # MAWASHI still risks deal-in; FOLD is the safe choice.
            if diff_below >= 12000:
                if result.decision in (Decision.PUSH, Decision.MAWASHI):
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
                # Only downgrade if there's a real threat or our hand isn't valuable enough to risk it.
                if result.decision == Decision.PUSH:
                    pts_for_1st = gs.points_needed_for_placement(1)
                    hand_value = estimate_hand_value(gs)
                    if threat >= 1.0 or hand_value < pts_for_1st:
                        return PushFoldResult(
                            Decision.MAWASHI,
                            result.confidence,
                            f"all-last 2nd, 3rd is close ({diff_below}pts gap), avoiding risk: {result.reason}"
                        )
            return result

        if placement == 3:
            # All-last 3rd: ラス回避が最優先.
            # If 4th is close behind, push harder to secure agari and avoid 4th.
            # If 4th is far behind, can play more carefully.
            if diff_below < 4000:
                # 4th is close. We cannot afford to fully fold and hope 4th doesn't win.
                if result.decision == Decision.FOLD:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        result.confidence,
                        f"all-last 3rd, 4th close ({diff_below}pts), can't fully fold: {result.reason}"
                    )
                # Upgrade MAWASHI to PUSH only if we are very close to winning (shanten <= 1).
                # Reckless pushing of a bad hand against a strong threat just causes deal-in to 4th.
                if result.decision == Decision.MAWASHI and shanten <= 1:
                    return PushFoldResult(
                        Decision.PUSH,
                        result.confidence,
                        f"all-last 3rd, 4th close ({diff_below}pts), push to avoid ラス: {result.reason}"
                    )
            return result

        if placement == 4:
            # All-last 4th: 4th is already the worst outcome.
            # Folding doesn't improve placement — at minimum MAWASHI to keep hand alive.
            from .placement_strategy import compute_placement_adjustment
            p_adj = compute_placement_adjustment(gs)
            hand_value = estimate_hand_value(gs)
            
            # 条件を満たしうる手（または親）なら、Mortalがオリたがっていても強気に攻める
            pts_for_3rd = gs.points_needed_for_placement(3)
            required_pts = p_adj.min_push_value if p_adj.min_push_value > 0 else pts_for_3rd
            
            if gs.is_dealer_me or hand_value >= required_pts * 0.5:
                if result.decision in (Decision.FOLD, Decision.MAWASHI):
                    return PushFoldResult(
                        Decision.PUSH,
                        0.85,
                        "all-last 4th, must push to escape 4th"
                    )
            
            # 自分が親ではなく、順位逆転に必要な最低打点に届かず、
            # 相手から強い攻撃を受けている場合 → PUSHをMAWASHIに降格
            # ただし4位である以上、FOLDまでは落とさない（降りても4位のまま）
            if threat >= 1.0 and result.decision == Decision.PUSH:
                if not gs.is_dealer_me and p_adj.min_push_value > 0 and hand_value < p_adj.min_push_value * 0.5:
                    return PushFoldResult(
                        Decision.MAWASHI,
                        0.7,
                        "all-last 4th, cheap hand vs threat, cannot reach min push value, mawashi"
                    )
            
            # Final safety net: 4th place should never FOLD.
            # Folding doesn't change placement (already worst), so at minimum MAWASHI
            # to keep hand alive for any chance of improvement.
            if result.decision == Decision.FOLD:
                return PushFoldResult(
                    Decision.MAWASHI,
                    0.6,
                    f"all-last 4th, upgrading fold to mawashi (folding is pointless at 4th): {result.reason}"
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
                # Very desperate: push if we have a realistic chance (shanten <= 1), 
                # otherwise just try to mawashi to avoid meaningless deal-in
                if shanten <= 1:
                    if result.decision in (Decision.FOLD, Decision.MAWASHI):
                        return PushFoldResult(
                            Decision.PUSH,
                            result.confidence,
                            f"south 4th, desperate deficit, push: {result.reason}"
                        )
                else:
                    if result.decision == Decision.FOLD:
                        return PushFoldResult(
                            Decision.MAWASHI,
                            result.confidence,
                            f"south 4th, desperate deficit, avoid full fold: {result.reason}"
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
