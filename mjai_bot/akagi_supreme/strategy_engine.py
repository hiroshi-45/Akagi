# -*- coding: utf-8 -*-
"""Strategy engine: the brain of akagi_supreme.

Wraps Mortal's neural network decisions with strategic overlays:
1. Game state tracking
2. Push/fold evaluation
3. Placement-aware adjustments
4. Safety-integrated action reranking
5. Meld/riichi decision overrides

The engine RESPECTS Mortal's Q-values as the primary signal and only
adjusts when strategic context clearly demands it.

Design principle: Minimize overrides. Mortal's NN already encodes
most tactical knowledge. We only intervene for:
- Placement-driven strategy changes (ラス回避, トップ取り)
- All-last special logic
- Clear defensive situations where NN may not weight placement enough
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from ..strategy.safety import (
    SafetyContext, aggregate_danger, bucketize,
    parse_tile, is_honor, only_tiles, SUITS
)
from .game_state import GameState, Tile, tile_base
from .push_fold import (
    Decision, PushFoldResult, evaluate_push_fold,
    adjust_for_placement, estimate_hand_value, estimate_risk_of_deal_in
)
from .placement_strategy import (
    PlacementAdjustment, compute_placement_adjustment, should_damaten
)


# Action type indices in the 46-action space
# 0-33: tile discards (1m-9m, 1p-9p, 1s-9s, E,S,W,N,P,F,C)
# 34-36: red five discards (5mr, 5pr, 5sr)
# 37: reach, 38: chi_low, 39: chi_mid, 40: chi_high
# 41: pon, 42: kan_select, 43: hora, 44: ryukyoku, 45: none

ACTION_TILE_NAMES = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
]

IDX_REACH = 37
IDX_CHI_LOW = 38
IDX_CHI_MID = 39
IDX_CHI_HIGH = 40
IDX_PON = 41
IDX_KAN = 42
IDX_HORA = 43
IDX_RYUKYOKU = 44
IDX_NONE = 45

DISCARD_INDICES = set(range(37))  # 0-36 are all tile discards
MELD_INDICES = {IDX_CHI_LOW, IDX_CHI_MID, IDX_CHI_HIGH, IDX_PON, IDX_KAN}


class StrategyEngine:
    """Applies strategic overlays to Mortal's raw action selection."""

    def __init__(self):
        self.gs = GameState()
        self._last_shanten: int = 6  # cached shanten from libriichi

    def process_event(self, event: dict) -> None:
        """Update internal game state from MJAI event."""
        self.gs.process_event(event)

    def set_shanten(self, shanten: int) -> None:
        """Update shanten from libriichi's PlayerState."""
        self._last_shanten = shanten

    def adjust_action(
        self,
        q_values: List[float],
        mask: List[bool],
        mortal_action: int,
        is_greedy: bool,
    ) -> int:
        """Adjust Mortal's selected action based on strategic context.

        This is the core method. It takes Mortal's Q-values and selected action,
        and may override the selection when strategic factors demand it.

        Returns:
            Adjusted action index.
        """
        if not self.gs._initialized:
            return mortal_action

        # === CRITICAL: Always take winning moves first ===
        # Check mask before checking mortal_action to prevent miss when
        # Mortal returns IDX_NONE but hora is available.
        # Note: In ranked Majsoul, winning is virtually always correct.
        # The extremely rare exception (yakuman pursuit) is not worth
        # the complexity of checking.
        if IDX_HORA < len(mask) and mask[IDX_HORA]:
            return IDX_HORA

        # === Always take winning moves ===
        if mortal_action == IDX_HORA:
            return mortal_action

        # === Evaluate strategic context ===
        acceptance = self.gs.estimate_acceptance_count()
        pf_result = evaluate_push_fold(self.gs, self._last_shanten, acceptance)
        pf_result = adjust_for_placement(pf_result, self.gs)
        p_adj = compute_placement_adjustment(self.gs)

        # === Pass override: check if we should meld when Mortal passes ===
        if mortal_action == IDX_NONE:
            override = self._check_pass_override(q_values, mask, p_adj, pf_result)
            if override is not None:
                return override
            return mortal_action

        # === Riichi decision ===
        if mortal_action == IDX_REACH:
            return self._adjust_riichi(q_values, mask, p_adj, acceptance)

        # === Meld decisions (chi/pon/kan) ===
        if mortal_action in MELD_INDICES:
            return self._adjust_meld(mortal_action, q_values, mask, p_adj, pf_result)

        # === Discard decisions ===
        if mortal_action in DISCARD_INDICES:
            return self._adjust_discard(mortal_action, q_values, mask, pf_result, p_adj)

        return mortal_action

    def _check_pass_override(
        self,
        q_values: List[float],
        mask: List[bool],
        p_adj: PlacementAdjustment,
        pf_result: PushFoldResult,
    ) -> Optional[int]:
        """Check if we should override Mortal's pass (IDX_NONE) with a meld.

        This handles cases where placement demands speed (e.g. all-last 4th)
        but Mortal's NN doesn't weight placement urgency enough.

        Returns the override action index, or None to keep the pass.
        """
        gs = self.gs

        # Only override in desperate situations
        if not gs.is_all_last:
            return None
        if gs.my_placement <= 2:
            return None  # 1st/2nd don't need desperate melds

        # Check if any meld options are available
        available_melds = []
        for idx in MELD_INDICES:
            if idx < len(mask) and mask[idx]:
                available_melds.append(idx)
        if not available_melds:
            return None

        # All-last 4th: strongly consider melding for speed
        if gs.my_placement == 4:
            # Check if meld multiplier encourages it
            if p_adj.meld_multiplier >= 1.1:
                # Pick the meld with highest Q-value
                best_meld = max(available_melds, key=lambda idx: q_values[idx])
                none_q = q_values[IDX_NONE] if IDX_NONE < len(q_values) else 0.0
                meld_q = q_values[best_meld]
                # Override if Q-values are close (within 0.15) — Mortal slightly
                # prefers pass but placement demands speed
                if meld_q >= none_q - 0.15:
                    return best_meld

        # All-last 3rd with 4th close: consider melding
        if gs.my_placement == 3 and gs.diff_to_below < 4000:
            if p_adj.meld_multiplier >= 1.05:
                best_meld = max(available_melds, key=lambda idx: q_values[idx])
                none_q = q_values[IDX_NONE] if IDX_NONE < len(q_values) else 0.0
                meld_q = q_values[best_meld]
                if meld_q >= none_q - 0.08:
                    return best_meld

        return None

    def _adjust_riichi(
        self,
        q_values: List[float],
        mask: List[bool],
        p_adj: PlacementAdjustment,
        acceptance_count: int = 0,
    ) -> int:
        """Decide whether to declare riichi or damaten."""
        hand_value = estimate_hand_value(self.gs)

        # Check if damaten is strategically preferred
        if should_damaten(self.gs, p_adj, hand_value=hand_value,
                          acceptance_count=acceptance_count):
            best_discard = self._find_best_discard(q_values, mask)
            if best_discard is not None:
                return best_discard
            return IDX_REACH

        # Apply riichi multiplier to Q-value comparison
        if p_adj.riichi_multiplier < 0.8:
            best_discard = self._find_best_discard(q_values, mask)
            if best_discard is not None:
                riichi_q = q_values[IDX_REACH] if IDX_REACH < len(q_values) else float('-inf')
                discard_q = q_values[best_discard]
                # Use difference-based comparison (not ratio) to handle negative Q-values
                q_diff = riichi_q - discard_q
                # Only riichi if its Q-value clearly exceeds best discard
                # With riichi_multiplier < 0.8, we need riichi to be clearly better
                if q_diff < 0.05:
                    return best_discard

        return IDX_REACH

    def _adjust_meld(
        self,
        mortal_action: int,
        q_values: List[float],
        mask: List[bool],
        p_adj: PlacementAdjustment,
        pf_result: PushFoldResult,
    ) -> int:
        """Decide whether to accept or decline a meld opportunity.

        Enhanced with:
        - Fold mode: skip chi, only allow defensive pon/kan
        - Post-meld defense consideration (open hand = less safe tiles)
        - Meld value assessment based on what the meld contributes
        """
        # In full fold mode: skip chi entirely, be cautious with pon
        if pf_result.decision == Decision.FOLD:
            if mortal_action in {IDX_CHI_LOW, IDX_CHI_MID, IDX_CHI_HIGH}:
                if IDX_NONE < len(mask) and mask[IDX_NONE]:
                    return IDX_NONE  # pass on chi while folding
            # For pon in fold mode: only accept if Mortal strongly prefers it
            if mortal_action == IDX_PON:
                if IDX_NONE < len(mask) and mask[IDX_NONE]:
                    pon_q = q_values[mortal_action]
                    none_q = q_values[IDX_NONE]
                    # Need significant Q-value advantage to pon while folding
                    if pon_q < none_q + 0.1:
                        return IDX_NONE

        # Cautious mode: slight bias against chi (reduces defense)
        if pf_result.decision == Decision.CAUTIOUS:
            if mortal_action in {IDX_CHI_LOW, IDX_CHI_MID, IDX_CHI_HIGH}:
                if IDX_NONE < len(mask) and mask[IDX_NONE]:
                    chi_q = q_values[mortal_action]
                    none_q = q_values[IDX_NONE]
                    # Chi needs clear Q-value advantage in cautious mode
                    # Use difference-based comparison for negative Q-value safety
                    q_diff = chi_q - none_q
                    threshold = 0.05 if p_adj.meld_multiplier >= 1.0 else 0.10
                    if q_diff < threshold:
                        return IDX_NONE

        # Apply meld multiplier for general discouragement
        if p_adj.meld_multiplier < 0.85:
            meld_q = q_values[mortal_action]
            none_available = IDX_NONE < len(mask) and mask[IDX_NONE]
            if none_available:
                none_q = q_values[IDX_NONE]
                # Use difference-based: meld needs to exceed none by enough margin
                q_diff = meld_q - none_q
                # Scale threshold by how much we're discouraging melds
                threshold = 0.05 * (1.0 - p_adj.meld_multiplier) / 0.15
                if q_diff < threshold:
                    return IDX_NONE

        return mortal_action

    def _adjust_discard(
        self,
        mortal_action: int,
        q_values: List[float],
        mask: List[bool],
        pf_result: PushFoldResult,
        p_adj: PlacementAdjustment,
    ) -> int:
        """Adjust tile discard selection with safety consideration.

        Enhanced with:
        - Genbutsu (現物) priority in fold mode
        - Higher safety cap for full fold (up to 1.0)
        """
        safety_weight = pf_result.safety_weight + p_adj.extra_safety

        # Full fold: allow safety to dominate completely
        if pf_result.decision == Decision.FOLD:
            safety_weight = max(0.0, min(1.0, safety_weight))
        else:
            safety_weight = max(0.0, min(0.80, safety_weight))

        # If no safety concern, trust Mortal completely
        if safety_weight <= 0.03:
            return mortal_action

        # Build safety context for danger evaluation
        ctx = self._build_safety_context()
        if ctx is None:
            return mortal_action

        # === Genbutsu (現物) priority in fold/cautious mode ===
        # When folding, always prefer genbutsu (tiles in opponents' rivers) first
        if pf_result.decision in (Decision.FOLD, Decision.CAUTIOUS) and safety_weight >= 0.30:
            genbutsu_action = self._find_genbutsu_discard(mask)
            if genbutsu_action is not None:
                # In full fold with high safety weight, always use genbutsu
                if pf_result.decision == Decision.FOLD and safety_weight >= 0.50:
                    return genbutsu_action
                # In cautious, prefer genbutsu if Mortal's choice isn't much better
                mortal_q = q_values[mortal_action]
                genbutsu_q = q_values[genbutsu_action]
                q_diff = mortal_q - genbutsu_q
                # Allow up to q_diff threshold based on safety weight
                threshold = 0.15 * (1.0 - safety_weight)
                if q_diff < threshold:
                    return genbutsu_action

        # Collect available discard candidates
        available = [idx for idx in DISCARD_INDICES
                     if idx < len(mask) and mask[idx]]
        if not available:
            return mortal_action

        # Score each available discard: combined Q-value and safety
        # Use robust normalization: percentile-based instead of min-max
        q_vals = [q_values[idx] for idx in available]
        q_vals_sorted = sorted(q_vals)
        n = len(q_vals_sorted)
        if n <= 1:
            return mortal_action

        # Use 10th and 90th percentile for robust normalization
        q_low = q_vals_sorted[max(0, n // 10)]
        q_high = q_vals_sorted[min(n - 1, n - 1 - n // 10)]
        q_range = max(q_high - q_low, 0.01)

        candidates = []
        for idx in available:
            tile_name = ACTION_TILE_NAMES[idx]
            danger = aggregate_danger(tile_name, ctx)
            # Normalize danger to 0-1 using context-aware max
            # Riichi present: max danger ~ 1.8; no riichi: max ~ 1.2
            max_danger = 1.8 if any(ctx.riichi_flags) else 1.2
            safety = 1.0 - min(danger / max_danger, 1.0)

            # Robust Q-value normalization
            q_norm = max(0.0, min(1.0, (q_values[idx] - q_low) / q_range))

            combined = (1.0 - safety_weight) * q_norm + safety_weight * safety
            candidates.append((idx, combined))

        if not candidates:
            return mortal_action

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _find_best_discard(
        self,
        q_values: List[float],
        mask: List[bool],
    ) -> Optional[int]:
        """Find the best discard action by Q-value."""
        best_idx = None
        best_q = float('-inf')
        for idx in DISCARD_INDICES:
            if idx >= len(mask) or not mask[idx]:
                continue
            if q_values[idx] > best_q:
                best_q = q_values[idx]
                best_idx = idx
        return best_idx

    def _find_genbutsu_discard(self, mask: List[bool]) -> Optional[int]:
        """Find a genbutsu (現物/safe tile) discard from available options.

        Genbutsu = tiles that appear in riichi players' rivers (completely safe).
        Among multiple genbutsu, prefer the one that's safe against the most opponents.
        """
        gs = self.gs
        # Collect genbutsu sets per riichi player
        genbutsu_tiles: Dict[str, int] = {}  # tile_name -> count of riichi players it's safe against
        for i, p in enumerate(gs.players):
            if i == gs.player_id:
                continue
            if not p.riichi_declared:
                continue
            for tile, _ in p.river:
                base = tile_base(tile)
                genbutsu_tiles[base] = genbutsu_tiles.get(base, 0) + 1

        if not genbutsu_tiles:
            return None

        # Find available discards that are genbutsu
        best_idx = None
        best_safe_count = 0
        for idx in DISCARD_INDICES:
            if idx >= len(mask) or not mask[idx]:
                continue
            tile_name = ACTION_TILE_NAMES[idx]
            safe_count = genbutsu_tiles.get(tile_name, 0)
            if safe_count > best_safe_count:
                best_safe_count = safe_count
                best_idx = idx

        return best_idx

    def _build_safety_context(self) -> Optional[SafetyContext]:
        """Build a SafetyContext from our GameState for danger evaluation."""
        gs = self.gs
        try:
            riichi_early = {}
            for i, p in enumerate(gs.players):
                if p.riichi_declared and p.riichi_turn >= 0:
                    riichi_early[i] = p.riichi_turn

            return SafetyContext(
                riichi_flags=gs.riichi_flags,
                rivers=gs.rivers_dict,
                my_index=gs.player_id,
                remaining_tiles=gs.remaining_tiles,
                dealer=gs.dealer,
                dora_indicators=gs.dora_indicators,
                my_tiles=gs.my_hand,
                riichi_early_turns=riichi_early if riichi_early else None,
            )
        except Exception:
            return None
