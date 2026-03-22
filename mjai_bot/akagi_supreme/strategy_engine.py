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
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from ..strategy.safety import (
    SafetyContext, aggregate_danger, bucketize,
    parse_tile, is_honor, only_tiles, SUITS
)
from .game_state import GameState, Tile
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

        # === Always take winning moves ===
        if mortal_action == IDX_HORA:
            return mortal_action

        # === Always allow pass when Mortal says pass ===
        if mortal_action == IDX_NONE:
            return mortal_action

        # === Evaluate strategic context ===
        pf_result = evaluate_push_fold(self.gs, self._last_shanten)
        pf_result = adjust_for_placement(pf_result, self.gs)
        p_adj = compute_placement_adjustment(self.gs)

        # === Hora override: if we can win, check if we should ===
        if mask[IDX_HORA] if IDX_HORA < len(mask) else False:
            # Almost always take the win
            # Exception: all-last 1st place with huge lead, tsumo might cause
            # others to lose more? No, always take wins.
            return IDX_HORA

        # === Riichi decision ===
        if mortal_action == IDX_REACH:
            return self._adjust_riichi(q_values, mask, p_adj)

        # === Meld decisions (chi/pon/kan) ===
        if mortal_action in MELD_INDICES:
            return self._adjust_meld(mortal_action, q_values, mask, p_adj, pf_result)

        # === Discard decisions ===
        if mortal_action in DISCARD_INDICES:
            return self._adjust_discard(mortal_action, q_values, mask, pf_result, p_adj)

        return mortal_action

    def _adjust_riichi(
        self,
        q_values: List[float],
        mask: List[bool],
        p_adj: PlacementAdjustment,
    ) -> int:
        """Decide whether to declare riichi or damaten."""
        hand_value = estimate_hand_value(self.gs)

        # Check if damaten is strategically preferred (passes hand_value for cost analysis)
        if should_damaten(self.gs, p_adj, hand_value=hand_value):
            # Find the best discard instead of riichi
            best_discard = self._find_best_discard(q_values, mask, safety_weight=0.0)
            if best_discard is not None:
                return best_discard
            # Fall back to riichi if no discard available
            return IDX_REACH

        # Apply riichi multiplier to Q-value comparison
        if p_adj.riichi_multiplier < 0.8:
            # Strongly discouraged: compare adjusted Q-values
            best_discard = self._find_best_discard(q_values, mask, safety_weight=0.0)
            if best_discard is not None:
                riichi_q = q_values[IDX_REACH] if IDX_REACH < len(q_values) else float('-inf')
                discard_q = q_values[best_discard]
                # Require discard Q-value to be at least 85% of riichi Q-value
                # (was 70% — too lenient, would damaten too often)
                if discard_q >= riichi_q * 0.85:
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
        """Decide whether to accept or decline a meld opportunity."""
        if pf_result.decision == Decision.FOLD:
            # In full fold mode: only accept pon/kan of safe tiles, skip chi
            if mortal_action in {IDX_CHI_LOW, IDX_CHI_MID, IDX_CHI_HIGH}:
                if mask[IDX_NONE] if IDX_NONE < len(mask) else False:
                    return IDX_NONE  # pass on chi while folding

        # Apply meld multiplier
        if p_adj.meld_multiplier < 0.85:
            # Discouraged melding: compare Q-values
            meld_q = q_values[mortal_action]
            none_q = q_values[IDX_NONE] if (IDX_NONE < len(q_values) and
                                             (mask[IDX_NONE] if IDX_NONE < len(mask) else False)) else float('-inf')
            adjusted_meld_q = meld_q * p_adj.meld_multiplier
            if none_q > adjusted_meld_q:
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
        """Adjust tile discard selection with safety consideration."""
        safety_weight = pf_result.safety_weight + p_adj.extra_safety
        safety_weight = max(0.0, min(0.85, safety_weight))

        # If no safety concern, trust Mortal completely
        if safety_weight <= 0.02:
            return mortal_action

        # Build safety context for danger evaluation
        ctx = self._build_safety_context()
        if ctx is None:
            return mortal_action

        # Score each available discard: combined Q-value and safety
        candidates = []
        q_max = max(q_values[i] for i in range(len(q_values)) if i in DISCARD_INDICES and
                     (mask[i] if i < len(mask) else False))
        q_min = min(q_values[i] for i in range(len(q_values)) if i in DISCARD_INDICES and
                     (mask[i] if i < len(mask) else False))
        q_range = max(q_max - q_min, 0.01)

        for idx in DISCARD_INDICES:
            if idx >= len(mask) or not mask[idx]:
                continue

            tile_name = ACTION_TILE_NAMES[idx]
            danger = aggregate_danger(tile_name, ctx)
            safety = 1.0 - min(danger / 1.8, 1.0)  # normalize 0-1

            # Normalize Q-value to 0-1 range
            q_norm = (q_values[idx] - q_min) / q_range

            # Combined score
            combined = (1.0 - safety_weight) * q_norm + safety_weight * safety
            candidates.append((idx, combined, q_norm, safety, danger))

        if not candidates:
            return mortal_action

        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_idx = candidates[0][0]

        return best_idx

    def _find_best_discard(
        self,
        q_values: List[float],
        mask: List[bool],
        safety_weight: float,
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
