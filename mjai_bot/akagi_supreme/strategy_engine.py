# -*- coding: utf-8 -*-
"""Strategy engine: the brain of akagi_supreme.

Wraps Mortal's neural network decisions with strategic overlays:
1. Game state tracking
2. Push/fold evaluation (binary: PUSH, MAWASHI, or FOLD)
3. Placement-aware adjustments
4. Meld/riichi decision overrides
5. Safe tile selection (genbutsu → suji → one-chance)

The engine RESPECTS Mortal's Q-values as the primary signal and only
adjusts when strategic context clearly demands it.

Design principles:
- Minimize overrides. Mortal's NN already encodes most tactical knowledge.
- PUSH = trust Mortal completely. No safety blending (avoids double-counting).
- FOLD = full betaori using pure safety logic (genbutsu → suji → kabe).
- MAWASHI = among safe-enough tiles, pick the one with best Q-value.
- Never force hora when Mortal chose otherwise (着順 reasons).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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

from akagi.logging_utils import setup_logger

logger = setup_logger("akagi_supreme_strategy")


# Action tile names (shared between 4P and 3P — tiles 0-36 are the same)
ACTION_TILE_NAMES = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
]

DISCARD_INDICES = set(range(37))  # 0-36 are all tile discards


@dataclass
class ActionConfig:
    """Action index configuration for different player counts.

    4P action space (46): 0-36 tiles, 37 reach, 38-40 chi, 41 pon, 42 kan,
                          43 hora, 44 ryukyoku, 45 none
    3P action space (44): 0-36 tiles, 37 reach, 38 pon, 39 kan,
                          40 hora, 41 ryukyoku, 42 none, 43 kita
    """
    idx_reach: int
    idx_pon: int
    idx_kan: int
    idx_hora: int
    idx_ryukyoku: int
    idx_none: int
    chi_indices: frozenset
    meld_indices: frozenset = field(init=False)

    def __post_init__(self):
        self.meld_indices = self.chi_indices | {self.idx_pon, self.idx_kan}


ACTION_CONFIG_4P = ActionConfig(
    idx_reach=37, idx_pon=41, idx_kan=42, idx_hora=43,
    idx_ryukyoku=44, idx_none=45,
    chi_indices=frozenset({38, 39, 40}),
)

ACTION_CONFIG_3P = ActionConfig(
    idx_reach=37, idx_pon=38, idx_kan=39, idx_hora=40,
    idx_ryukyoku=41, idx_none=42,
    chi_indices=frozenset(),  # no chi in 3-player mahjong
)

# Default aliases for backwards compatibility (4P)
IDX_REACH = ACTION_CONFIG_4P.idx_reach
IDX_CHI_LOW = 38
IDX_CHI_MID = 39
IDX_CHI_HIGH = 40
IDX_PON = ACTION_CONFIG_4P.idx_pon
IDX_KAN = ACTION_CONFIG_4P.idx_kan
IDX_HORA = ACTION_CONFIG_4P.idx_hora
IDX_RYUKYOKU = ACTION_CONFIG_4P.idx_ryukyoku
IDX_NONE = ACTION_CONFIG_4P.idx_none
MELD_INDICES = ACTION_CONFIG_4P.meld_indices

# Safety thresholds for tile categorization
DANGER_SAFE = 0.25      # genbutsu-level safe
DANGER_MODERATE = 0.60  # suji/kabe level
DANGER_HIGH = 1.0       # dangerous


class StrategyEngine:
    """Applies strategic overlays to Mortal's raw action selection."""

    def __init__(self, action_config: ActionConfig = None):
        self.gs = GameState()
        self._last_shanten: int = 6
        self.ac = action_config or ACTION_CONFIG_4P

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

        Binary approach:
        - PUSH: trust Mortal completely (return mortal_action as-is)
        - MAWASHI: for discards, pick safest among safe-enough tiles using
                   Q-value tiebreaking. For non-discards, trust Mortal.
        - FOLD: full betaori - pick the absolute safest tile.

        Returns:
            Adjusted action index.
        """
        if not self.gs._initialized:
            return mortal_action

        ac = self.ac

        # === Hora handling ===
        # If Mortal chose hora, take it.
        if mortal_action == ac.idx_hora:
            return mortal_action

        # If hora is available but Mortal declined: respect Mortal in most
        # cases (着順 reasons — e.g. cheap ron that drops placement).
        # Exception: all-last 4th MUST take any agari. 4th is already the
        # worst outcome, so any win is an improvement regardless of value.
        if (ac.idx_hora < len(mask) and mask[ac.idx_hora]
                and gs.is_all_last and gs.my_placement == 4):
            return ac.idx_hora

        # === Evaluate strategic context ===
        acceptance = self.gs.estimate_acceptance_count()
        pf_result = evaluate_push_fold(self.gs, self._last_shanten, acceptance)
        pf_result = adjust_for_placement(pf_result, self.gs)
        p_adj = compute_placement_adjustment(self.gs)

        # === Pass override: check if we should meld when Mortal passes ===
        if mortal_action == ac.idx_none:
            override = self._check_pass_override(q_values, mask, p_adj, pf_result)
            if override is not None:
                return override
            return mortal_action

        # === Riichi decision ===
        if mortal_action == ac.idx_reach:
            return self._adjust_riichi(q_values, mask, p_adj, acceptance)

        # === Meld decisions (chi/pon/kan) ===
        if mortal_action in ac.meld_indices:
            return self._adjust_meld(mortal_action, q_values, mask, p_adj, pf_result)

        # === Discard decisions ===
        if mortal_action in DISCARD_INDICES:
            return self._adjust_discard(mortal_action, q_values, mask, pf_result)

        return mortal_action

    def _check_pass_override(
        self,
        q_values: List[float],
        mask: List[bool],
        p_adj: PlacementAdjustment,
        pf_result: PushFoldResult,
    ) -> Optional[int]:
        """Check if we should override Mortal's pass (IDX_NONE) with a meld.

        Only in desperate situations (all-last 4th, etc.) where placement
        demands speed. Even then, only override when meld Q-values are close
        to pass Q-values (Mortal almost chose to meld).
        """
        gs = self.gs

        if not gs.is_all_last:
            return None
        if gs.my_placement <= 1:
            return None

        ac = self.ac
        available_melds = [idx for idx in ac.meld_indices
                           if idx < len(mask) and mask[idx]]
        if not available_melds:
            return None

        best_meld = max(available_melds, key=lambda idx: q_values[idx])
        none_q = q_values[ac.idx_none] if ac.idx_none < len(q_values) else 0.0
        meld_q = q_values[best_meld]

        # All-last 4th: take ANY meld (worst placement, nothing to lose).
        # Top players know 4th is already the worst outcome, so every
        # chance to speed up the hand is worth taking. No Q-value gate.
        if gs.my_placement == 4:
            return best_meld

        # All-last 3rd with 4th close: need speed to stay safe
        if gs.my_placement == 3 and gs.diff_to_below < 4000:
            if p_adj.meld_multiplier >= 1.0:
                if meld_q >= none_q - 0.10:
                    return best_meld

        # All-last 3rd trying to climb to 2nd: meld for speed when
        # 2nd is within reach (e.g. small han gap via direct hit)
        if gs.my_placement == 3 and gs.diff_to_above <= 8000:
            pts_for_2nd = gs.points_needed_for_placement(2)
            han_needed = gs.min_han_for_points(pts_for_2nd, is_tsumo=False)
            if han_needed <= 3 and p_adj.meld_multiplier >= 1.0:
                if meld_q >= none_q - 0.10:
                    return best_meld

        # All-last 2nd trying to climb to 1st via fast agari
        if gs.my_placement == 2 and gs.diff_to_above <= 4000:
            if p_adj.meld_multiplier >= 1.0:
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
        ac = self.ac
        hand_value = estimate_hand_value(self.gs)

        if should_damaten(self.gs, p_adj, hand_value=hand_value,
                          acceptance_count=acceptance_count):
            best_discard = self._find_best_discard(q_values, mask)
            if best_discard is not None:
                return best_discard
            return ac.idx_reach

        # Apply riichi multiplier to Q-value comparison
        if p_adj.riichi_multiplier < 0.8:
            best_discard = self._find_best_discard(q_values, mask)
            if best_discard is not None:
                riichi_q = q_values[ac.idx_reach] if ac.idx_reach < len(q_values) else float('-inf')
                discard_q = q_values[best_discard]
                q_diff = riichi_q - discard_q
                if q_diff < 0.05:
                    return best_discard

        return ac.idx_reach

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
        - Fold mode: skip chi, be cautious with pon
        - Yakuhai/dora pon awareness: always accept valuable pons
        - Placement-driven meld encouragement
        """
        gs = self.gs
        ac = self.ac

        # === Always accept yakuhai pon (役牌ポン) ===
        if mortal_action == ac.idx_pon:
            # Check if the pon target is a yakuhai
            # We infer from Mortal choosing pon — check if it's a yakuhai tile
            # by looking at what tile was just discarded (most recent in an
            # opponent's river)
            pon_tile = self._get_last_opponent_discard()
            if pon_tile and gs.is_my_yakuhai(tile_base(pon_tile)):
                return mortal_action  # always pon yakuhai

            # Check if pon target is a dora
            if pon_tile:
                base = tile_base(pon_tile)
                if base in gs.doras or pon_tile.endswith("r"):
                    return mortal_action  # always pon dora

        # === Menzen loss evaluation (門前維持価値) ===
        # When hand is closed and near tenpai, melding loses riichi eligibility.
        # Penalize meld unless the Q-value advantage is strong enough to
        # compensate for the lost option value of declaring riichi.
        if not gs.my_melds and ac.idx_none < len(mask) and mask[ac.idx_none]:
            menzen_penalty = self._evaluate_menzen_loss(mortal_action)
            if menzen_penalty > 0.0:
                meld_q = q_values[mortal_action]
                none_q = q_values[ac.idx_none]
                # Require meld Q-value to exceed pass by the menzen penalty
                if meld_q < none_q + menzen_penalty:
                    return ac.idx_none

        # === In FOLD: skip chi entirely, very cautious with pon ===
        # Exception: all-last 4th should never fold — meld for speed.
        if pf_result.decision == Decision.FOLD and not (gs.is_all_last and gs.my_placement == 4):
            if mortal_action in ac.chi_indices:
                if ac.idx_none < len(mask) and mask[ac.idx_none]:
                    return ac.idx_none
            if mortal_action == ac.idx_pon:
                if ac.idx_none < len(mask) and mask[ac.idx_none]:
                    pon_q = q_values[mortal_action]
                    none_q = q_values[ac.idx_none]
                    if pon_q < none_q + 0.15:
                        return ac.idx_none

        # === In MAWASHI: slight bias against chi ===
        if pf_result.decision == Decision.MAWASHI:
            if mortal_action in ac.chi_indices:
                if ac.idx_none < len(mask) and mask[ac.idx_none]:
                    chi_q = q_values[mortal_action]
                    none_q = q_values[ac.idx_none]
                    q_diff = chi_q - none_q
                    threshold = 0.05 if p_adj.meld_multiplier >= 1.0 else 0.10
                    if q_diff < threshold:
                        return ac.idx_none

        # Apply meld multiplier for general discouragement
        if p_adj.meld_multiplier < 0.85:
            meld_q = q_values[mortal_action]
            none_available = ac.idx_none < len(mask) and mask[ac.idx_none]
            if none_available:
                none_q = q_values[ac.idx_none]
                q_diff = meld_q - none_q
                threshold = 0.05 * (1.0 - p_adj.meld_multiplier) / 0.15
                if q_diff < threshold:
                    return ac.idx_none

        return mortal_action

    def _evaluate_menzen_loss(self, mortal_action: int) -> float:
        """Evaluate the Q-value penalty for losing menzen (riichi option).

        When the hand is closed and near tenpai, melding sacrifices:
        1. Riichi declaration (ippatsu + ura dora + intimidation)
        2. Menzen tsumo yaku
        3. Higher scoring potential (pinfu, etc.)

        Returns a Q-value threshold the meld must exceed to be worthwhile.
        0.0 means no penalty (meld freely).
        """
        gs = self.gs
        ac = self.ac

        # Already has melds — no menzen to lose
        if gs.my_melds:
            return 0.0

        # All-last 4th: speed is everything, no menzen penalty.
        # 4th is already the worst outcome; meld for any chance to climb.
        if gs.is_all_last and gs.my_placement == 4:
            return 0.0

        # Chi breaks menzen more than pon in terms of hand structure options
        is_chi = mortal_action in ac.chi_indices

        # Shanten-based penalty: closer to tenpai = bigger cost of melding
        shanten = self._last_shanten

        if shanten <= 0:
            # Already tenpai menzen — very high cost to break it
            # (loses riichi option entirely)
            return 0.20 if is_chi else 0.15

        if shanten == 1:
            # Iishanten menzen — moderate cost (close to riichi)
            # Early game: riichi is very strong, penalize more
            # Late game: speed matters more, penalize less
            if gs.my_turn <= 8:
                return 0.10 if is_chi else 0.07
            else:
                return 0.05 if is_chi else 0.03

        if shanten == 2:
            # Ryanshanten — small cost, melding for speed is often fine
            return 0.03 if is_chi else 0.01

        # Far from tenpai — no meaningful menzen advantage
        return 0.0

    def _adjust_discard(
        self,
        mortal_action: int,
        q_values: List[float],
        mask: List[bool],
        pf_result: PushFoldResult,
    ) -> int:
        """Adjust tile discard selection based on push/fold decision.

        Binary approach (no blending):
        - PUSH: trust Mortal completely (return mortal_action)
        - MAWASHI: pick safest tile among "safe enough" candidates,
                   using Q-value for tiebreaking
        - FOLD: pick the absolute safest tile (full betaori)
        """
        # PUSH: trust Mortal 100%
        if pf_result.decision == Decision.PUSH:
            return mortal_action

        # Build safety context
        ctx = self._build_safety_context()
        if ctx is None:
            return mortal_action  # can't evaluate safety, trust Mortal

        # Get available discard candidates with danger scores
        candidates = self._score_discards_by_safety(mask, ctx)
        if not candidates:
            return mortal_action

        # === FOLD: pure betaori — pick the safest tile ===
        if pf_result.decision == Decision.FOLD:
            # First try genbutsu
            genbutsu = self._find_genbutsu_discard(mask)
            if genbutsu is not None:
                return genbutsu
            # Otherwise pick lowest danger tile
            candidates.sort(key=lambda x: x[1])  # sort by danger ascending
            return candidates[0][0]

        # === MAWASHI: safe tiles first, Q-value tiebreak ===
        # 1. Categorize tiles by safety level
        safe_tiles = [(idx, d) for idx, d in candidates if d <= DANGER_SAFE]
        moderate_tiles = [(idx, d) for idx, d in candidates if DANGER_SAFE < d <= DANGER_MODERATE]
        dangerous_tiles = [(idx, d) for idx, d in candidates if d > DANGER_MODERATE]

        # 2. Pick from the safest available category, using Q-value for tiebreak
        if safe_tiles:
            # Among safe tiles, pick the one with best Q-value (maintain hand)
            return max(safe_tiles, key=lambda x: q_values[x[0]])[0]

        if moderate_tiles:
            # Among moderate tiles, prefer Q-value but lean toward safety
            # Pick the tile with best Q-value among the safer half
            moderate_tiles.sort(key=lambda x: x[1])  # sort by danger
            safer_half = moderate_tiles[:max(1, len(moderate_tiles) // 2 + 1)]
            return max(safer_half, key=lambda x: q_values[x[0]])[0]

        # Only dangerous tiles available: just pick the least dangerous
        candidates.sort(key=lambda x: x[1])
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

        Genbutsu sources (in order of reliability):
        1. Tiles in riichi players' own rivers (completely safe)
        2. Tiles discarded by others after riichi that weren't ron'd
           (post_riichi_safe — also completely safe since hand is locked)

        Priority among genbutsu:
        1. Tiles safe against the most riichi players
        2. Among ties, prefer tiles that don't break hand structure
           (isolated tiles > connected tiles)
        """
        gs = self.gs
        genbutsu_tiles: Dict[str, int] = {}
        for i, p in enumerate(gs.players):
            if i == gs.player_id:
                continue
            if not p.riichi_declared:
                continue
            # Tiles in the riichi player's own river
            for tile, _ in p.river:
                base = tile_base(tile)
                genbutsu_tiles[base] = genbutsu_tiles.get(base, 0) + 1
            # Tiles others discarded after this player's riichi (passed on)
            for base in p.post_riichi_safe:
                genbutsu_tiles[base] = genbutsu_tiles.get(base, 0) + 1

        if not genbutsu_tiles:
            return None

        # Find available discards that are genbutsu
        # Normalize tile names: "5mr" → "5m" for lookup
        genbutsu_options = []
        for idx in DISCARD_INDICES:
            if idx >= len(mask) or not mask[idx]:
                continue
            tile_name = ACTION_TILE_NAMES[idx]
            lookup_name = tile_base(tile_name)  # normalize red tiles
            safe_count = genbutsu_tiles.get(lookup_name, 0)
            if safe_count > 0:
                isolation = self._tile_isolation_score(tile_name)
                genbutsu_options.append((idx, safe_count, isolation))

        if not genbutsu_options:
            return None

        # Pick best: most safe players first, then most isolated
        genbutsu_options.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return genbutsu_options[0][0]

    def _tile_isolation_score(self, tile_name: str) -> float:
        """Score how isolated a tile is in our hand (higher = more isolated = better to discard).

        Tiles that don't contribute to hand progression are better fold discards.
        """
        gs = self.gs
        hand = gs.my_hand
        if not hand:
            return 0.0

        # Check how many tiles in hand connect to this tile
        connections = 0
        s, r, _ = parse_tile(tile_name)
        if r is None:
            # Honor tile: count copies
            count = sum(1 for t in hand if tile_base(t) == tile_name)
            return 1.0 if count <= 1 else 0.0  # isolated honor = good to discard

        for t in hand:
            ts, tr, _ = parse_tile(t)
            if tr is None:
                continue
            if ts == s and abs(tr - r) <= 2:
                connections += 1

        # More connections = less isolated = worse to discard when folding
        if connections <= 1:
            return 1.0  # isolated
        if connections <= 2:
            return 0.5
        return 0.0  # well-connected

    def _score_discards_by_safety(
        self, mask: List[bool], ctx: SafetyContext
    ) -> List[Tuple[int, float]]:
        """Score all available discards by danger level.

        Returns list of (action_index, danger_score).
        """
        candidates = []
        for idx in DISCARD_INDICES:
            if idx >= len(mask) or not mask[idx]:
                continue
            tile_name = ACTION_TILE_NAMES[idx]
            danger = aggregate_danger(tile_name, ctx)
            candidates.append((idx, danger))
        return candidates

    def _get_last_opponent_discard(self) -> Optional[str]:
        """Get the most recently discarded tile by an opponent.

        Uses tracked state instead of river length heuristic, which was
        incorrect (longest river != most recent discard).
        """
        return self.gs._last_discard_tile

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
