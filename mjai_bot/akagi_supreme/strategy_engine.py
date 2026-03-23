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
- Hora: trust Mortal's decision (Mortal already encodes 着順 conditions).
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


# Per-tile index to tile name for wait display
INDEX_TO_TILE = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]


class StrategyEngine:
    """Applies strategic overlays to Mortal's raw action selection."""

    def __init__(self, action_config: ActionConfig = None):
        self.gs = GameState()
        self._last_shanten: int = 6
        self.ac = action_config or ACTION_CONFIG_4P
        self.last_thought: List[str] = []

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
        self.last_thought.clear()
        
        if not self.gs._initialized:
            self.last_thought.append("【システム】まだゲームが始まっていないので、当面はMortalの基本判断に従います。")
            return mortal_action

        ac = self.ac

        # === Hora handling ===
        if mortal_action == ac.idx_hora:
            self.last_thought.append("【決断】アガリのチャンス！迷わず和了（アガリ）を選択します。")
            return mortal_action

        # If hora is available but Mortal declined: respect Mortal in most
        # cases (着順 reasons — e.g. cheap ron that drops placement).
        # Exceptions where top players ALWAYS take agari:
        # 1. All-last 4th: worst outcome, any win is an improvement
        # 2. All-last 1st: ending the game preserves 1st place
        # 3. All-last 2nd/3rd: agari can only improve or maintain placement
        # All-last: whether to take agari or not depends heavily on placement conditions.
        # Mortal is already trained on these conditions and will output PASS (NONE)
        # instead of HORA if an agari locks in a suboptimal placement. 
        # We trust Mortal's decision completely for Hora.

        # === Evaluate strategic context ===
        acceptance = self.gs.estimate_acceptance_count()
        pf_result = evaluate_push_fold(self.gs, self._last_shanten, acceptance)
        pf_result = adjust_for_placement(pf_result, self.gs, self._last_shanten)
        p_adj = compute_placement_adjustment(self.gs)
        
        # Placement info formatting
        placement_text = f"現在 {self.gs.my_placement}着 ({self.gs.my_score}点)"
        if self.gs.my_placement > 1:
            placement_text += f"で、上位とは {self.gs.diff_to_above}点差"
        if self.gs.my_placement < 4:
            placement_text += f"、下位とは {self.gs.diff_to_below}点差"
        if self.gs.is_all_last:
            placement_text += "のオーラスです。"
        else:
            placement_text += "です。"

        round_text = self._format_round_info()
        self.last_thought.append(f"【状況】{round_text}{placement_text}")

        # === 手牌構成の詳細 ===
        threat = self.gs.max_opponent_threat()
        from .push_fold import estimate_hand_value, estimate_risk_of_deal_in
        hand_value = estimate_hand_value(self.gs)

        shanten_text = f"あと{self._last_shanten}シャンテン" if self._last_shanten > 0 else "テンパイ"
        self.last_thought.append(f"【手牌】{shanten_text}（有効牌: 約{acceptance}枚、想定打点: {hand_value:.0f}点）")
        
        # 手牌構成の詳細分析
        hand_detail = self._format_hand_composition(hand_value)
        if hand_detail:
            self.last_thought.append(f"  └ {hand_detail}")

        # テンパイ時の待ち情報
        if self._last_shanten <= 0:
            wait_info = self._format_wait_info()
            if wait_info:
                self.last_thought.append(f"【待ち】{wait_info}")

        # === 脅威分析 ===
        threat_detail = self._format_threat_analysis()
        if threat_detail:
            self.last_thought.append(f"【脅威分析】{threat_detail}")

        # === Push/Fold判断の詳細 ===
        decision_map = {"PUSH": "攻め(PUSH)", "FOLD": "ベタオリ(FOLD)", "MAWASHI": "回し打ち(MAWASHI)"}
        decision_jp = decision_map.get(pf_result.decision.name, "進める")
        pf_reason_jp = self._translate_pf_reason(pf_result.reason)
        
        risk = estimate_risk_of_deal_in(self.gs)
        self.last_thought.append(
            f"【判断】{decision_jp}（確信度: {pf_result.confidence:.0%}）"
        )
        self.last_thought.append(
            f"  └ 理由: {pf_reason_jp}"
        )
        if threat > 0.0:
            self.last_thought.append(
                f"  └ 放銃リスク: 約{risk:.0f}点 vs 自手価値: 約{hand_value:.0f}点"
            )

        if p_adj.riichi_multiplier != 1.0 or p_adj.meld_multiplier != 1.0:
            adj_parts = []
            if p_adj.riichi_multiplier != 1.0:
                adj_parts.append(f"リーチ {p_adj.riichi_multiplier:.2f}倍")
            if p_adj.meld_multiplier != 1.0:
                adj_parts.append(f"鳴き {p_adj.meld_multiplier:.2f}倍")
            adj_reason = p_adj.reason if p_adj.reason else "順位調整"
            self.last_thought.append(f"【順位調整】{', '.join(adj_parts)}（{adj_reason}）")

        # === Pass override: check if we should meld when Mortal passes ===
        if mortal_action == ac.idx_none:
            override = self._check_pass_override(q_values, mask, p_adj, pf_result)
            if override is not None:
                self.last_thought.append(f"【決断】Mortalは「スルー(PASS)」を推奨していますが、ここは順位と速度を最優先して「鳴き({override})」に切り替えます。")
                return override
            self.last_thought.append("【決断】無理に鳴いたりはせず、ここはスルー(PASS)を選択します。")
            return mortal_action

        # === Riichi decision ===
        if mortal_action == ac.idx_reach:
            action = self._adjust_riichi(q_values, mask, p_adj, acceptance)
            if action != ac.idx_reach:
                self.last_thought.append(f"【決断】Mortalは「リーチ」を推奨しましたが、順位状況と打点効率を考慮して、あえてダマテン（打 {action}）に構えます。")
            else:
                self.last_thought.append("【決断】勝負の時です！ここでリーチを宣言します！")
            return action

        # === Meld decisions (chi/pon/kan) ===
        if mortal_action in ac.meld_indices:
            action = self._adjust_meld(mortal_action, q_values, mask, p_adj, pf_result)
            if action != mortal_action:
                self.last_thought.append(f"【決断】鳴きのチャンスですが、門前を維持するか守備に回りたいので、今回は見送ります（-> Action {action}）。")
            else:
                self.last_thought.append("【決断】ここは鳴いて手を進めます！")
            return action

        # === Discard decisions ===
        if mortal_action in DISCARD_INDICES:
            override = self._check_riichi_override(mortal_action, q_values, mask, p_adj, hand_value, acceptance)
            if override is not None:
                action = override
                self.last_thought.append("【決断】Mortalは普通の打牌を推奨しましたが、攻撃的にいくべき場面なので強制的にリーチに踏み切ります！")
            else:
                action = self._adjust_discard(mortal_action, q_values, mask, pf_result)
                
            if action != mortal_action and override is None:
                tile_orig = ACTION_TILE_NAMES[mortal_action] if mortal_action < len(ACTION_TILE_NAMES) else str(mortal_action)
                tile_new = ACTION_TILE_NAMES[action] if action < len(ACTION_TILE_NAMES) else str(action)
                self.last_thought.append(f"【決断】Mortalの推奨は「{tile_orig}」ですが、より安全性を高めるため「{tile_new}」を切り飛ばします。")
            elif override is None:
                tile = ACTION_TILE_NAMES[mortal_action] if mortal_action < len(ACTION_TILE_NAMES) else str(mortal_action)
                self.last_thought.append(f"【決断】素直に「{tile}」を切るのが一番良さそうですね。")
            return action

        self.last_thought.append(f"【決断】その他のアクション（{mortal_action}）を選択しました。")
        return mortal_action

    def _check_riichi_override(
        self,
        mortal_action: int,
        q_values: List[float],
        mask: List[bool],
        p_adj: PlacementAdjustment,
        hand_value: float = 0.0,
        acceptance_count: int = 0,
    ) -> Optional[int]:
        """Override discard with riichi if strategy demands aggression."""
        ac = self.ac
        if ac.idx_reach >= len(mask) or not mask[ac.idx_reach]:
            return None

        if p_adj.riichi_multiplier >= 1.05 and not should_damaten(
            self.gs, p_adj, hand_value=hand_value, acceptance_count=acceptance_count
        ):
            riichi_q = q_values[ac.idx_reach]
            discard_q = q_values[mortal_action]
            # If riichi is close and multiplier boosts it past discard, override
            # Mortal Q-values can be negative, so we must divide negative values to boost them
            adjusted_riichi_q = riichi_q * p_adj.riichi_multiplier if riichi_q > 0 else riichi_q / p_adj.riichi_multiplier
            if adjusted_riichi_q >= discard_q - 0.05:
                return ac.idx_reach
                
        return None

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

        # All-last 4th: aggressively take melds (worst placement, nothing to lose)
        # ONLY IF the meld doesn't ruin the point condition (meld_multiplier >= 1.0).
        if gs.my_placement == 4:
            if p_adj.meld_multiplier >= 1.0:
                # Do not force meld if the Q-value is catastrophically low (e.g., breaks yaku/form)
                if meld_q >= none_q - 0.15:
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
        # Only override if 3rd is far enough (>= 4000) to avoid risking 2nd place
        # Top players balance "climb to 1st" vs "protect 2nd" — when 3rd is close,
        # a careless meld that leads to deal-in could drop us to 3rd.
        if gs.my_placement == 2 and gs.diff_to_above <= 4000:
            if gs.diff_to_below >= 4000 and p_adj.meld_multiplier >= 1.0:
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

        # Apply riichi multiplier to Q-value comparison.
        # When placement strategy discourages riichi (multiplier < 0.8),
        # discount riichi's Q-value before comparing to damaten.
        # Without this, raw Q comparison almost never overrides Mortal's
        # riichi choice, making the riichi_multiplier ineffective.
        if p_adj.riichi_multiplier < 0.8:
            best_discard = self._find_best_discard(q_values, mask)
            if best_discard is not None:
                riichi_q = q_values[ac.idx_reach] if ac.idx_reach < len(q_values) else float('-inf')
                discard_q = q_values[best_discard]
                # Discount riichi Q-value by placement multiplier (handle negative Q correctly)
                adjusted_riichi_q = riichi_q * p_adj.riichi_multiplier if riichi_q > 0 else riichi_q / p_adj.riichi_multiplier
                if adjusted_riichi_q < discard_q + 0.05:
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
        - Fold mode: skip chi, be cautious with pon (checked FIRST)
        - Yakuhai/dora pon awareness: accept valuable pons when NOT folding
        - Placement-driven meld encouragement
        """
        gs = self.gs
        ac = self.ac

        # === In FOLD: reject melds first (top priority except all-last 4th) ===
        # Top players in full betaori do NOT meld — even yakuhai pon.
        # Taking a pon while folding opens the hand, commits to attack mode,
        # and risks deal-in on subsequent discards. The only exception is
        # all-last 4th where there's nothing to lose.
        if pf_result.decision == Decision.FOLD and not (gs.is_all_last and gs.my_placement == 4):
            if ac.idx_none < len(mask) and mask[ac.idx_none]:
                return ac.idx_none

        # === Menzen loss evaluation (門前維持価値) ===
        # MUST be checked BEFORE yakuhai/dora pon fast-path.
        # When hand is closed and near tenpai, melding loses riichi eligibility.
        # Top players would NOT pon even yakuhai if they're already menzen
        # tenpai with a decent hand — the riichi option is too valuable.
        if not gs.my_melds and ac.idx_none < len(mask) and mask[ac.idx_none]:
            menzen_penalty = self._evaluate_menzen_loss(mortal_action)
            if menzen_penalty > 0.0:
                meld_q = q_values[mortal_action]
                none_q = q_values[ac.idx_none]
                # Require meld Q-value to exceed pass by the menzen penalty
                if meld_q < none_q + menzen_penalty:
                    return ac.idx_none

        # === Accept yakuhai pon (役牌ポン) when not folding ===
        # Only reached if menzen loss check above didn't reject the meld.
        if pf_result.decision == Decision.PUSH:
            if mortal_action == ac.idx_pon:
                pon_tile = self._get_last_opponent_discard()
                if pon_tile and gs.is_my_yakuhai(tile_base(pon_tile)):
                    return mortal_action  # pon yakuhai (menzen cost already cleared)

                # Check if pon target is a dora
                if pon_tile:
                    base = tile_base(pon_tile)
                    if base in gs.doras or pon_tile.endswith("r"):
                        return mortal_action  # pon dora (menzen cost already cleared)

        # === In MAWASHI: slight bias against chi ===
        if pf_result.decision == Decision.MAWASHI:
            if mortal_action in ac.chi_indices:
                # 着順的に速度が重要な場合（meld_multiplier >= 1.1）はチー拒否しない
                # トッププレイヤーはオーラスの切迫した状況では門前にこだわらない
                if p_adj.meld_multiplier >= 1.1:
                    pass  # チーを受理（Mortalの判断を尊重）
                elif ac.idx_none < len(mask) and mask[ac.idx_none]:
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

        is_chi = mortal_action in ac.chi_indices
        shanten = self._last_shanten

        # === 七対子ルート検出: ポンしたら七対子が崩壊するため絶対拒否 ===
        # トッププレイヤーは七対子5対子以上ならポンでは絶対に崩さない
        hand = gs.my_hand
        if hand and not is_chi:  # ポン/カンの場合のみ
            counts_for_chiitoi = {}
            for t in hand:
                tb = tile_base(t)
                counts_for_chiitoi[tb] = counts_for_chiitoi.get(tb, 0) + 1
            pair_count = sum(1 for c in counts_for_chiitoi.values() if c >= 2)
            if pair_count >= 5:
                return 0.30  # 七対子ルートの鳴きは極めて高いペナルティ

        # 予想打点によるペナルティの増減 (トッププレイヤーは高い手ほど遠くから鳴く)
        hand_value = estimate_hand_value(gs)
        
        # 満貫以上が見込めるなら門前ペナルティは無視 (速度最優先で鳴く)
        if hand_value >= 8000:
            return 0.0

        # 価値が低い(2000点未満等)場合はペナルティを強め、無意味な鳴きを強制的に抑制
        value_penalty_mult = 1.0
        if hand_value < 2000:
            value_penalty_mult = 2.5
        elif hand_value >= 3900:
            value_penalty_mult = 0.5

        if shanten <= 0:
            # Already tenpai menzen — very high cost to break it
            base = 0.20 if is_chi else 0.15
            return base * value_penalty_mult

        if shanten == 1:
            # Iishanten menzen — moderate cost (close to riichi)
            if gs.my_turn <= 8:
                base = 0.10 if is_chi else 0.07
            else:
                base = 0.05 if is_chi else 0.03
            return base * value_penalty_mult

        if shanten == 2:
            # Ryanshanten — far from riichi. Low-value melds are bad but
            # penalty should not be excessive (avoid double-scaling with value_penalty_mult).
            # Top players often do meld from ryanshanten to accelerate.
            if hand_value < 2000:
                # Low-value ryanshanten: use fixed penalty without further multiplier
                return 0.06 if is_chi else 0.04
            base = 0.03 if is_chi else 0.01
            return base * value_penalty_mult

        # Far from tenpai
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
                genbutsu_name = ACTION_TILE_NAMES[genbutsu] if genbutsu < len(ACTION_TILE_NAMES) else str(genbutsu)
                self.last_thought.append(f"【安全度分析】現物「{genbutsu_name}」を発見 → 最も安全な牌として選択")
                return genbutsu
            # Otherwise pick lowest danger tile, with isolation tiebreak
            # Among equally dangerous tiles, prefer isolated ones (less hand value loss)
            candidates.sort(key=lambda x: (x[1], -self._tile_isolation_score(ACTION_TILE_NAMES[x[0]] if x[0] < len(ACTION_TILE_NAMES) else "")))  # danger asc, isolation desc
            chosen = candidates[0]
            chosen_name = ACTION_TILE_NAMES[chosen[0]] if chosen[0] < len(ACTION_TILE_NAMES) else str(chosen[0])
            # Show top 3 candidates with danger scores
            top3 = candidates[:3]
            cand_text = ", ".join(
                f"{ACTION_TILE_NAMES[idx] if idx < len(ACTION_TILE_NAMES) else str(idx)}(危険度:{d:.2f})"
                for idx, d in top3
            )
            self.last_thought.append(f"【安全度分析】現物なし → 危険度の低い牌を選択: {cand_text}")
            return chosen[0]

        # === MAWASHI: safe tiles first, isolation & Q-value tiebreak ===
        # 1. Categorize tiles by safety level
        safe_tiles = [(idx, d) for idx, d in candidates if d <= DANGER_SAFE]
        moderate_tiles = [(idx, d) for idx, d in candidates if DANGER_SAFE < d <= DANGER_MODERATE]
        dangerous_tiles = [(idx, d) for idx, d in candidates if d > DANGER_MODERATE]

        def mawashi_score(idx):
            q = q_values[idx]
            tile_name = ACTION_TILE_NAMES[idx]
            isolation = self._tile_isolation_score(tile_name)
            return q + (isolation * 0.15)

        # 2. Pick from the safest available category, prioritizing safety over scoring
        # Top players in MAWASHI treat ALL tiles in the safe bucket equally and
        # pick the one with the best hand progression (Q-value). Limiting to only
        # the N safest tiles misses potentially much better Q-value tiles that are
        # still well within the safety threshold (all <= DANGER_SAFE = 0.25).
        if safe_tiles:
            chosen_idx = max(safe_tiles, key=lambda x: mawashi_score(x[0]))[0]
            chosen_name = ACTION_TILE_NAMES[chosen_idx] if chosen_idx < len(ACTION_TILE_NAMES) else str(chosen_idx)
            safe_tiles.sort(key=lambda x: x[1])
            top3 = safe_tiles[:3]
            safe_names = ", ".join(
                f"{ACTION_TILE_NAMES[idx] if idx < len(ACTION_TILE_NAMES) else str(idx)}(危:{d:.2f})"
                for idx, d in top3
            )
            self.last_thought.append(f"【安全度分析】安全牌あり（{len(safe_tiles)}候補）→ Q値考慮で「{chosen_name}」選択 [{safe_names}]")
            return chosen_idx

        if moderate_tiles:
            # Moderate danger range (0.25-0.60) has more variance, so balance
            # danger and Q-value: consider all moderate tiles for Q-value tiebreak
            # rather than artificially limiting the pool.
            chosen_idx = max(moderate_tiles, key=lambda x: mawashi_score(x[0]))[0]
            chosen_name = ACTION_TILE_NAMES[chosen_idx] if chosen_idx < len(ACTION_TILE_NAMES) else str(chosen_idx)
            self.last_thought.append(f"【安全度分析】安全牌なし → 中程度の危険度から「{chosen_name}」を選択")
            return chosen_idx

        # Only dangerous tiles available: pick least dangerous with isolation tiebreak
        candidates.sort(key=lambda x: (x[1], -self._tile_isolation_score(ACTION_TILE_NAMES[x[0]] if x[0] < len(ACTION_TILE_NAMES) else "")))
        chosen = candidates[0]
        chosen_name = ACTION_TILE_NAMES[chosen[0]] if chosen[0] < len(ACTION_TILE_NAMES) else str(chosen[0])
        self.last_thought.append(f"【安全度分析】安全な牌がなく危険牌のみ → 最低危険度の「{chosen_name}」(危険度:{chosen[1]:.2f})を選択")
        return chosen[0]

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
        1. Tiles safe against the most riichi players (共通安全牌を最優先)
        2. Among ties, prefer tiles safe against higher threat players
        3. Among ties, prefer tiles that don't break hand structure
           (isolated tiles > connected tiles)
        """
        gs = self.gs
        threatening_players = []
        genbutsu_per_player = {}
        
        for i, p in enumerate(gs.players):
            if i == gs.player_id:
                continue
            # リーチ者、または超高脅威（1.2以上）の鳴き手に対して現物を探す
            threat = gs._threat_of(i)
            if p.riichi_declared or threat >= 1.2:
                threatening_players.append((i, threat))
                safe_set = set()
                # Tiles in the threatening player's own river (normalize red dora)
                for tile, _ in p.river:
                    safe_set.add(tile_base(tile))
                # Tiles others discarded after this player's riichi (passed on)
                for base in p.post_riichi_safe:
                    safe_set.add(base)
                genbutsu_per_player[i] = safe_set

        if not threatening_players:
            return None

        # Find available discards that are genbutsu
        # Normalize tile names: "5mr" → "5m" for lookup
        genbutsu_options = []
        for idx in DISCARD_INDICES:
            if idx >= len(mask) or not mask[idx]:
                continue
            tile_name = ACTION_TILE_NAMES[idx]
            lookup_name = tile_base(tile_name)  # normalize red tiles
            
            safe_count = 0
            safe_score_sum = 0.0
            for p_idx, threat in threatening_players:
                if lookup_name in genbutsu_per_player[p_idx]:
                    safe_count += 1
                    safe_score_sum += threat
                    
            if safe_count > 0:
                isolation = self._tile_isolation_score(tile_name)
                genbutsu_options.append((idx, safe_count, safe_score_sum, isolation))

        if not genbutsu_options:
            return None

        # Pick best: most safe players first (共通安牌優先), then highest threat sum, then most isolated
        genbutsu_options.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
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
            # Honor tile: count copies in hand
            count = sum(1 for t in hand if tile_base(t) == tile_name)
            # Single copy = truly isolated, good to discard
            # Two copies = pair (useful for chiitoi, potential pon) — NOT isolated
            # Three copies = set (going to be used) — NOT isolated
            return 1.0 if count <= 1 else 0.0

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
        Post-riichi safe tiles (tiles others discarded after riichi without
        being ronned) are overridden to danger=0 since they are confirmed
        safe against the riichi player.
        """
        gs = self.gs
        candidates = []
        for idx in DISCARD_INDICES:
            if idx >= len(mask) or not mask[idx]:
                continue
            tile_name = ACTION_TILE_NAMES[idx]
            danger = aggregate_danger(tile_name, ctx)

            # Override danger for post-riichi safe tiles.
            # These tiles were discarded by other players after a riichi
            # declaration and not ronned — confirmed 100% safe against
            # that riichi player.  The safety module doesn't know about
            # post_riichi_safe, so we apply the override here.
            if danger > 0:
                lookup = tile_base(tile_name)
                for i, p in enumerate(gs.players):
                    if i == gs.player_id:
                        continue
                    if p.riichi_declared and lookup in p.post_riichi_safe:
                        # Safe against this riichi player — cap danger
                        danger = min(danger, DANGER_SAFE * 0.5)
                        break

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
            riichi_flags = []
            for i, p in enumerate(gs.players):
                # 警戒対象: リーチ者 または 脅威度1.2以上（3副露など）のテンパイ濃厚な相手
                is_threat = p.riichi_declared or gs._threat_of(i) >= 1.2
                riichi_flags.append(is_threat)
                if is_threat:
                    turn = p.riichi_turn if p.riichi_declared else gs.my_turn
                    if turn >= 0:
                        riichi_early[i] = turn

            return SafetyContext(
                riichi_flags=riichi_flags,
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

    # ========================================================
    # Thought log helper methods (思考ログヘルパー)
    # ========================================================

    def _format_round_info(self) -> str:
        """Format round and turn info."""
        gs = self.gs
        wind_jp = {"E": "東", "S": "南", "W": "西", "N": "北"}
        w = wind_jp.get(gs.round_wind, gs.round_wind)
        dealer_text = "親" if gs.is_dealer_me else "子"
        honba_text = f" {gs.honba}本場" if gs.honba > 0 else ""
        kyotaku_text = f" 供託{gs.kyotaku}本" if gs.kyotaku > 0 else ""
        turn_text = f"[{gs.my_turn}巡目]"
        return f"{w}{gs.round_number}局{honba_text}{kyotaku_text}{turn_text}({dealer_text}) — "

    def _format_hand_composition(self, hand_value: float) -> str:
        """Format hand composition details for thought log."""
        gs = self.gs
        parts = []

        # ドラ枚数
        dora_count = gs.count_dora_in_hand()
        if dora_count > 0:
            parts.append(f"ドラ{dora_count}枚")

        # 門前 / 副露
        if gs.my_melds:
            meld_types = []
            for m in gs.my_melds:
                meld_types.append(m.meld_type)
            parts.append(f"副露{len(gs.my_melds)}({', '.join(meld_types)})")
        else:
            parts.append("門前")

        # 役牌
        hand = gs.my_hand
        all_tiles = list(hand)
        for m in gs.my_melds:
            all_tiles.extend(m.tiles)

        yakuhai_list = []
        from .game_state import tile_base as _tb, YAKUHAI_HONORS
        checked = set()
        for t in all_tiles:
            base = _tb(t)
            if base in checked:
                continue
            checked.add(base)
            if gs.is_my_yakuhai(base):
                count = sum(1 for x in all_tiles if _tb(x) == base)
                if count >= 3:
                    yakuhai_list.append(f"{base}暗刻")
                elif count >= 2:
                    yakuhai_list.append(f"{base}対子")
        if yakuhai_list:
            parts.append(f"役牌[{'/'.join(yakuhai_list)}]")

        # 染め手傾向
        suit_counts = {"m": 0, "p": 0, "s": 0, "z": 0}
        from .game_state import parse_tile as _pt, HONORS as _HONORS
        for t in all_tiles:
            if t in _HONORS:
                suit_counts["z"] += 1
            else:
                s, _, _ = _pt(t)
                if s in suit_counts:
                    suit_counts[s] += 1
        total_num = suit_counts["m"] + suit_counts["p"] + suit_counts["s"]
        if total_num > 0:
            dominant = max("mps", key=lambda x: suit_counts[x])
            dominant_cnt = suit_counts[dominant]
            suit_jp = {"m": "萬子", "p": "筒子", "s": "索子"}
            if dominant_cnt + suit_counts["z"] >= len(all_tiles) - 1:
                parts.append(f"混一色寄り({suit_jp[dominant]})")
            elif dominant_cnt + suit_counts["z"] >= len(all_tiles) - 2:
                parts.append(f"{suit_jp[dominant]}寄り")

        # 対々 / 七対子傾向
        counts = {}
        for t in hand:
            base = _tb(t)
            counts[base] = counts.get(base, 0) + 1
        pairs = sum(1 for c in counts.values() if c >= 2)
        if pairs >= 5:
            parts.append("七対子ルート")
        elif pairs >= 3:
            pon_melds = sum(1 for m in gs.my_melds if m.meld_type in ("pon", "daiminkan", "kakan", "ankan"))
            if pon_melds + pairs >= 3:
                parts.append("対々和傾向")

        # 断么九チェック
        all_tanyao = True
        for t in all_tiles:
            if t in _HONORS:
                all_tanyao = False
                break
            s, r, _ = _pt(t)
            if r is not None and (r == 1 or r == 9):
                all_tanyao = False
                break
        if all_tanyao and len(all_tiles) >= 4:
            parts.append("断么九")

        if not parts:
            return ""
        return " / ".join(parts)

    def _format_threat_analysis(self) -> str:
        """Format opponent threat details for thought log."""
        gs = self.gs
        threat_parts = []

        for i, p in enumerate(gs.players):
            if i == gs.player_id:
                continue
            threat = gs._threat_of(i)
            if threat < 0.3:
                continue

            seat_jp = {0: "東家", 1: "南家", 2: "西家", 3: "北家"}
            # Adjust seat display relative to dealer
            winds = ["E", "S", "W", "N"]
            seat_wind = winds[(i - gs.dealer) % 4]
            seat_name = seat_jp.get((i - gs.dealer) % 4, f"Player{i}")

            reasons = []
            if p.riichi_declared:
                early = "早い" if p.riichi_turn <= 6 else "遅め"
                reasons.append(f"リーチ({early}/{p.riichi_turn}巡目)")
                if p.riichi_ippatsu:
                    reasons.append("一発圏内")
                if p.is_dealer:
                    reasons.append("親")
            if p.is_open():
                n = p.num_melds()
                reasons.append(f"副露{n}")
                # 染め手
                target_suit = p.detect_honitsu_chinitsu()
                if target_suit:
                    suit_jp = {"m": "萬子", "p": "筒子", "s": "索子"}
                    reasons.append(f"染め手({suit_jp.get(target_suit, target_suit)})")
                # 対々
                if p.detect_toitoi_signal():
                    reasons.append("対々和")
            if not p.riichi_declared and p.tedashi_after_tsumogiri_streak():
                reasons.append("ダマテン信号")

            if reasons:
                level_text = "⚠️" if threat >= 1.5 else "⚡" if threat >= 1.0 else "👁️"
                threat_parts.append(f"{level_text}{seat_name}[{threat:.1f}]: {', '.join(reasons)}")

        if not threat_parts:
            return ""
        return " | ".join(threat_parts)

    def _format_wait_info(self) -> str:
        """Format tenpai wait information for thought log."""
        gs = self.gs
        try:
            wait_details = gs.wait_tile_details()
            if not wait_details:
                return ""
            
            wait_parts = []
            total = 0
            for idx, cnt in wait_details:
                if idx < len(INDEX_TO_TILE):
                    tile_name = INDEX_TO_TILE[idx]
                    wait_parts.append(f"{tile_name}×{cnt}")
                    total += cnt
            
            if not wait_parts:
                return ""
            
            shape_text = "多面張" if len(wait_details) >= 3 else f"{len(wait_details)}面待ち" if len(wait_details) == 2 else "単騎/辺張/嵌張"
            return f"{', '.join(wait_parts)}（計{total}枚, {shape_text}）"
        except Exception:
            return ""

    def _translate_pf_reason(self, reason: str) -> str:
        """Translate push/fold reason to natural Japanese."""
        translations = {
            "already in riichi - locked": "リーチ宣言済み — 手は固定されています",
            "opponent ippatsu turn, far from tenpai - strict fold": "相手の一発巡目で手が遠い — 安全第一でオリます",
            "tenpai - push": "テンパイなので攻めます",
            "iishanten, low threat": "イーシャンテンで脅威が低い — 攻め続けます",
            "iishanten, early-mid game": "イーシャンテン、序中盤 — 攻めます",
            "iishanten, early-mid game, decent value": "イーシャンテン、序中盤で打点あり — 攻めます",
            "iishanten, late game": "イーシャンテン、終盤 — 回し打ちに切り替え",
            "ryanshanten, early round, low threat": "リャンシャンテン、序盤で脅威なし — まだ攻めます",
            "ryanshanten, early, moderate threat": "リャンシャンテン、序盤で中程度の脅威 — 回し打ち",
            "ryanshanten, cautious play": "リャンシャンテン — 慎重に進めます",
            "ryanshanten, strong threats": "リャンシャンテンで強い脅威あり — ベタオリ",
            "ryanshanten, late, no threat": "リャンシャンテン、終盤で脅威なし — 回し打ち",
        }
        # Check for partial matches
        for eng, jpn in translations.items():
            if reason == eng:
                return jpn

        # Pattern-based translations
        if "cheap bad-shape tenpai vs strong threat" in reason:
            return "安手愚形テンパイ vs 強い脅威 — オリます"
        if "cheap good-shape tenpai vs high threat late" in reason:
            return "安手好形テンパイだが終盤で高脅威 — 回し打ち"
        if "tenpai but cheap bad-shape vs extreme threat" in reason:
            return "安手愚形テンパイ vs 極高脅威 — 回し打ち"
        if "iishanten vs multiple threats" in reason:
            if "baiman" in reason:
                return "イーシャンテン vs 複数脅威、倍満以上あり — 回し打ち"
            return "イーシャンテン vs 複数脅威 — オリます"
        if "iishanten" in reason and "not enough value" in reason:
            return "イーシャンテンだが打点不足 vs 脅威 — オリます"
        if "iishanten" in reason and "bad shape" in reason:
            return "イーシャンテンだが愚形で脅威あり — オリます"
        if "iishanten" in reason and "good-shape" in reason:
            return "イーシャンテン、好形で価値あり — 攻めor回し打ち"
        if "iishanten" in reason and "1st/2nd place" in reason:
            return "イーシャンテンだが上位を守る — オリます"
        if "all-last" in reason:
            if "4th" in reason:
                return f"オーラス4着 — {reason.split(',')[-1].strip() if ',' in reason else '巻き返しを図ります'}"
            if "1st" in reason:
                return f"オーラス1着 — {'リードを守ります' if 'protect' in reason or 'fold' in reason or 'big lead' in reason else '慎重に進めます'}"
        if "south" in reason:
            return f"南場の順位状況に合わせた判断"
        if "shanten=" in reason:
            return f"手が遠い（{reason}）"
        if "desperate" in reason:
            return "窮地 — できる限り攻めます"

        return reason  # fallback: return original English
