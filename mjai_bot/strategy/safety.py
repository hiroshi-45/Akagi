# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Iterable, Union
import logging
import os

log = logging.getLogger("akagi.safety")

# === Honor safety tuning (小さめの効果・env可変) ===
HONOR_BASE_BONUS           = float(os.getenv("AKAGI_HONOR_BASE_BONUS", "0.08"))
HONOR_SEEN2_BONUS          = float(os.getenv("AKAGI_HONOR_SEEN2_BONUS", "0.05"))
HONOR_SEEN3_BONUS          = float(os.getenv("AKAGI_HONOR_SEEN3_BONUS", "0.15"))
HONOR_ENDGAME_BOOST        = float(os.getenv("AKAGI_HONOR_ENDGAME_BOOST", "1.3"))
HONOR_DORA_PENALTY         = float(os.getenv("AKAGI_HONOR_DORA_PENALTY", "0.18"))
HONOR_YAKUHAI_UNSEEN_PENAL = float(os.getenv("AKAGI_HONOR_YAKUHAI_UNSEEN_PENAL", "0.06"))


SUITS = ("m", "p", "s")
HONORS = {"E", "S", "W", "N", "P", "F", "C"}  # 東南西北 白發中
Tile = str
RiverItem = Tuple[Tile, bool]  # (tile, tsumogiri_flag)

# ------------------------------
# 基本ユーティリティ
# ------------------------------
def _is_genbutsu_for_any_riichi(tile: Tile, ctx: "SafetyContext") -> bool:
    # 簡易判定: 立直者の河に同一牌が1枚でもあれば現物扱い（厳密な順目は見ない）
    for i, f in enumerate(ctx.riichi_flags):
        if not f:
            continue
        river = ctx.rivers.get(i, [])
        for t in only_tiles(river):
            if t == tile:
                return True
    return False

def _visible_honor_count(tile: Tile, rivers: Dict[int, List], my_tiles: Optional[List[Tile]] = None) -> int:
    target = tile
    cnt = 0
    for _, river in rivers.items():
        for t in only_tiles(river):
            if t == target:
                cnt += 1
    if my_tiles:
        for t in my_tiles:
            if t == target:
                cnt += 1
    return cnt

def honor_safety_bonus(tile: Tile, ctx: "SafetyContext") -> float:
    # 現物ではない字牌の安全度だけ、少し底上げ
    if not is_honor(tile):
        return 0.0
    if _is_genbutsu_for_any_riichi(tile, ctx):
        return 0.0  # 現物は対象外

    bonus = HONOR_BASE_BONUS

    # 見え枚数で上積み
    seen = _visible_honor_count(tile, ctx.rivers, ctx.my_tiles)
    if seen >= 3:
        bonus += HONOR_SEEN3_BONUS
    elif seen >= 2:
        bonus += HONOR_SEEN2_BONUS

    # 終盤（残りツモ少）で強化
    try:
        if ctx.remaining_tiles <= 14:
            bonus *= HONOR_ENDGAME_BOOST
    except Exception:
        pass

    # ドラ字牌なら控えめに減算（=危険寄り）
    try:
        if ctx.dora_indicators and tile in ctx.dora_indicators:
            bonus -= HONOR_DORA_PENALTY
    except Exception:
        pass

    # 役牌が全く見えないならわずかに減算（過信防止）
    try:
        yakuhai = {"P", "F", "C"}
        river_seen = 0
        if tile in yakuhai:
            for _, river in ctx.rivers.items():
                for t in only_tiles(river):
                    if t == tile:
                        river_seen += 1
            if river_seen == 0:
                bonus -= HONOR_YAKUHAI_UNSEEN_PENAL
    except Exception:
        pass

    return max(0.0, bonus)


def parse_tile(t: Tile) -> Tuple[str, Optional[int], bool]:
    """
    return (suit_or_honor, rank(None for honor), is_red)
    e.g. '5mr' -> ('m', 5, True), '9p' -> ('p', 9, False), 'E' -> ('E', None, False)
    """
    if t in HONORS:
        return (t, None, False)
    is_red = t.endswith("r")
    core = t[:-1] if is_red else t
    suit = core[-1]
    rank = int(core[:-1])
    return (suit, rank, is_red)

def is_honor(t: Tile) -> bool:
    return t in HONORS

def only_tiles(river: Iterable[Union[Tile, RiverItem]]) -> List[Tile]:
    out: List[Tile] = []
    for x in river:
        if isinstance(x, tuple):
            out.append(x[0])
        else:
            out.append(x)
    return out

def hand_cuts(river: Iterable[RiverItem]) -> List[Tile]:
    """手出し(=not tsumogiri) のみを抽出。旧形式([tile,...])は“手出し扱い”で後方互換"""
    out: List[Tile] = []
    for x in river:
        if isinstance(x, tuple):
            t, tsumogiri = x
            if not tsumogiri:
                out.append(t)
        else:
            out.append(x)
    return out

# ------------------------------
# ドラ関連
# ------------------------------
def _dora_next_number(n: int) -> int:
    return 1 if n == 9 else n + 1

def indicator_to_dora(ind: Tile) -> Tile:
    """ドラ表示牌 -> ドラ"""
    if ind in HONORS:
        order = ["E", "S", "W", "N"] if ind in {"E", "S", "W", "N"} else ["P", "F", "C"]
        i = order.index(ind)
        return order[(i + 1) % len(order)]
    s, r, _ = parse_tile(ind)
    if r is None:
        return ind
    return f"{_dora_next_number(r)}{s}"

def expand_dora_numbers(dora_inds: Optional[List[Tile]]) -> Dict[str, Set[int]]:
    """
    スート別のドラ数字集合: {'m': {5}, 'p': {7}, 's': set()}
    字牌は除外（±1,±2の概念なし）
    """
    by: Dict[str, Set[int]] = {s: set() for s in SUITS}
    if not dora_inds:
        return by
    for ind in dora_inds:
        d = indicator_to_dora(ind)
        s, r, _ = parse_tile(d)
        if r is not None and s in SUITS:
            by[s].add(r)
    return by

# ------------------------------
# スジ/裏筋/壁/赤ドラ/跨ぎドラ
# ------------------------------
def suji_partner_ranks(r: int) -> Set[int]:
    # This function is conceptually flawed if used directly for 'if r in seen_partner'.
    # Keeping for backwards safety but suji_safe will be rewritten entirely.
    return set()

def suji_safe(tile: Tile, opp_all_tiles: List[Tile], seq_conf: float = 1.0) -> bool:
    """
    スジ判定。seq_conf は直前系列からの“信頼度”係数（>1で強め、<1で弱め）。
    """
    s, r, _ = parse_tile(tile)
    if r is None:
        return False
        
    cut_ranks = set()
    for d in opp_all_tiles:
        sd, rd, _ = parse_tile(d)
        if sd == s and rd is not None:
            cut_ranks.add(rd)
            
    # 正しいスジの定義
    # 1安全 <- 4切り
    # 2安全 <- 5切り
    # 3安全 <- 6切り
    # 4安全 <- 1と7両方切り (中スジ)
    # 5安全 <- 2と8両方切り
    # 6安全 <- 3と9両方切り
    # 7安全 <- 4切り
    # 8安全 <- 5切り
    # 9安全 <- 6切り
    if r == 1: return 4 in cut_ranks
    if r == 2: return 5 in cut_ranks
    if r == 3: return 6 in cut_ranks
    if r == 4: return (1 in cut_ranks and 7 in cut_ranks)
    if r == 5: return (2 in cut_ranks and 8 in cut_ranks)
    if r == 6: return (3 in cut_ranks and 9 in cut_ranks)
    if r == 7: return 4 in cut_ranks
    if r == 8: return 5 in cut_ranks
    if r == 9: return 6 in cut_ranks
    return False

def urasuji_danger(tile: Tile, opp_hand_cuts: List[Tile]) -> bool:
    """
    裏スジ（捨て牌の隣の牌が絡むスジ）の危険度。
    例：1切りからの2-5待ちなど。
    """
    s, r, _ = parse_tile(tile)
    if r is None:
        return False
        
    uramap = {
        1: [2, 5],
        2: [1, 4, 3, 6],
        3: [2, 5, 4, 7],
        4: [3, 6, 5, 8],
        5: [1, 4, 6, 9],
        6: [2, 5, 4, 7],
        7: [3, 6, 5, 8],
        8: [4, 7, 6, 9],
        9: [5, 8]
    }
    
    targets = set()
    for d in opp_hand_cuts:
        sd, rd, _ = parse_tile(d)
        if sd == s and rd is not None:
            for u in uramap.get(rd, []):
                targets.add(u)
    return r in targets

def count_visible_numbers(rivers: Dict[int, List[Union[Tile, RiverItem]]],
                          my_tiles: Optional[List[Tile]] = None) -> Dict[str, Dict[int, int]]:
    cnt = {s: {r: 0 for r in range(1, 10)} for s in SUITS}
    tiles = []
    for _, river in rivers.items():
        tiles.extend([t for t in only_tiles(river) if not is_honor(t)])
    if my_tiles:
        tiles.extend([t for t in my_tiles if not is_honor(t)])
    for t in tiles:
        s, r, _ = parse_tile(t)
        if r is not None:
            cnt[s][r] += 1
    return cnt

def kabe_bonus(tile: Tile, visible: Dict[str, Dict[int, int]], endgame_boost: float = 1.0) -> float:
    """
    壁読み（ノーチャンス・ワンチャンス）の安全加点。
    終盤（endgame_boost>1）では加点を強める。
    """
    s, r, _ = parse_tile(tile)
    if r is None or s not in SUITS:
        return 0.0
    v = visible.get(s, {})
    bonus = 0.0
    
    is_no_chance = False
    is_one_chance = False
    
    # 壁の法則に基づく両面待ちの否定
    if r == 1:
        if v.get(2, 0) >= 4: is_no_chance = True
        elif v.get(2, 0) == 3: is_one_chance = True
        if v.get(3, 0) >= 4: is_no_chance = True
        elif v.get(3, 0) == 3: is_one_chance = True
    elif r == 2:
        if v.get(3, 0) >= 4: is_no_chance = True
        elif v.get(3, 0) == 3: is_one_chance = True
        if v.get(4, 0) >= 4: is_no_chance = True
        elif v.get(4, 0) == 3: is_one_chance = True
    elif r == 3:
        if v.get(4, 0) >= 4: is_no_chance = True
        elif v.get(4, 0) == 3: is_one_chance = True
        if v.get(5, 0) >= 4: is_no_chance = True
        elif v.get(5, 0) == 3: is_one_chance = True
    elif r == 4:
        # 4 can be waited on via 23 (ryanmen), 45 (ryanmen for 3 or 6), or 56 (ryanmen)
        # If 3 is walled, 12-waiting-on-3 pattern is blocked (one side blocked)
        # If 6 is walled, 67-waiting-on-6 pattern is blocked (one side blocked)
        if v.get(3, 0) >= 4:
            is_one_chance = True
        if v.get(6, 0) >= 4:
            is_one_chance = True
    elif r == 5:
        # 5 can be waited on via 34, 56, 46 (kanchan)
        # If 4 is walled, 34-ryanmen is blocked
        # If 6 is walled, 67→56-ryanmen is blocked
        if v.get(4, 0) >= 4:
            is_one_chance = True
        if v.get(6, 0) >= 4:
            is_one_chance = True
    elif r == 6:
        # Symmetric to 4
        if v.get(4, 0) >= 4:
            is_one_chance = True
        if v.get(7, 0) >= 4:
            is_one_chance = True
    elif r == 7:
        if v.get(5, 0) >= 4: is_no_chance = True
        elif v.get(5, 0) == 3: is_one_chance = True
        if v.get(6, 0) >= 4: is_no_chance = True
        elif v.get(6, 0) == 3: is_one_chance = True
    elif r == 8:
        if v.get(6, 0) >= 4: is_no_chance = True
        elif v.get(6, 0) == 3: is_one_chance = True
        if v.get(7, 0) >= 4: is_no_chance = True
        elif v.get(7, 0) == 3: is_one_chance = True
    elif r == 9:
        if v.get(7, 0) >= 4: is_no_chance = True
        elif v.get(7, 0) == 3: is_one_chance = True
        if v.get(8, 0) >= 4: is_no_chance = True
        elif v.get(8, 0) == 3: is_one_chance = True

    if is_no_chance:
        bonus += 0.35
    elif is_one_chance:
        bonus += 0.15
        
    return bonus * endgame_boost

def no_chance_bonus(tile: Tile, visible: Dict[str, Dict[int, int]], remaining_tiles: int) -> float:
    """
    kabe_bonusに統合済み。後方互換で0.0を返す。
    """
    return 0.0

def red_dora_pressure(tile: Tile) -> float:
    """赤5そのもの・同スート5周辺のわずかな危険増。"""
    s, r, is_red = parse_tile(tile)
    if r is None or s not in SUITS:
        return 0.0
    if is_red and r == 5:
        return 0.20
    if r in (4, 5, 6):
        return 0.05
    return 0.0

def dora_pressure(tile: Tile, dora_by_suit: Dict[str, Set[int]]) -> float:
    s, r, _ = parse_tile(tile)
    if r is None or s not in SUITS:
        return 0.0
    ds = dora_by_suit.get(s, set())
    if r in ds or (r - 1) in ds or (r + 1) in ds:
        return 0.10
    if (r - 2) in ds or (r + 2) in ds:  # 跨ぎドラ
        return 0.15
    return 0.0

def sequence_confidence(opp_hand_cuts_list: List[Tile]) -> float:
    """
    直近の手出し系列から“形読みの信頼度”を出す。例えば 9s→8s の連続等でやや強める。
    簡易: 直近3枚のうち同一スートの連続/近接があれば 1.15、2回以上なら 1.25。
    """
    last = opp_hand_cuts_list[-3:]
    if not last:
        return 1.0
    by_suit: Dict[str, List[int]] = {"m": [], "p": [], "s": []}
    for t in last:
        s, r, _ = parse_tile(t)
        if r is not None and s in SUITS:
            by_suit[s].append(r)
    bump = 0
    for s in SUITS:
        ranks = by_suit[s]
        ranks.sort(reverse=True)
        for i in range(len(ranks) - 1):
            if abs(ranks[i] - ranks[i + 1]) <= 1:
                bump += 1
    if bump >= 2:
        return 1.25
    if bump == 1:
        return 1.15
    return 1.0

def matagi_danger(tile: Tile, opp_hand_cuts: List[Tile]) -> float:
    """
    またぎスジの危険度。直近の手出し牌（ターツ落としやテンパイ宣言牌周辺）は危険。
    """
    s, r, _ = parse_tile(tile)
    if r is None or s not in SUITS:
        return 0.0
    
    recent_cuts = opp_hand_cuts[-3:]
    danger = 0.0
    for d in recent_cuts:
        sd, rd, _ = parse_tile(d)
        if rd is None or sd != s:
            continue
        if abs(r - rd) == 1 or abs(r - rd) == 2:
            danger += 0.10
    return danger

# ------------------------------
# コンテキスト & 総合危険度
# ------------------------------
@dataclass
class SafetyContext:
    riichi_flags: List[bool]
    rivers: Dict[int, List[Union[Tile, RiverItem]]]
    my_index: int
    remaining_tiles: int
    dealer: int
    dora_indicators: Optional[List[Tile]] = None
    my_tiles: Optional[List[Tile]] = None
    riichi_early_turns: Optional[Dict[int, int]] = None  # actor->宣言順目（小さいほど早い）
    # 早い親リーチ補正
    early_dealer_riichi_boost_at: int = int(os.getenv("AKAGI_EARLY_DEALER_RIICHI_TURN", "8"))
    early_dealer_riichi_add: float = float(os.getenv("AKAGI_EARLY_DEALER_RIICHI_ADD", "0.10"))

def danger_against_player(tile: Tile,
                          opp_river: List[Union[Tile, RiverItem]],
                          visible: Dict[str, Dict[int, int]],
                          dora_by_suit: Dict[str, Set[int]],
                          opp_turn_riichi: Optional[int],
                          is_dealer: bool,
                          ctx: SafetyContext) -> float:
    """
    1家に対する危険度（0.0=超安全 ... 1.5=かなり危険）
    """
    # 現物
    if tile in only_tiles(opp_river):
        return 0.0

    s, r, _ = parse_tile(tile)
    
    if is_honor(tile):
        # 字牌は現物でなければ中程度のまま（副露読みなどは別途）
        base = 1.0
    else:
        # Top player logic: Base danger depends heavily on tile rank (端牌 vs 中張牌)
        if r in (1, 9):
            base = 0.5   # 端牌 without suji is moderately safe
        elif r in (2, 8):
            base = 0.7
        elif r in (3, 7):
            base = 0.9
        else:
            base = 1.15  # 無筋 4,5,6 is extremely dangerous

        # スジ/裏筋
        opp_hand = hand_cuts(opp_river)
        opp_all = only_tiles(opp_river)
        seq_conf = sequence_confidence(opp_hand)
        
        # スジはツモ切りも含めた全捨て牌を参照する
        if suji_safe(tile, opp_all, seq_conf):
            # Suji makes the tile safer.
            # Example: 1/9 -> 0.5 - 0.35 = 0.15 (very safe)
            # 4/5/6 -> 1.15 - 0.35 = 0.8 (still somewhat dangerous as Nakasuji)
            base -= 0.35 * min(seq_conf, 1.3)  # 信頼度を軽く効かせる
        # 裏筋・跨ぎ筋は手出しのみを参照する
        elif urasuji_danger(tile, opp_hand):
            base += 0.15

        base += matagi_danger(tile, opp_hand)

        # 壁/ノーチャンス（終盤強化）
        end_boost = 1.3 if ctx.remaining_tiles <= 14 else 1.0
        base -= kabe_bonus(tile, visible, end_boost)
        base -= no_chance_bonus(tile, visible, ctx.remaining_tiles)

        # 赤ドラ/ドラ周辺
        base += red_dora_pressure(tile)
        base += dora_pressure(tile, dora_by_suit)

    # 早い親リーチ補正
    if is_dealer and opp_turn_riichi is not None:
        if opp_turn_riichi <= ctx.early_dealer_riichi_boost_at:
            base += ctx.early_dealer_riichi_add

    return max(0.0, min(1.6, base))

def aggregate_danger(tile: Tile, ctx: SafetyContext) -> float:
    """
    立直者（複数時は最も危険な相手）への危険。平場は穏やかにドラ等のみ反映。
    カン直後の新ドラは ctx.dora_indicators の変化で即時反映される。
    """
    visible = count_visible_numbers(ctx.rivers, ctx.my_tiles)
    dora_by_suit = expand_dora_numbers(ctx.dora_indicators)
    per: List[float] = []
    for i, riichi in enumerate(ctx.riichi_flags):
        if not riichi:
            continue
        opp_turn = None if ctx.riichi_early_turns is None else ctx.riichi_early_turns.get(i)
        per.append(
            danger_against_player(
                tile,
                ctx.rivers.get(i, []),
                visible,
                dora_by_suit,
                opp_turn,
                is_dealer=(i == ctx.dealer),
                ctx=ctx,
            )
        )
    if not per:
        # 平場: 壁/ドラ/赤のみ軽く
        base = max(0.0, 0.7 - kabe_bonus(tile, visible))
        base += dora_pressure(tile, dora_by_suit) * 0.5
        base += red_dora_pressure(tile) * 0.5
        return max(0.0, min(1.2, base))

    d = max(per)
    # 複数立直で +補正
    if sum(1 for r in ctx.riichi_flags if r) >= 2:
        d += 0.15
    # 終盤 +補正
    if ctx.remaining_tiles <= 18:
        d += 0.15
    
    # 字牌現物ボーナス
    d = max(0.0, d - honor_safety_bonus(tile, ctx))
    return max(0.0, min(1.8, d))

def bucketize(d: float) -> str:
    if d <= 0.1: return "ZERO(現物)"
    if d <= 0.35: return "LOW(スジ/壁優勢)"
    if d <= 0.7: return "MID"
    if d <= 1.1: return "HIGH"
    return "VHIGH"
