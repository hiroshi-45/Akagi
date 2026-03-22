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
    # 捨て牌ランク r に対して「スジ安全」になるランクの集合を返す。
    # 両面待ちのペア: (1,4), (2,5), (3,6), (4,7), (5,8), (6,9)
    # 4切り → 1 と 7 が安全、6切り → 3 と 9 が安全
    mp = {1: (4,), 2: (5,), 3: (6,), 4: (1, 7), 5: (2, 8), 6: (3, 9), 7: (4,), 8: (5,), 9: (6,)}
    v = mp.get(r)
    if v is None:
        return set()
    return set(v)

def suji_safe(tile: Tile, opp_hand_cuts: List[Tile], seq_conf: float = 1.0) -> bool:
    """
    スジ判定。seq_conf は直前系列からの“信頼度”係数（>1で強め、<1で弱め）。
    ここでは bool を返すが、後段で信頼係数を用いて減点幅を可変化する。
    """
    s, r, _ = parse_tile(tile)
    if r is None:
        return False
    seen_partner = set()
    for d in opp_hand_cuts:
        sd, rd, _ = parse_tile(d)
        if rd is None:
            continue
        if sd == s:
            seen_partner |= suji_partner_ranks(rd)
    return r in seen_partner

def urasuji_danger(tile: Tile, opp_hand_cuts: List[Tile]) -> bool:
    s, r, _ = parse_tile(tile)
    if r is None:
        return False
    uramap = {6: 3, 7: 4, 8: 5, 3: 6, 4: 7, 5: 8, 2: 5, 1: 4, 9: 6}
    targets = set()
    for d in opp_hand_cuts:
        sd, rd, _ = parse_tile(d)
        if rd is None:
            continue
        if sd == s:
            u = uramap.get(rd)
            if u:
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
    壁読みの安全加点。終盤（endgame_boost>1）では加点を強める。
    """
    s, r, _ = parse_tile(tile)
    if r is None or s not in SUITS:
        return 0.0
    v = visible.get(s, {})
    bonus = 0.0
    if v.get(1, 0) >= 4 and r == 2:
        bonus += 0.25
    if v.get(9, 0) >= 4 and r == 8:
        bonus += 0.25
    for n in range(2, 9):
        if v.get(n, 0) >= 4 and abs(r - n) == 1:
            bonus += 0.15
    return bonus * endgame_boost

def no_chance_bonus(tile: Tile, visible: Dict[str, Dict[int, int]], remaining_tiles: int) -> float:
    """
    ノーチャンス/両無筋寄りの追加安全。簡易:
      - 同スートで『端の壁＋周辺3枚以上見え』等は +α
      - 終盤(<=14)で +強化
    """
    s, r, _ = parse_tile(tile)
    if r is None or s not in SUITS:
        return 0.0
    v = visible.get(s, {})
    add = 0.0
    # 端ノーチャンス強化
    if r == 2 and v.get(1, 0) >= 4 and (v.get(3, 0) + v.get(4, 0)) >= 3:
        add += 0.08
    if r == 8 and v.get(9, 0) >= 4 and (v.get(6, 0) + v.get(7, 0)) >= 3:
        add += 0.08
    if remaining_tiles <= 14:
        add *= 1.5
    return add

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

    base = 1.0
    s, r, _ = parse_tile(tile)
    if is_honor(tile):
        # 字牌は現物でなければ中程度のまま（副露読みなどは別途）
        pass
    else:
        # スジ/裏筋（手出しのみ参照）
        opp_hand = hand_cuts(opp_river)
        seq_conf = sequence_confidence(opp_hand)
        if suji_safe(tile, opp_hand, seq_conf):
            base -= 0.35 * min(seq_conf, 1.3)  # 信頼度を軽く効かせる
        elif urasuji_danger(tile, opp_hand):
            base += 0.15

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
