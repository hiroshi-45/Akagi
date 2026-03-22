# -*- coding: utf-8 -*-
"""Comprehensive game state tracker for strategic decision-making.

Tracks all observable information throughout a game and provides
derived metrics used by the strategy engine.

Enhanced with:
- Tedashi/tsumogiri pattern tracking for threat reading
- River suit bias analysis for染め手 detection
- Acceptance count estimation
- Turn-aware threat assessment
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Tile constants
SUITS = ("m", "p", "s")
HONORS = frozenset({"E", "S", "W", "N", "P", "F", "C"})
YAKUHAI_HONORS = frozenset({"P", "F", "C"})  # 白發中
WIND_HONORS = {"E": 0, "S": 1, "W": 2, "N": 3}

Tile = str


def parse_tile(t: Tile) -> Tuple[str, Optional[int], bool]:
    if t in HONORS:
        return (t, None, False)
    is_red = t.endswith("r")
    core = t[:-1] if is_red else t
    suit = core[-1]
    rank = int(core[:-1])
    return (suit, rank, is_red)


def is_honor(t: Tile) -> bool:
    return t in HONORS


def tile_to_index(t: Tile) -> int:
    """Convert tile string to 0-33 index (ignoring red)."""
    if t in HONORS:
        honor_order = ["E", "S", "W", "N", "P", "F", "C"]
        return 27 + honor_order.index(t)
    s, r, _ = parse_tile(t)
    suit_offset = {"m": 0, "p": 9, "s": 18}
    return suit_offset[s] + (r - 1)


def tile_base(t: Tile) -> Tile:
    """Strip red indicator: '5mr' -> '5m', '5m' -> '5m', 'E' -> 'E'."""
    if t in HONORS:
        return t
    return t.rstrip("r")


def indicator_to_dora(ind: Tile) -> Tile:
    if ind in HONORS:
        order = ["E", "S", "W", "N"] if ind in {"E", "S", "W", "N"} else ["P", "F", "C"]
        i = order.index(ind)
        return order[(i + 1) % len(order)]
    s, r, _ = parse_tile(ind)
    if r is None:
        return ind
    next_r = 1 if r == 9 else r + 1
    return f"{next_r}{s}"


@dataclass
class MeldInfo:
    """A player's meld (chi/pon/kan)."""
    meld_type: str  # "chi", "pon", "daiminkan", "ankan", "kakan"
    tiles: List[Tile] = field(default_factory=list)
    from_player: int = -1


@dataclass
class PlayerInfo:
    """Per-player tracked information."""
    score: int = 25000
    river: List[Tuple[Tile, bool]] = field(default_factory=list)  # (tile, tsumogiri)
    melds: List[MeldInfo] = field(default_factory=list)
    riichi_declared: bool = False
    riichi_turn: int = -1  # turn when riichi was declared
    riichi_ippatsu: bool = False
    is_dealer: bool = False
    # Derived
    safe_tiles_for_me: Set[Tile] = field(default_factory=set)  # genbutsu
    # Tedashi tracking: count of consecutive tsumogiri before a tedashi
    _consecutive_tsumogiri: int = 0
    _last_tedashi_turn: int = 0
    _tedashi_count: int = 0  # total tedashi count this round

    def river_tiles(self) -> List[Tile]:
        return [t for t, _ in self.river]

    def hand_cut_tiles(self) -> List[Tile]:
        """Tiles discarded by hand (not tsumogiri)."""
        return [t for t, tg in self.river if not tg]

    def tsumogiri_tiles(self) -> List[Tile]:
        """Tiles discarded by tsumogiri."""
        return [t for t, tg in self.river if tg]

    def is_open(self) -> bool:
        return any(m.meld_type in ("chi", "pon", "daiminkan", "kakan") for m in self.melds)

    def num_melds(self) -> int:
        return len(self.melds)

    def river_suit_counts(self) -> Dict[str, int]:
        """Count suits in river for染め手 detection."""
        counts: Dict[str, int] = {"m": 0, "p": 0, "s": 0, "z": 0}
        for t, _ in self.river:
            if t in HONORS:
                counts["z"] += 1
            else:
                s, _, _ = parse_tile(t)
                if s in counts:
                    counts[s] += 1
        return counts

    def meld_suit_set(self) -> Set[str]:
        """Suits used in melds."""
        suits = set()
        for m in self.melds:
            for t in m.tiles:
                if t in HONORS:
                    suits.add("z")
                else:
                    s, _, _ = parse_tile(t)
                    suits.add(s)
        return suits

    def detect_honitsu_chinitsu(self) -> Optional[str]:
        """Detect if this player appears to be going for honitsu/chinitsu.

        Returns the target suit or None.
        """
        if not self.is_open():
            return None
        if self.num_melds() < 2:
            return None

        river_counts = self.river_suit_counts()
        meld_suits = self.meld_suit_set() - {"z"}

        if len(meld_suits) != 1:
            return None
        target_suit = list(meld_suits)[0]

        # Check if they're discarding the other two suits heavily
        other_suit_count = sum(v for k, v in river_counts.items() if k != target_suit and k != "z")
        total_river = len(self.river)
        if total_river >= 4 and other_suit_count / max(total_river, 1) >= 0.5:
            return target_suit

        return None

    def detect_toitoi_signal(self) -> bool:
        """Detect if melds suggest toitoi/honitsu pattern."""
        pon_count = sum(1 for m in self.melds if m.meld_type in ("pon", "daiminkan", "kakan", "ankan"))
        return pon_count >= 2

    def tedashi_after_tsumogiri_streak(self) -> bool:
        """Whether last event was tedashi after a streak of tsumogiri.

        This is a strong tenpai signal used by top players.
        """
        return self._consecutive_tsumogiri >= 3 and self._tedashi_count > 0

    def apparent_threat_level(self, current_turn: int = 0) -> float:
        """Heuristic threat level from discards, melds, and behavioral patterns.

        Enhanced with:
        - Tedashi pattern reading (tsumogiri streak → tedashi = tenpai signal)
        - Suit bias in river (honitsu/chinitsu detection)
        - Toitoi signal
        - Turn-aware riichi assessment
        """
        threat = 0.0

        # === Riichi ===
        if self.riichi_declared:
            threat += 1.5
            if self.riichi_turn <= 6:
                threat += 0.5  # early riichi
            if self.is_dealer:
                threat += 0.3  # dealer riichi is more expensive

        # === Open hand threats ===
        if self.is_open():
            n = self.num_melds()
            # Yakuhai pon
            yakuhai_count = 0
            for m in self.melds:
                if m.meld_type == "pon" and len(m.tiles) > 0 and tile_base(m.tiles[0]) in YAKUHAI_HONORS:
                    yakuhai_count += 1
                    threat += 0.6
            # Double yakuhai
            if yakuhai_count >= 2:
                threat += 0.4

            # Honitsu/Chinitsu signal
            target_suit = self.detect_honitsu_chinitsu()
            if target_suit is not None:
                threat += 0.8 if n >= 3 else 0.5

            # Toitoi signal
            if self.detect_toitoi_signal():
                threat += 0.5

            # Many melds = closer to tenpai
            if n >= 3:
                threat += 0.7
            elif n >= 2:
                threat += 0.3

        # === Tedashi pattern: tsumogiri streak then tedashi = tenpai signal ===
        if not self.riichi_declared and self.tedashi_after_tsumogiri_streak():
            threat += 0.6

        # === Mid-to-late game with few discards from hand = holding hand ===
        if current_turn >= 10 and not self.riichi_declared and not self.is_open():
            river_len = len(self.river)
            tedashi_ratio = self._tedashi_count / max(river_len, 1)
            if river_len >= 6 and tedashi_ratio <= 0.3:
                # Mostly tsumogiri = likely already tenpai or iishanten
                threat += 0.3

        return threat


@dataclass
class GameState:
    """Full observable game state for strategic decisions."""
    # Game-level
    player_id: int = 0
    round_wind: str = "E"  # bakaze
    round_number: int = 1  # kyoku (1-based)
    honba: int = 0
    kyotaku: int = 0  # riichi sticks on table

    # Round-level
    dealer: int = 0  # oya seat
    turn: int = 0  # current turn count
    dora_indicators: List[Tile] = field(default_factory=list)
    remaining_tiles: int = 70  # approximate tiles left in wall

    # Per-player
    players: List[PlayerInfo] = field(default_factory=lambda: [PlayerInfo() for _ in range(4)])

    # My hand
    my_hand: List[Tile] = field(default_factory=list)
    my_tsumo: Optional[Tile] = None

    # Tile visibility tracker (34-element counts: how many of each tile seen)
    visible_counts: List[int] = field(default_factory=lambda: [0] * 34)

    # Tracking
    _initialized: bool = False

    def reset_round(self) -> None:
        self.turn = 0
        self.remaining_tiles = 70
        self.my_hand = []
        self.my_tsumo = None
        self.dora_indicators = []
        self.visible_counts = [0] * 34
        for p in self.players:
            p.river = []
            p.melds = []
            p.riichi_declared = False
            p.riichi_turn = -1
            p.riichi_ippatsu = False
            p.safe_tiles_for_me = set()
            p._consecutive_tsumogiri = 0
            p._last_tedashi_turn = 0
            p._tedashi_count = 0

    def reset_game(self) -> None:
        self._initialized = False
        self.players = [PlayerInfo() for _ in range(4)]
        self.reset_round()

    # === Derived metrics ===

    @property
    def my_info(self) -> PlayerInfo:
        return self.players[self.player_id]

    @property
    def my_score(self) -> int:
        return self.players[self.player_id].score

    @property
    def scores(self) -> List[int]:
        return [p.score for p in self.players]

    @property
    def my_placement(self) -> int:
        """1 = first, 4 = last."""
        my_s = self.my_score
        rank = 1
        for i, p in enumerate(self.players):
            if i != self.player_id:
                if p.score > my_s or (p.score == my_s and i < self.player_id):
                    rank += 1
        return rank

    @property
    def diff_to_first(self) -> int:
        return max(p.score for p in self.players) - self.my_score

    @property
    def diff_to_above(self) -> int:
        """Point difference to the player directly above me."""
        my_s = self.my_score
        above_scores = sorted([p.score for i, p in enumerate(self.players)
                                if p.score > my_s or (p.score == my_s and i < self.player_id)],
                               reverse=True)
        if not above_scores:
            return 0
        return above_scores[-1] - my_s

    @property
    def diff_to_below(self) -> int:
        """Point difference to the player directly below me."""
        my_s = self.my_score
        below_scores = sorted([p.score for i, p in enumerate(self.players)
                                if p.score < my_s or (p.score == my_s and i > self.player_id)])
        if not below_scores:
            return 0
        return my_s - below_scores[-1]

    @property
    def is_all_last(self) -> bool:
        """Are we in the final round (South 4 / オーラス)?"""
        return self.round_wind == "S" and self.round_number == 4

    @property
    def is_south(self) -> bool:
        return self.round_wind in ("S", "W", "N")

    @property
    def is_late_game(self) -> bool:
        return self.is_south and self.round_number >= 3

    @property
    def is_dealer_me(self) -> bool:
        return self.dealer == self.player_id

    @property
    def num_riichi_opponents(self) -> int:
        return sum(1 for i, p in enumerate(self.players)
                   if i != self.player_id and p.riichi_declared)

    @property
    def riichi_flags(self) -> List[bool]:
        return [p.riichi_declared for p in self.players]

    @property
    def rivers_dict(self) -> Dict[int, List]:
        return {i: p.river for i, p in enumerate(self.players)}

    @property
    def doras(self) -> List[Tile]:
        return [indicator_to_dora(ind) for ind in self.dora_indicators]

    @property
    def is_endgame(self) -> bool:
        """Late in a round (few tiles remaining)."""
        return self.remaining_tiles <= 18

    @property
    def my_turn(self) -> int:
        """Approximate per-player turn number (0-based)."""
        return self.turn // 4

    def count_dora_in_hand(self) -> int:
        """Count dora tiles in my hand (fixed: no double-counting red dora)."""
        dora_list = self.doras
        count = 0
        for t in self.my_hand:
            base = tile_base(t)
            # Check normal dora match
            for d in dora_list:
                if base == d:
                    count += 1
            # Red dora counts as 1 dora (independent of dora indicator)
            if t.endswith("r"):
                count += 1
        return count

    def unseen_count(self, tile: Tile) -> int:
        """How many of this tile remain unseen (in wall or other hands)."""
        idx = tile_to_index(tile)
        if 0 <= idx < 34:
            return max(0, 4 - self.visible_counts[idx])
        return 0

    def estimate_acceptance_count(self) -> int:
        """Rough estimate of useful tile count for improving hand.

        Counts tiles that could form mentsu with existing pairs/partial-mentsu.
        This is a simplified version — real acceptance count needs shanten analysis.
        """
        hand_counts = _hand_to_34(self.my_hand)
        acceptance = 0

        # For each tile type, check if drawing it could help
        for idx in range(34):
            if self.visible_counts[idx] >= 4:
                continue  # no copies left
            remaining = 4 - self.visible_counts[idx]
            if remaining <= 0:
                continue

            # Check if this tile connects to something in hand
            if idx < 27:
                # Number tile
                suit_offset = (idx // 9) * 9
                rank = idx - suit_offset  # 0-8
                # Would form pair
                if hand_counts[idx] == 1:
                    acceptance += remaining
                    continue
                # Would form mentsu with adjacent
                if rank >= 1 and hand_counts[idx - 1] >= 1:
                    acceptance += remaining
                    continue
                if rank <= 7 and hand_counts[idx + 1] >= 1:
                    acceptance += remaining
                    continue
                if rank >= 2 and hand_counts[idx - 2] >= 1:
                    acceptance += remaining
                    continue
                if rank <= 6 and hand_counts[idx + 2] >= 1:
                    acceptance += remaining
                    continue
            else:
                # Honor tile: only pairs/triplets matter
                if hand_counts[idx] >= 1:
                    acceptance += remaining

        return acceptance

    def my_wind(self) -> str:
        """My seat wind."""
        # Seat wind is relative to dealer
        winds = ["E", "S", "W", "N"]
        return winds[(self.player_id - self.dealer) % 4]

    def is_my_yakuhai(self, tile: Tile) -> bool:
        """Check if a tile is yakuhai for me."""
        if tile in YAKUHAI_HONORS:
            return True
        if tile == self.round_wind:
            return True
        if tile == self.my_wind():
            return True
        return False

    def threat_level_total(self) -> float:
        """Sum of all opponents' threat levels."""
        return sum(self.players[i].apparent_threat_level(self.turn)
                   for i in range(4) if i != self.player_id)

    def max_opponent_threat(self) -> float:
        return max((self.players[i].apparent_threat_level(self.turn)
                    for i in range(4) if i != self.player_id), default=0.0)

    def highest_threat_player(self) -> Optional[int]:
        """Return seat index of the most threatening opponent."""
        best_i = None
        best_t = 0.0
        for i in range(4):
            if i == self.player_id:
                continue
            t = self.players[i].apparent_threat_level(self.turn)
            if t > best_t:
                best_t = t
                best_i = i
        return best_i

    # === Point calculation helpers ===

    def points_needed_for_placement(self, target_placement: int) -> int:
        """Calculate minimum points needed to reach target placement.

        Returns the point deficit to overcome (0 if already at or above target).
        """
        sorted_scores = sorted(
            [(p.score, i) for i, p in enumerate(self.players)],
            key=lambda x: (-x[0], x[1])
        )
        if target_placement < 1 or target_placement > 4:
            return 0
        target_score, target_seat = sorted_scores[target_placement - 1]
        my_s = self.my_score
        if my_s >= target_score and self.my_placement <= target_placement:
            return 0
        # Need to exceed (or equal with seat priority) the target
        diff = target_score - my_s
        if self.player_id > target_seat:
            diff += 100  # need to strictly exceed for seat tiebreak
        return max(0, diff)

    def min_han_for_points(self, target_points: int, is_tsumo: bool = False) -> int:
        """Rough estimate: minimum han needed to reach target_points.

        Uses 30fu as baseline.
        """
        if target_points <= 0:
            return 1
        # Point table (30fu, non-dealer ron)
        # 1 han: 1000, 2 han: 2000, 3 han: 3900, 4 han: 7700,
        # 5+ (mangan): 8000, 6-7 (haneman): 12000, 8-10 (baiman): 16000
        ron_points = [0, 1000, 2000, 3900, 7700, 8000, 12000, 12000, 16000, 16000, 16000]
        tsumo_total = [0, 1300, 2600, 5200, 8000, 8000, 12000, 12000, 16000, 16000, 16000]
        if self.is_dealer_me:
            ron_points = [0, 1500, 2900, 5800, 11600, 12000, 18000, 18000, 24000, 24000, 24000]
            tsumo_total = [0, 2000, 3900, 7700, 12000, 12000, 18000, 18000, 24000, 24000, 24000]

        pts = tsumo_total if is_tsumo else ron_points
        for han in range(1, len(pts)):
            if pts[han] >= target_points:
                return han
        return 11  # beyond baiman

    # === Event processing ===

    def process_event(self, event: dict) -> None:
        """Update game state from an MJAI event."""
        etype = event.get("type", "")

        if etype == "start_game":
            self.player_id = event["id"]
            self.reset_game()
            self._initialized = True
            return

        if not self._initialized:
            return

        if etype == "start_kyoku":
            self._handle_start_kyoku(event)
        elif etype == "tsumo":
            self._handle_tsumo(event)
        elif etype == "dahai":
            self._handle_dahai(event)
        elif etype == "chi":
            self._handle_meld(event, "chi")
        elif etype == "pon":
            self._handle_meld(event, "pon")
        elif etype in ("daiminkan", "ankan", "kakan"):
            self._handle_kan(event, etype)
        elif etype == "reach":
            self._handle_reach(event)
        elif etype == "reach_accepted":
            self._handle_reach_accepted(event)
        elif etype == "dora":
            self._handle_dora(event)
        elif etype in ("hora", "end_kyoku", "ryukyoku"):
            pass  # round ended
        elif etype == "end_game":
            self.reset_game()

    def _handle_start_kyoku(self, event: dict) -> None:
        self.reset_round()
        self.round_wind = event.get("bakaze", "E")
        self.round_number = event.get("kyoku", 1)
        self.honba = event.get("honba", 0)
        self.kyotaku = event.get("kyotaku", 0)
        self.dealer = event.get("oya", 0)

        scores = event.get("scores", [25000] * 4)
        for i, s in enumerate(scores):
            self.players[i].score = s
            self.players[i].is_dealer = (i == self.dealer)

        dora_marker = event.get("dora_marker")
        if dora_marker:
            self.dora_indicators = [dora_marker]
            self._mark_visible(dora_marker)

        tehais = event.get("tehais", [])
        if self.player_id < len(tehais):
            my_tiles = tehais[self.player_id]
            self.my_hand = [t for t in my_tiles if t != "?"]
            for t in self.my_hand:
                self._mark_visible(t)

    def _handle_tsumo(self, event: dict) -> None:
        actor = event.get("actor", -1)
        pai = event.get("pai", "?")
        self.remaining_tiles = max(0, self.remaining_tiles - 1)
        self.turn += 1

        if actor == self.player_id and pai != "?":
            self.my_tsumo = pai
            self.my_hand.append(pai)
            self._mark_visible(pai)

    def _handle_dahai(self, event: dict) -> None:
        actor = event.get("actor", -1)
        pai = event.get("pai", "?")
        tsumogiri = event.get("tsumogiri", False)

        if pai == "?":
            return

        player = self.players[actor]
        player.river.append((pai, tsumogiri))
        player.riichi_ippatsu = False  # cleared after discard

        # Track tedashi/tsumogiri patterns
        if tsumogiri:
            player._consecutive_tsumogiri += 1
        else:
            player._consecutive_tsumogiri = 0
            player._tedashi_count += 1
            player._last_tedashi_turn = self.turn

        if actor == self.player_id:
            if pai in self.my_hand:
                self.my_hand.remove(pai)
            self.my_tsumo = None
        else:
            self._mark_visible(pai)
            # Track genbutsu for me
            player.safe_tiles_for_me.add(pai)

    def _handle_meld(self, event: dict, meld_type: str) -> None:
        actor = event.get("actor", -1)
        consumed = event.get("consumed", [])
        pai = event.get("pai", "?")

        meld = MeldInfo(
            meld_type=meld_type,
            tiles=consumed + [pai] if pai != "?" else consumed,
            from_player=event.get("target", -1),
        )
        self.players[actor].melds.append(meld)

        if actor == self.player_id:
            for t in consumed:
                if t in self.my_hand:
                    self.my_hand.remove(t)
        else:
            for t in consumed:
                self._mark_visible(t)

        # Melds break ippatsu for all riichi players
        for p in self.players:
            p.riichi_ippatsu = False

    def _handle_kan(self, event: dict, kan_type: str) -> None:
        actor = event.get("actor", -1)
        consumed = event.get("consumed", [])
        pai = event.get("pai", "?")

        meld = MeldInfo(
            meld_type=kan_type,
            tiles=consumed + ([pai] if pai != "?" else []),
            from_player=event.get("target", -1),
        )
        self.players[actor].melds.append(meld)

        if actor == self.player_id:
            for t in consumed:
                if t in self.my_hand:
                    self.my_hand.remove(t)
            if pai != "?" and pai in self.my_hand:
                self.my_hand.remove(pai)
        else:
            for t in consumed:
                self._mark_visible(t)
            if pai != "?":
                self._mark_visible(pai)

        for p in self.players:
            p.riichi_ippatsu = False

    def _handle_reach(self, event: dict) -> None:
        actor = event.get("actor", -1)
        self.players[actor].riichi_declared = True
        self.players[actor].riichi_turn = self.turn

    def _handle_reach_accepted(self, event: dict) -> None:
        actor = event.get("actor", -1)
        self.players[actor].riichi_declared = True
        self.players[actor].riichi_ippatsu = True
        self.players[actor].score -= 1000
        self.kyotaku += 1

    def _handle_dora(self, event: dict) -> None:
        dora_marker = event.get("dora_marker")
        if dora_marker:
            self.dora_indicators.append(dora_marker)
            self._mark_visible(dora_marker)

    def _mark_visible(self, tile: Tile) -> None:
        if tile == "?":
            return
        idx = tile_to_index(tile)
        if 0 <= idx < 34:
            self.visible_counts[idx] = min(4, self.visible_counts[idx] + 1)


def _hand_to_34(hand: List[Tile]) -> List[int]:
    """Convert hand tile list to 34-element count array."""
    counts = [0] * 34
    for t in hand:
        idx = tile_to_index(t)
        if 0 <= idx < 34:
            counts[idx] += 1
    return counts
