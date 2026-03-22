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

    def estimate_open_hand_points(self, round_wind: str = "E",
                                   seat_wind: str = "E",
                                   doras: list = None) -> int:
        """Estimate minimum points for an open hand based on visible melds.

        Used for threat assessment: how much would dealing in cost?
        Returns estimated points (ron, non-dealer baseline).
        """
        if not self.is_open():
            return 0

        han = 0
        fu = 30  # base fu for open hand

        # Count yakuhai from melds
        dragon_count = 0
        for m in self.melds:
            if m.meld_type in ("pon", "daiminkan", "kakan", "ankan"):
                if len(m.tiles) > 0:
                    t = tile_base(m.tiles[0])
                    if t in YAKUHAI_HONORS:  # 白發中
                        han += 1
                        dragon_count += 1
                    if t == round_wind:
                        han += 1
                    if t == seat_wind:
                        han += 1
                    # Ankan adds extra fu
                    if m.meld_type == "ankan":
                        if t in HONORS:
                            fu += 32
                        else:
                            fu += 16
                    elif m.meld_type in ("pon", "daiminkan", "kakan"):
                        if t in HONORS:
                            fu += 4
                        else:
                            fu += 2

        # Small three dragons (小三元): 2 dragon pons = at least 4 han
        if dragon_count >= 2:
            han += 2  # 小三元 is worth 2 han on its own + the 2 yakuhai pons

        # Honitsu/Chinitsu from melds + river
        target_suit = self.detect_honitsu_chinitsu()
        if target_suit is not None:
            han += 2  # open honitsu

        # Toitoi
        if self.detect_toitoi_signal():
            han += 2  # open toitoi

        # Dora in melds
        if doras:
            for m in self.melds:
                for t in m.tiles:
                    base = tile_base(t)
                    for d in doras:
                        if base == d:
                            han += 1
                    if t.endswith("r"):
                        han += 1

        if han == 0:
            return 0  # no yaku visible

        # Estimate points from han/fu
        return _calculate_points(han, fu, self.is_dealer, False)

    def tedashi_after_tsumogiri_streak(self) -> bool:
        """Whether last event was tedashi after a streak of tsumogiri.

        This is a strong tenpai signal used by top players.
        """
        return self._consecutive_tsumogiri >= 3 and self._tedashi_count > 0

    def detect_honitsu_from_river(self) -> Optional[str]:
        """Detect honitsu tendency from river discards (works for closed hands too).

        If a player discards 2 suits heavily while keeping 1 suit,
        they're likely going for honitsu/chinitsu.
        """
        river_counts = self.river_suit_counts()
        total_number = river_counts["m"] + river_counts["p"] + river_counts["s"]
        if total_number < 6:
            return None  # not enough data

        for target in ("m", "p", "s"):
            others = sum(v for k, v in river_counts.items() if k != target and k != "z")
            if others >= total_number * 0.75:
                return target
        return None

    def apparent_threat_level(self, current_turn: int = 0,
                              round_wind: str = "E",
                              seat_wind: str = "E") -> float:
        """Heuristic threat level from discards, melds, and behavioral patterns.

        Enhanced with:
        - Tedashi pattern reading (tsumogiri streak → tedashi = tenpai signal)
        - Suit bias in river (honitsu/chinitsu detection for open AND closed hands)
        - Toitoi signal
        - Turn-aware riichi assessment
        - Wind yakuhai detection (seat wind, round wind)
        - Compound dragon detection (小三元 potential)
        - Hand-cut content analysis (middle tiles = stronger hand signal)
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
            yakuhai_count = 0
            dragon_count = 0
            for m in self.melds:
                if m.meld_type == "pon" and len(m.tiles) > 0:
                    t = tile_base(m.tiles[0])
                    if t in YAKUHAI_HONORS:  # 白發中
                        yakuhai_count += 1
                        dragon_count += 1
                        threat += 0.6
                    elif t == round_wind or t == seat_wind:
                        yakuhai_count += 1
                        threat += 0.5  # wind yakuhai

            # Double yakuhai
            if yakuhai_count >= 2:
                threat += 0.4

            # Small three dragons potential (小三元)
            if dragon_count >= 2:
                threat += 0.6

            # Honitsu/Chinitsu signal (open hand)
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

        # === Closed hand honitsu detection from river ===
        if not self.is_open() and not self.riichi_declared:
            river_honitsu = self.detect_honitsu_from_river()
            if river_honitsu is not None:
                threat += 0.4

        # === Tedashi pattern: tsumogiri streak then tedashi = tenpai signal ===
        if not self.riichi_declared and self.tedashi_after_tsumogiri_streak():
            threat += 0.6

        # === Hand-cut content analysis ===
        # Discarding middle tiles (3-7) from hand = stronger hand signal
        if not self.riichi_declared:
            hand_cuts = self.hand_cut_tiles()
            if len(hand_cuts) >= 3:
                mid_count = 0
                for t in hand_cuts[-4:]:  # check recent hand-cuts
                    s, r, _ = parse_tile(t)
                    if r is not None and 3 <= r <= 7:
                        mid_count += 1
                if mid_count >= 2:
                    threat += 0.3  # discarding valuable middle tiles = hand is strong

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
    _is_tonpu: bool = False  # True for east-only (東風戦)

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
        """Are we in the final round (オーラス)?

        Handles both east-only (東風戦: E4) and east-south (半荘: S4).
        """
        if self._is_tonpu:
            return self.round_wind == "E" and self.round_number == 4
        return self.round_wind == "S" and self.round_number == 4

    @property
    def is_south(self) -> bool:
        return self.round_wind in ("S", "W", "N")

    @property
    def is_late_game(self) -> bool:
        if self._is_tonpu:
            return self.round_wind == "E" and self.round_number >= 3
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
        """Estimate useful tile count for improving hand (reducing shanten).

        Uses a simplified shanten-reduction check: for each possible draw,
        simulate adding it and removing each tile, checking if the resulting
        hand has better mentsu/partial-mentsu structure.

        This is more accurate than "any connected tile" — it requires the
        draw to actually improve the hand structure.
        """
        hand_counts = _hand_to_34(self.my_hand)
        current_deficiency = _estimate_deficiency(hand_counts)
        acceptance = 0

        for idx in range(34):
            remaining = max(0, 4 - self.visible_counts[idx])
            if remaining <= 0:
                continue

            # Simulate drawing this tile
            hand_counts[idx] += 1

            # Check if any discard reduces deficiency
            improved = False
            for discard_idx in range(34):
                if hand_counts[discard_idx] <= 0:
                    continue
                if discard_idx == idx and hand_counts[discard_idx] <= 1:
                    continue  # can't discard the only copy we just drew

                hand_counts[discard_idx] -= 1
                new_deficiency = _estimate_deficiency(hand_counts)
                hand_counts[discard_idx] += 1

                if new_deficiency < current_deficiency:
                    improved = True
                    break

            hand_counts[idx] -= 1

            if improved:
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

    def _opponent_wind(self, seat: int) -> str:
        """Get a player's seat wind."""
        winds = ["E", "S", "W", "N"]
        return winds[(seat - self.dealer) % 4]

    def _threat_of(self, seat: int) -> float:
        """Get threat level of a specific player with full context."""
        return self.players[seat].apparent_threat_level(
            self.turn, self.round_wind, self._opponent_wind(seat))

    def threat_level_total(self) -> float:
        """Sum of all opponents' threat levels."""
        return sum(self._threat_of(i)
                   for i in range(4) if i != self.player_id)

    def max_opponent_threat(self) -> float:
        return max((self._threat_of(i)
                    for i in range(4) if i != self.player_id), default=0.0)

    def highest_threat_player(self) -> Optional[int]:
        """Return seat index of the most threatening opponent."""
        best_i = None
        best_t = 0.0
        for i in range(4):
            if i == self.player_id:
                continue
            t = self._threat_of(i)
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

    def min_han_for_points(self, target_points: int, is_tsumo: bool = False,
                           fu: int = 30) -> int:
        """Minimum han needed to reach target_points.

        Uses actual point calculation with configurable fu (default 30).
        Common fu values: 20 (pinfu tsumo), 25 (chiitoitsu), 30, 40, 50.
        """
        if target_points <= 0:
            return 1
        for han in range(1, 14):
            pts = _calculate_points(han, fu, self.is_dealer_me, is_tsumo)
            if pts >= target_points:
                return han
        return 13  # yakuman

    def points_needed_direct_hit(self, target_seat: int, target_placement: int) -> int:
        """Points needed via direct hit (ron) on target_seat to reach target_placement.

        Direct hit transfers points: target loses what we gain (+ honba).
        This means we need less than half the raw point difference.
        """
        target_score = self.players[target_seat].score
        # Find the score of the player at target_placement
        sorted_scores = sorted(
            [(p.score, i) for i, p in enumerate(self.players)],
            key=lambda x: (-x[0], x[1])
        )
        if target_placement < 1 or target_placement > 4:
            return 0
        threshold_score = sorted_scores[target_placement - 1][0]
        threshold_seat = sorted_scores[target_placement - 1][1]

        my_s = self.my_score
        if my_s >= threshold_score and self.my_placement <= target_placement:
            return 0

        # Direct hit: we gain X points, target loses X points
        # New scores: my_s + X >= threshold, target_score - X (may drop)
        # We need: my_s + X > threshold_score (or >= with seat priority)
        needed = threshold_score - my_s
        if self.player_id > threshold_seat:
            needed += 100
        return max(0, needed)

    def noten_penalty_effect(self) -> int:
        """Estimate point change from noten penalty (ノーテン罰符) at ryukyoku.

        Returns negative value if we'd lose points (noten), positive if we'd gain.
        Assumes we are noten; actual tenpai status should be checked by caller.
        """
        # Noten penalty: 3000 pts split among tenpai/noten players
        # Worst case (only we are noten): -3000
        # If 2 noten: -1500 each, 1 tenpai gets 3000
        # If 3 noten: -1000 each, 1 tenpai gets 3000
        # Conservative estimate: assume we're the only noten
        return -3000

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
        # Detect east-south game (if we ever see south wind, it's not tonpu)
        if self.round_wind == "S":
            self._is_tonpu = False
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


def _estimate_deficiency(counts: List[int]) -> int:
    """Estimate hand deficiency (roughly correlates with shanten).

    Counts how many more tiles are needed to complete 4 mentsu + 1 jantai.
    Uses a greedy approach: extract complete mentsu first, then partial blocks.

    Lower deficiency = closer to tenpai.
    """
    best = 8  # worst case: 8 tiles away (shanten=8)

    # Try each tile as the pair (jantai)
    for pair_idx in range(34):
        if counts[pair_idx] < 2:
            continue
        counts[pair_idx] -= 2
        mentsu, partial = _count_mentsu_and_partial(counts)
        # Need 4 mentsu total; each mentsu = 0 deficiency, each partial = 1
        needed = 4 - mentsu
        if needed <= 0:
            deficiency = 0
        else:
            # partials can become mentsu with 1 more tile each
            usable_partial = min(partial, needed)
            deficiency = (needed - usable_partial) * 2 + usable_partial
        best = min(best, deficiency)
        counts[pair_idx] += 2

    # Also try without a pair (tanki wait)
    mentsu, partial = _count_mentsu_and_partial(counts)
    needed = 4 - mentsu
    if needed <= 0:
        deficiency = 1  # need pair only
    else:
        usable_partial = min(partial, needed)
        deficiency = (needed - usable_partial) * 2 + usable_partial + 1
    best = min(best, deficiency)

    return best


def _count_mentsu_and_partial(counts: List[int]) -> tuple:
    """Count complete mentsu and partial blocks in hand (greedy).

    Returns (mentsu_count, partial_count).
    """
    c = list(counts)  # work on copy
    mentsu = 0
    partial = 0

    # Extract complete mentsu first (greedy)
    # Kotsu (triplets) for honors
    for i in range(27, 34):
        while c[i] >= 3:
            c[i] -= 3
            mentsu += 1

    # Shuntsu (sequences) for number tiles
    for suit_start in (0, 9, 18):
        for rank in range(7):
            idx = suit_start + rank
            while c[idx] >= 1 and c[idx + 1] >= 1 and c[idx + 2] >= 1:
                c[idx] -= 1
                c[idx + 1] -= 1
                c[idx + 2] -= 1
                mentsu += 1

    # Remaining kotsu for number tiles
    for i in range(27):
        while c[i] >= 3:
            c[i] -= 3
            mentsu += 1

    # Count partial blocks (pairs, adjacent pairs for sequences)
    # Pairs
    for i in range(34):
        if c[i] >= 2:
            c[i] -= 2
            partial += 1

    # Adjacent/skip pairs (partial sequences)
    for suit_start in (0, 9, 18):
        for rank in range(8):
            idx = suit_start + rank
            if c[idx] >= 1 and c[idx + 1] >= 1:
                c[idx] -= 1
                c[idx + 1] -= 1
                partial += 1
        for rank in range(7):
            idx = suit_start + rank
            if c[idx] >= 1 and c[idx + 2] >= 1:
                c[idx] -= 1
                c[idx + 2] -= 1
                partial += 1

    return mentsu, partial


def _ceil100(n: int) -> int:
    """Round up to nearest 100."""
    return ((n + 99) // 100) * 100


def _calculate_points(han: int, fu: int, is_dealer: bool, is_tsumo: bool) -> int:
    """Calculate points from han and fu using standard Mahjong point tables.

    Handles all fu values (20, 25, 30, 40, 50, etc.) and mangan+ thresholds.
    """
    # Mangan and above: fixed values
    if han >= 13:
        return 48000 if is_dealer else 32000  # yakuman
    if han >= 11:
        return 36000 if is_dealer else 24000  # sanbaiman
    if han >= 8:
        return 24000 if is_dealer else 16000  # baiman
    if han >= 6:
        return 18000 if is_dealer else 12000  # haneman
    if han >= 5:
        return 12000 if is_dealer else 8000  # mangan

    # Chiitoitsu special case (25fu fixed, 2han minimum)
    if fu == 25:
        if han < 2:
            han = 2
        basic = 25 * (2 ** (han + 2))
        if basic >= 2000:
            return 12000 if is_dealer else 8000
        if is_dealer:
            if is_tsumo:
                per = _ceil100(basic * 2)
                return per * 3
            return _ceil100(basic * 6)
        if is_tsumo:
            ko = _ceil100(basic)
            oya = _ceil100(basic * 2)
            return ko * 2 + oya
        return _ceil100(basic * 4)

    # Standard calculation
    basic = fu * (2 ** (han + 2))

    # Mangan cap
    if basic >= 2000:
        return 12000 if is_dealer else 8000

    if is_dealer:
        if is_tsumo:
            per = _ceil100(basic * 2)
            return per * 3
        return _ceil100(basic * 6)

    if is_tsumo:
        ko = _ceil100(basic)
        oya = _ceil100(basic * 2)
        return ko * 2 + oya
    return _ceil100(basic * 4)
