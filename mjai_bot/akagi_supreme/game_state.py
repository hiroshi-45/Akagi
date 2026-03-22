# -*- coding: utf-8 -*-
"""Comprehensive game state tracker for strategic decision-making.

Tracks all observable information throughout a game and provides
derived metrics used by the strategy engine.
"""
from __future__ import annotations

import json
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

    def river_tiles(self) -> List[Tile]:
        return [t for t, _ in self.river]

    def hand_cut_tiles(self) -> List[Tile]:
        """Tiles discarded by hand (not tsumogiri)."""
        return [t for t, tg in self.river if not tg]

    def is_open(self) -> bool:
        return any(m.meld_type in ("chi", "pon", "daiminkan", "kakan") for m in self.melds)

    def num_melds(self) -> int:
        return len(self.melds)

    def apparent_threat_level(self) -> float:
        """Heuristic threat level from discards and melds."""
        threat = 0.0
        if self.riichi_declared:
            threat += 1.5
            if self.riichi_turn <= 6:
                threat += 0.5  # early riichi is scarier
        if self.is_open():
            n = self.num_melds()
            # Check for obvious yakuhai pon
            for m in self.melds:
                if m.meld_type == "pon" and len(m.tiles) > 0 and m.tiles[0] in YAKUHAI_HONORS:
                    threat += 0.6
            if n >= 3:
                threat += 0.7  # honitsu / chinitsu likely
            elif n >= 2:
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

    def count_dora_in_hand(self) -> int:
        """Count dora tiles in my hand."""
        dora_set = self.doras
        count = 0
        for t in self.my_hand:
            # Check normal dora
            base = t.rstrip("r")
            for d in dora_set:
                if base == d or (t.endswith("r") and d.startswith("5") and d[-1] == base[-1]):
                    count += 1
            # Red dora
            if t.endswith("r"):
                count += 1
        return count

    def threat_level_total(self) -> float:
        """Sum of all opponents' threat levels."""
        return sum(self.players[i].apparent_threat_level()
                   for i in range(4) if i != self.player_id)

    def max_opponent_threat(self) -> float:
        return max((self.players[i].apparent_threat_level()
                    for i in range(4) if i != self.player_id), default=0.0)

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

        # Clear ippatsu for this player if they draw
        if self.players[actor].riichi_ippatsu:
            # ippatsu is valid only for the first go-around
            pass  # cleared on next discard or meld

    def _handle_dahai(self, event: dict) -> None:
        actor = event.get("actor", -1)
        pai = event.get("pai", "?")
        tsumogiri = event.get("tsumogiri", False)

        if pai == "?":
            return

        player = self.players[actor]
        player.river.append((pai, tsumogiri))
        player.riichi_ippatsu = False  # cleared after discard

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
