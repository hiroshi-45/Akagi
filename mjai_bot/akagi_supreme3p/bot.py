"""akagi_supreme3p bot: 3-player (sanma) variant of akagi_supreme.

Wraps Mortal3P's neural network with the same strategic overlays as
akagi_supreme, adapted for 3-player mahjong (no chi, different action
space, 3 active players).
"""
import json

from akagi.logging_utils import setup_logger
from . import model

logger = setup_logger("akagi_supreme3p_bot")


class Bot:
    """Supreme 3-player Mahjong bot combining Mortal3P AI with strategic intelligence."""

    def __init__(self):
        self.player_id: int = None
        self.model = None
        self.model_module = model
        self._player_state = None

    def react(self, events: str) -> str:
        try:
            events = json.loads(events)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse events: {events}, {e}")
            return json.dumps({"type": "none"}, separators=(",", ":"))

        return_action = None
        for e in events:
            if e["type"] == "start_game":
                self.player_id = e["id"]
                self.model = self.model_module.load_model(self.player_id)
                self._init_player_state()
                self._update_strategy(e)
                continue

            if self.model is None or self.player_id is None:
                logger.error("Model is not loaded yet")
                continue

            if e["type"] == "end_game":
                self._update_strategy(e)
                self.player_id = None
                self.model = None
                self._player_state = None
                continue

            self._update_strategy(e)
            self._update_player_state(e)

            if self._player_state is not None:
                try:
                    shanten = self._player_state.shanten
                    self._get_supreme_engine().update_shanten(shanten)
                except Exception:
                    pass

            return_action = self.model.react(json.dumps(e, separators=(",", ":")))

        ot_settings = self.model_module._ot_settings

        if return_action is None:
            if ot_settings.get('online'):
                raw_data = {
                    "type": "none",
                    "meta": {"online": self._get_is_online()},
                }
                return_action = json.dumps(raw_data, separators=(",", ":"))
            else:
                return_action = json.dumps({"type": "none"}, separators=(",", ":"))
            return return_action
        else:
            if ot_settings.get('online'):
                raw_data = json.loads(return_action)
                if "meta" not in raw_data:
                    raw_data["meta"] = {}
                raw_data["meta"]["online"] = self._get_is_online()
                return_action = json.dumps(raw_data, separators=(",", ":"))
            return return_action

    def _init_player_state(self) -> None:
        """Initialize a separate PlayerState for shanten tracking (3P)."""
        try:
            from ..mortal3p.libriichi3p.state import PlayerState
            self._player_state = PlayerState(self.player_id)
            event = {"type": "start_game", "names": ["0", "1", "2", "3"], "id": self.player_id}
            self._player_state.update(json.dumps(event, separators=(",", ":")))
        except Exception as e:
            logger.warning(f"Could not create PlayerState for shanten (3P): {e}")
            self._player_state = None

    def _update_player_state(self, event: dict) -> None:
        if self._player_state is None:
            return
        try:
            event_json = json.dumps(event, separators=(",", ":"))
            self._player_state.update(event_json)
        except Exception:
            pass

    def _update_strategy(self, event: dict) -> None:
        engine = self._get_supreme_engine()
        if engine is not None:
            try:
                engine.update_state(json.dumps(event, separators=(",", ":")))
            except Exception as e:
                logger.warning(f"Strategy update failed: {e}")

    def _get_supreme_engine(self):
        if self.model is not None and hasattr(self.model, 'engine'):
            engine = self.model.engine
            from ..akagi_supreme.supreme_engine import SupremeEngine
            if isinstance(engine, SupremeEngine):
                return engine
        return None

    def _get_is_online(self) -> bool:
        if self.model is not None and hasattr(self.model, 'engine'):
            engine = self.model.engine
            from ..akagi_supreme.supreme_engine import SupremeEngine
            if isinstance(engine, SupremeEngine):
                return getattr(engine.base_engine, 'is_online', False)
            return getattr(engine, 'is_online', False)
        return False
