"""akagi_supreme bot: the strongest Mahjong AI.

Wraps Mortal's neural network with strategic overlays for:
- Placement-aware push/fold decisions
- Safety-integrated tile selection
- Situation-dependent riichi/damaten choices
- Intelligent meld acceptance/rejection
- All-last and endgame optimization
"""
import json

from akagi.logging_utils import setup_logger
from . import model

logger = setup_logger("akagi_supreme_bot")


class Bot:
    """Supreme Mahjong bot combining Mortal AI with strategic intelligence.

    Uses Mortal's neural network for base decisions and overlays strategic
    adjustments based on game context, placement, and opponent behavior.
    """

    def __init__(self):
        self.player_id: int = None
        self.model = None
        self._engine = None
        self.model_module = model
        self._player_state = None  # libriichi PlayerState for shanten tracking

    def react(self, events: str) -> str:
        """Process MJAI events and return an action.

        The flow:
        1. Parse events
        2. For each event, update both:
           a. Our strategy engine (via SupremeEngine.update_state)
           b. libriichi's PlayerState (for shanten)
           c. The libriichi Bot (for Mortal inference)
        3. The libriichi Bot calls SupremeEngine.react_batch() internally
        4. SupremeEngine intercepts and adjusts the action
        5. Return the adjusted action
        """
        try:
            events = json.loads(events)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse events: {events}, {e}")
            return json.dumps({"type": "none"}, separators=(",", ":"))

        return_action = None
        for e in events:
            if e["type"] == "start_game":
                self.player_id = e["id"]
                self.model, self._engine = self.model_module.load_model(self.player_id)
                # Create a PlayerState for shanten tracking
                self._init_player_state()
                # Feed start_game to strategy engine
                self._update_strategy(e)
                continue

            if self.model is None or self.player_id is None:
                logger.error("Model is not loaded yet")
                continue

            if e["type"] == "end_game":
                self._update_strategy(e)
                self.player_id = None
                self.model = None
                self._engine = None
                self._player_state = None
                continue

            # Update strategy engine with event
            self._update_strategy(e)

            # Update PlayerState for shanten
            self._update_player_state(e)

            # Update shanten in strategy engine
            if self._player_state is not None:
                try:
                    shanten = self._player_state.shanten
                    self._get_supreme_engine().update_shanten(shanten)
                except Exception:
                    pass

            # Feed to libriichi Bot for Mortal inference
            # (SupremeEngine.react_batch will be called internally)
            return_action = self.model.react(json.dumps(e, separators=(",", ":")))

        ot_settings = self.model_module._ot_settings

        engine = self._get_supreme_engine()
        thought = []
        mortal_action = ""
        supreme_action = ""
        if engine is not None and hasattr(engine, "strategy"):
            thought = getattr(engine.strategy, "last_thought", [])
            mortal_action = getattr(engine.strategy, "last_mortal_action_name", "")
            supreme_action = getattr(engine.strategy, "last_supreme_action_name", "")

        if return_action is None:
            raw_data = {
                "type": "none",
                "meta": {
                    "thought": thought,
                    "mortal_action": mortal_action,
                    "supreme_action": supreme_action,
                },
            }
            if ot_settings.get('online'):
                raw_data["meta"]["online"] = self._get_is_online()
            return_action = json.dumps(raw_data, separators=(",", ":"))
            return return_action
        else:
            raw_data = json.loads(return_action)
            if "meta" not in raw_data:
                raw_data["meta"] = {}
            raw_data["meta"]["thought"] = thought
            raw_data["meta"]["mortal_action"] = mortal_action
            raw_data["meta"]["supreme_action"] = supreme_action
            if ot_settings.get('online'):
                raw_data["meta"]["online"] = self._get_is_online()
            return_action = json.dumps(raw_data, separators=(",", ":"))
            return return_action

    def _init_player_state(self) -> None:
        """Initialize a separate PlayerState for shanten tracking."""
        try:
            from ..mortal.libriichi.state import PlayerState
            self._player_state = PlayerState(self.player_id)
            # Feed start_game to PlayerState
            event = {"type": "start_game", "names": ["0", "1", "2", "3"], "id": self.player_id}
            self._player_state.update(json.dumps(event, separators=(",", ":")))
        except Exception as e:
            logger.warning(f"Could not create PlayerState for shanten: {e}")
            self._player_state = None

    def _update_player_state(self, event: dict) -> None:
        """Feed event to PlayerState for shanten updates."""
        if self._player_state is None:
            return
        try:
            event_json = json.dumps(event, separators=(",", ":"))
            self._player_state.update(event_json)
        except Exception:
            pass  # PlayerState may reject some events; that's OK

    def _update_strategy(self, event: dict) -> None:
        """Feed event to the strategy engine."""
        engine = self._get_supreme_engine()
        if engine is not None:
            try:
                engine.update_state(json.dumps(event, separators=(",", ":")))
            except Exception as e:
                logger.warning(f"Strategy update failed: {e}")

    def _get_supreme_engine(self):
        """Get the SupremeEngine directly cached from load_model."""
        return self._engine

    def _get_is_online(self) -> bool:
        """Check if the model's engine is currently online."""
        if self.model is not None and hasattr(self.model, 'engine'):
            engine = self.model.engine
            from .supreme_engine import SupremeEngine
            if isinstance(engine, SupremeEngine):
                return getattr(engine.base_engine, 'is_online', False)
            return getattr(engine, 'is_online', False)
        return False
