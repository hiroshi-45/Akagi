import json

from akagi.logging_utils import setup_logger

logger = setup_logger("mjai_mortal_bot")


class MortalBotBase:
    """Base class for mortal bot implementations (4P and 3P).

    Subclasses must set `self.model_module` to the appropriate model module
    (e.g., mjai_bot.mortal.model or mjai_bot.mortal3p.model).
    """

    def __init__(self, model_module):
        self.player_id: int = None
        self.model = None
        self.model_module = model_module

    def react(self, events: str) -> str:
        """Process events and return an action as a JSON string.

        One `start_game` event must be sent before any other events.
        Once the bot receives a `start_game` event, it will reinitialize itself and set the player_id.

        `start_game` event can be sent any time to reset the bot.
        `end_game` event can be sent to set model to None.

        :param events: JSON string of events
        :return: JSON string of action
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
                self.model = self.model_module.load_model(self.player_id)
                continue
            if self.model is None or self.player_id is None:
                logger.error("Model is not loaded yet")
                continue
            if e["type"] == "end_game":
                self.player_id = None
                self.model = None
                continue
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

    def _get_is_online(self) -> bool:
        """Check if the model's engine is currently online."""
        if self.model is not None and hasattr(self.model, 'engine'):
            return getattr(self.model.engine, 'is_online', False)
        return False
