from . import model
from ..mortal_common.bot import MortalBotBase


class Bot(MortalBotBase):
    def __init__(self):
        super().__init__(model_module=model)
