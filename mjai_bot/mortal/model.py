import pathlib

from .libriichi.mjai import Bot
from .libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

from ..mortal_common.model import (
    ChannelAttention, ResBlock, ResNet, Brain, AuxNet, DQN,
    MortalEngine, sample_top_p, load_ot_settings,
    load_model as _load_model,
)

_MODEL_DIR = pathlib.Path(__file__).parent
_ot_settings = load_ot_settings(_MODEL_DIR)


def load_model(seat: int) -> Bot:
    return _load_model(
        seat,
        bot_cls=Bot,
        obs_shape_fn=obs_shape,
        oracle_obs_shape_fn=oracle_obs_shape,
        action_space=ACTION_SPACE,
        model_dir=_MODEL_DIR,
        ot_settings=_ot_settings,
        react_batch_endpoint='/react_batch',
    )
