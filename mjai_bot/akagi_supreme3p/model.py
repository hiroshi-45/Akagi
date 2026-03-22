"""Model loading for akagi_supreme3p (3-player variant).

Uses Mortal3P's neural network but wraps it with SupremeEngine
for strategic overlays, configured with 3P action indices.
"""
import json
import pathlib

from ..mortal3p.libriichi3p.mjai import Bot as LibriichBot
from ..mortal3p.libriichi3p.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

from ..mortal_common.model import (
    Brain, DQN, MortalEngine, load_ot_settings,
    ChannelAttention, ResBlock, ResNet, AuxNet, sample_top_p,
)
from ..akagi_supreme.supreme_engine import SupremeEngine
from ..akagi_supreme.strategy_engine import ACTION_CONFIG_3P

import torch

# Use mortal3p's model directory for weights
_MORTAL3P_DIR = pathlib.Path(__file__).parent.parent / "mortal3p"
_MODEL_DIR = pathlib.Path(__file__).parent
_ot_settings = load_ot_settings(_MORTAL3P_DIR)


def load_model(seat: int) -> LibriichBot:
    """Load a mortal3p model wrapped with supreme strategy engine (3P)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    control_state_file = _MORTAL3P_DIR / "mortal.pth"
    state = torch.load(control_state_file, map_location=device)

    version = state['config']['control']['version']
    brain = Brain(
        obs_shape_fn=obs_shape,
        oracle_obs_shape_fn=oracle_obs_shape,
        version=version,
        conv_channels=state['config']['resnet']['conv_channels'],
        num_blocks=state['config']['resnet']['num_blocks'],
    ).eval()
    dqn = DQN(version=version, action_space=ACTION_SPACE).eval()
    brain.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    base_engine = MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        version=version,
        ot_settings=_ot_settings,
        react_batch_endpoint='/react_batch_3p',
        device=device,
        enable_amp=False,
        enable_quick_eval=False,
        enable_rule_based_agari_guard=True,
        name='akagi_supreme3p',
    )

    # Wrap with SupremeEngine using 3P action config
    supreme = SupremeEngine(base_engine, action_config=ACTION_CONFIG_3P)

    bot = LibriichBot(supreme, seat)
    return bot
