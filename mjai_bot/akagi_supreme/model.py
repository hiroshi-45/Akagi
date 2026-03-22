"""Model loading for akagi_supreme.

Uses Mortal's neural network (Brain + DQN) for inference but wraps it
with our StrategyEngine for enhanced action selection.
"""
import json
import pathlib

from ..mortal.libriichi.mjai import Bot as LibriichBot
from ..mortal.libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

from ..mortal_common.model import (
    Brain, DQN, MortalEngine, load_ot_settings,
    ChannelAttention, ResBlock, ResNet, AuxNet, sample_top_p,
)
from .supreme_engine import SupremeEngine

import torch

# Use Mortal's model directory for weights
_MORTAL_DIR = pathlib.Path(__file__).parent.parent / "mortal"
_MODEL_DIR = pathlib.Path(__file__).parent
_ot_settings = load_ot_settings(_MORTAL_DIR)


def load_model(seat: int) -> LibriichBot:
    """Load a mortal model wrapped with supreme strategy engine.

    Uses Mortal's mortal.pth weights but wraps inference in SupremeEngine
    which applies strategic overlays.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    control_state_file = _MORTAL_DIR / "mortal.pth"
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

    # Create the base MortalEngine
    base_engine = MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        version=version,
        ot_settings=_ot_settings,
        react_batch_endpoint='/react_batch',
        device=device,
        enable_amp=False,
        enable_quick_eval=False,
        enable_rule_based_agari_guard=True,
        name='akagi_supreme',
    )

    # Wrap with SupremeEngine for strategic overlays
    supreme = SupremeEngine(base_engine)

    bot = LibriichBot(supreme, seat)
    return bot
