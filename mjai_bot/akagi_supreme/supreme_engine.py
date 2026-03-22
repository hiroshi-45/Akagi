# -*- coding: utf-8 -*-
"""SupremeEngine: wraps MortalEngine to intercept and adjust actions.

This is the glue between Mortal's neural network and our strategy system.
It implements the same interface as MortalEngine (react_batch) so it can
be used as a drop-in replacement with libriichi's Bot.
"""
from __future__ import annotations

from typing import List, Tuple

from .strategy_engine import StrategyEngine

from akagi.logging_utils import setup_logger

logger = setup_logger("akagi_supreme")


class SupremeEngine:
    """Engine wrapper that applies strategic overlays to Mortal's output.

    Presents the same interface as MortalEngine so libriichi's Bot
    can use it transparently.
    """

    def __init__(self, base_engine):
        self.base_engine = base_engine
        self.strategy = StrategyEngine()

        # Forward all attributes from base engine that libriichi might access
        self.engine_type = 'mortal'
        self.device = base_engine.device
        self.is_oracle = base_engine.is_oracle
        self.version = base_engine.version
        self.enable_amp = base_engine.enable_amp
        self.enable_quick_eval = base_engine.enable_quick_eval
        self.enable_rule_based_agari_guard = base_engine.enable_rule_based_agari_guard
        self.name = base_engine.name
        self.boltzmann_epsilon = base_engine.boltzmann_epsilon
        self.boltzmann_temp = base_engine.boltzmann_temp
        self.top_p = base_engine.top_p
        self.stochastic_latent = base_engine.stochastic_latent
        self.brain = base_engine.brain
        self.dqn = base_engine.dqn

    @property
    def is_online(self):
        return self.base_engine.is_online

    @is_online.setter
    def is_online(self, value):
        self.base_engine.is_online = value

    @property
    def ot_settings(self):
        return self.base_engine.ot_settings

    def react_batch(self, obs, masks, invisible_obs):
        """Intercept Mortal's react_batch to apply strategic adjustments.

        The flow:
        1. Get Mortal's raw output (actions, Q-values, masks, greedy flags)
        2. For each action in the batch, apply strategy engine adjustments
        3. Return adjusted actions with original Q-values
        """
        # Get Mortal's raw decisions
        actions, q_out, masks_out, is_greedy = self.base_engine.react_batch(
            obs, masks, invisible_obs
        )

        # Apply strategic adjustments to each item in the batch
        adjusted_actions = []
        for i in range(len(actions)):
            original_action = actions[i]
            q_values = q_out[i]
            mask = [bool(m) for m in masks_out[i]]
            greedy = is_greedy[i]

            try:
                adjusted = self.strategy.adjust_action(
                    q_values, mask, original_action, greedy
                )
                adjusted_actions.append(adjusted)

                if adjusted != original_action:
                    logger.debug(
                        f"Strategy override: {original_action} -> {adjusted} "
                        f"(placement={self.strategy.gs.my_placement}, "
                        f"threat={self.strategy.gs.max_opponent_threat():.1f})"
                    )
            except Exception as e:
                logger.warning(f"Strategy engine error, using Mortal's action: {e}")
                adjusted_actions.append(original_action)

        return adjusted_actions, q_out, masks_out, is_greedy

    def update_state(self, event_json: str) -> None:
        """Called by our bot wrapper to feed events to the strategy engine."""
        import json
        try:
            event = json.loads(event_json)
            self.strategy.process_event(event)
        except Exception as e:
            logger.warning(f"Failed to update strategy state: {e}")

    def update_shanten(self, shanten: int) -> None:
        """Update shanten count from libriichi's PlayerState."""
        self.strategy.set_shanten(shanten)
