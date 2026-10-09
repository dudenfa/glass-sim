"""Run a trained (or baseline) policy as a normal market participant.

``PolicyAgent`` is an ``Agent`` that encodes its ``MarketView`` with the same
``ObservationEncoder`` used in training, applies the action mask, asks the
policy for an action and decodes it into orders. It can therefore be added to
any scenario, evaluated with the usual reports, and watched in the viewer.

Policies:

- ``ModelPolicy``: a saved ``MaskablePPO`` checkpoint (deterministic by default);
- ``RandomPolicy``: uniform over valid actions (baseline);
- ``HoldPolicy``: never trades (baseline: the cost of doing nothing is zero).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ..agents.base import Agent
from .actions import ACTION_NAMES, ActionConfig, ActionSpace
from .observation import ObservationConfig, ObservationEncoder


class HoldPolicy:
    name = "hold"

    def __call__(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        return 0


class RandomPolicy:
    name = "random"

    def __call__(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.choice(np.flatnonzero(mask)))


class ModelPolicy:
    """A MaskablePPO checkpoint saved by ``glass_sim.rl.train``."""

    name = "model"

    def __init__(self, model_path: Union[str, Path], deterministic: bool = True) -> None:
        from sb3_contrib import MaskablePPO

        self.path = Path(model_path)
        self.model = MaskablePPO.load(str(self.path), device="cpu")
        self.deterministic = deterministic

    def __call__(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return int(action)


def load_run_config(model_path: Union[str, Path]) -> Dict[str, Any]:
    """The ``config.json`` written next to a checkpoint by the training script."""
    path = Path(model_path)
    config_path = path.parent / "config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else {}


def encoder_and_actions_from_config(run_config: Dict[str, Any]) -> "tuple[ObservationEncoder, ActionSpace]":
    env = run_config.get("env", {})
    obs_cfg = ObservationConfig(**env.get("observation", {}))
    if env.get("private_signals"):
        obs_cfg = replace(obs_cfg, include_signal=True)
    act_cfg = ActionConfig(**env.get("actions", {}))
    return ObservationEncoder(obs_cfg), ActionSpace(act_cfg)


class PolicyAgent(Agent):
    """A market participant driven by a policy over the RL observation and action spaces."""

    agent_type = "rl_agent"
    default_prefix = "rl"

    def __init__(
        self,
        policy: Any,
        wake_interval: float = 5.0,
        encoder: Optional[ObservationEncoder] = None,
        actions: Optional[ActionSpace] = None,
        agent_type: Optional[str] = None,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.policy = policy
        self.encoder = encoder or ObservationEncoder()
        self.actions = actions or ActionSpace()
        if agent_type:
            self.agent_type = agent_type
        self.action_counts = np.zeros(self.actions.n, dtype=np.int64)
        self.decisions = 0

    def on_wakeup(self, view, rng):
        obs = self.encoder.encode(view)
        mask = self.actions.mask(view)
        action = int(self.policy(obs, mask, rng))
        if not mask[action]:
            action = 0
        self.action_counts[action] += 1
        self.decisions += 1
        return self.actions.decode(action, view)

    def action_distribution(self) -> Dict[str, float]:
        total = max(1, int(self.action_counts.sum()))
        return {name: round(float(count) / total, 4) for name, count in zip(ACTION_NAMES, self.action_counts)}


def policy_agent_from_checkpoint(model_path: Union[str, Path], deterministic: bool = True, agent_type: str = "rl_agent") -> PolicyAgent:
    run_config = load_run_config(model_path)
    encoder, actions = encoder_and_actions_from_config(run_config)
    wake = float(run_config.get("env", {}).get("wake_interval", 5.0))
    return PolicyAgent(ModelPolicy(model_path, deterministic), wake_interval=wake, encoder=encoder, actions=actions, agent_type=agent_type)
