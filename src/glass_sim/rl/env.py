"""Gymnasium environment: one learning agent trading in the perpetual-futures market.

Each episode builds a fresh market from a scenario, adds the learning agent as
an extra participant (an ``ExternalAgent`` with its own collateral and wake
rate), and advances the simulation until the agent must decide. A step is one
decision: the action is sent to the exchange (with latency), the market runs
until the agent's next wakeup, and the reward measures what happened to the
account's exit value in between (see ``reward.py``).

The episode ends when the market reaches the end of the episode duration
(``truncated``) or when the agent is liquidated out of the market
(``terminated``).

``action_masks()`` returns the valid actions for ``sb3_contrib.MaskablePPO``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from .. import scenarios
from ..agents.base import ExternalAgent
from ..config import AgentSpec, MarketConfig
from ..market import PerpetualMarket
from ..types import Transparency
from .actions import ACTION_NAMES, ActionConfig, ActionSpace
from .observation import ObservationConfig, ObservationEncoder
from .reward import RewardConfig, RewardTracker

LEARNER_ID = "rl_01"


@dataclass(frozen=True)
class EnvConfig:
    """Everything that defines a training environment."""

    scenario: str = "default"
    transparency: Optional[str] = None
    episode_duration: float = 3600.0
    train_seeds: tuple[int, ...] = tuple(range(1, 9))
    collateral: float = 100_000.0
    wake_interval: float = 5.0
    private_signals: bool = False
    news_rate_per_hour: Optional[float] = None
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def market_config(self, seed: int, duration: Optional[float] = None) -> MarketConfig:
        """The scenario plus the learner as an extra external participant."""
        config = scenarios.get(self.scenario, seed=seed, duration=float(duration or self.episode_duration))
        if self.transparency:
            config = replace(config, transparency=Transparency(self.transparency))
        if self.news_rate_per_hour is not None:
            config = replace(config, news=replace(config.news, rate_per_hour=self.news_rate_per_hour))
        learner = AgentSpec(
            ExternalAgent,
            count=1,
            collateral=self.collateral,
            params={"wake_interval": self.wake_interval},
            prefix="rl",
            private_signals=self.private_signals,
            external=True,
        )
        return replace(config, agents=config.agents + (learner,))


class PerpTradingEnv(gym.Env):
    """Single-learner Gymnasium environment; see the module docstring for the episode semantics."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        obs_config = self.config.observation
        if self.config.private_signals and not obs_config.include_signal:
            obs_config = replace(obs_config, include_signal=True)
        self.encoder = ObservationEncoder(obs_config)
        self.actions = ActionSpace(self.config.actions)
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(self.encoder.size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.actions.n)
        self.market: Optional[PerpetualMarket] = None
        self.tracker: Optional[RewardTracker] = None
        self._episode = 0
        # Parallel workers use disjoint seed sequences: worker k of n takes seeds k, k+n, k+2n, ...
        self.seed_offset = 0
        self.seed_stride = 1
        self.action_counts = np.zeros(self.actions.n, dtype=np.int64)

    # ------------------------------------------------------------- helpers

    @property
    def view(self):
        """The learner's ``MarketView``."""
        return self.market.view(LEARNER_ID)

    def _advance(self) -> bool:
        """Run the market until the learner must decide. False if the episode is over."""
        market = self.market
        account = market.accounts[LEARNER_ID]
        while market.pending_decision is None and not market.finished and account.active:
            market.step()
        return market.pending_decision == LEARNER_ID

    def action_masks(self) -> np.ndarray:
        """Valid actions for MaskablePPO; only ``hold`` when no decision is pending."""
        if self.market is None or self.market.pending_decision != LEARNER_ID:
            mask = np.zeros(self.actions.n, dtype=bool)
            mask[0] = True
            return mask
        return self.actions.mask(self.view)

    def episode_seed(self) -> int:
        """Seed of the next episode, striding through the training seeds so parallel workers never share a world."""
        seeds = self.config.train_seeds
        return int(seeds[(self.seed_offset + self._episode * self.seed_stride) % len(seeds)])

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        """Build a fresh market and advance to the learner's first decision."""
        super().reset(seed=seed)
        options = options or {}
        market_seed = int(options["market_seed"]) if "market_seed" in options else self.episode_seed()
        duration = options.get("duration")
        self._episode += 1
        self.market = PerpetualMarket(self.config.market_config(market_seed, duration))
        self.tracker = RewardTracker(self.market, LEARNER_ID, self.config.reward)
        self._advance()
        self.tracker.reset()
        return self.encoder.encode(self.view), {"market_seed": market_seed}

    def step(self, action: int):
        """Send the decoded action, run to the next decision and return the exit-value reward."""
        assert self.market is not None and self.tracker is not None, "call reset() first"
        action = int(action)
        self.action_counts[action] += 1
        view = self.view
        mask = self.actions.mask(view)
        orders = self.actions.decode(action, view) if mask[action] else []
        self.market.act(LEARNER_ID, orders)
        deciding = self._advance()
        reward, info = self.tracker.step()
        account = self.market.accounts[LEARNER_ID]
        terminated = not account.active
        truncated = not deciding and not terminated
        info.update({"action": ACTION_NAMES[action], "valid": bool(mask[action]), "time": self.market.time})
        return self.encoder.encode(self.view), float(reward), terminated, truncated, info


def make_env(config: Optional[EnvConfig] = None):
    """Factory for vectorized environments: ``make_env(cfg)`` returns a zero-argument constructor."""

    def _init():
        return PerpTradingEnv(config)

    return _init
