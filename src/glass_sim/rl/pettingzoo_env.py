"""PettingZoo interface: several learning agents in one market, for any MARL library.

GLASS learners decide asynchronously: each wakes on its own random clock and
the market runs between decisions. PettingZoo's agent-environment-cycle (AEC)
API describes exactly this: ``agent_selection`` is the learner whose decision
is pending, and ``step(action)`` sends its orders (with latency) and runs the
market until the next learner must decide. The order of turns therefore
follows the market, not a fixed rotation. There is no parallel-API version
because two learners never decide at the same instant.

Rewards are the same as in the single-agent environment (change of exit
value plus penalties). A learner's reward for a decision is known only when it
decides again, is liquidated or the episode ends; it is credited then, so
``last()`` at a learner's turn returns the reward earned since its previous
action.

A liquidated learner is terminated; at the end of the episode duration the
remaining learners are truncated. As usual in AEC, a terminated or truncated
agent is stepped once with ``None`` to remove it.

By default the observation is a dict ``{"observation": features,
"action_mask": int8 array}``, the convention of PettingZoo environments with
invalid actions; with ``mask_in_observation=False`` it is the feature vector
alone. The current mask is also in ``infos[agent]["action_mask"]``.

    from glass_sim.rl.marl import MultiAgentConfig
    from glass_sim.rl.pettingzoo_env import GlassAECEnv

    env = GlassAECEnv(MultiAgentConfig(scenario="small", n_learners=3, episode_duration=600))
    env.reset(seed=1)
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        action = None if terminated or truncated else env.action_space(agent).sample(obs["action_mask"])
        env.step(action)

Requires ``pip install glass-sim[rl]`` (PettingZoo is part of the ``rl`` extra).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from .components import Actions, Encoder, RewardFactory
from .marl import MultiAgentConfig, MultiAgentMarket


class GlassAECEnv(AECEnv):
    """Several learners in one perpetual-futures market (PettingZoo AEC API); see the module docstring."""

    metadata = {"name": "glass_perp_v0", "render_modes": [], "is_parallelizable": False}

    def __init__(
        self,
        config: Optional[MultiAgentConfig] = None,
        *,
        mask_in_observation: bool = True,
        encoder: Optional[Encoder] = None,
        actions: Optional[Actions] = None,
        reward: Optional[RewardFactory] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """``encoder``, ``actions`` and ``reward`` replace the parts built from ``config`` (see ``components.py``)."""
        super().__init__()
        if render_mode is not None:
            raise ValueError("GlassAECEnv has no render modes; use the live viewer (glass serve) to watch a market")
        self.config = config or MultiAgentConfig()
        self.core = MultiAgentMarket(self.config, encoder=encoder, actions=actions, reward=reward)
        self.mask_in_observation = mask_in_observation
        self.render_mode = None
        self.possible_agents = list(self.core.ids)
        n = self.core.actions.n
        low, high = getattr(self.core.encoder, "bounds", (-5.0, 5.0))
        features = spaces.Box(low=low, high=high, shape=(self.core.encoder.size,), dtype=np.float32)
        if mask_in_observation:
            obs_space = spaces.Dict({"observation": features, "action_mask": spaces.Box(0, 1, shape=(n,), dtype=np.int8)})
        else:
            obs_space = features
        self._observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self._action_spaces = {agent: spaces.Discrete(n) for agent in self.possible_agents}
        self._episode = 0
        self._features: dict[str, np.ndarray] = {}
        self._pending: Optional[str] = None
        self._pending_mask: Optional[np.ndarray] = None
        self.market_seed: Optional[int] = None

    # ------------------------------------------------------------- spaces

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    @property
    def market(self):
        """The underlying ``PerpetualMarket`` of the current episode (for analysis)."""
        return self.core.market

    @property
    def action_names(self) -> list[str]:
        return list(self.core.actions.names)

    # ------------------------------------------------------------ helpers

    def _hold_only(self) -> np.ndarray:
        mask = np.zeros(self.core.actions.n, dtype=np.int8)
        mask[0] = 1
        return mask

    def _apply(self, request: dict[str, Any]) -> None:
        """Credit rewards of finished transitions, mark ended agents and select the next agent."""
        for agent, reward, terminal, info in request["closed"]:
            if agent not in self.rewards:
                continue
            self.rewards[agent] += reward
            self.infos[agent] = dict(info)
            if terminal:
                self.terminations[agent] = True
        for agent in self.agents:
            if not self.terminations[agent] and not self.market.accounts[agent].active:
                self.terminations[agent] = True
        if request["done"]:
            self._pending = None
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.truncations[agent] = True
                self.infos[agent]["final_pnl_pct"] = request["final"][agent]
            self.agent_selection = self.agents[0]
        else:
            agent = request["agent"]
            self._pending = agent
            self._pending_mask = np.asarray(request["mask"], dtype=np.int8)
            self._features[agent] = np.asarray(request["obs"], dtype=np.float32)
            self.infos[agent]["action_mask"] = self._pending_mask
            self.agent_selection = agent
        self._accumulate_rewards()
        self._deads_step_first()

    # ---------------------------------------------------------- AEC API

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> None:
        """Start an episode. The market seed is ``options["market_seed"]``, else ``seed``, else the next training seed."""
        options = options or {}
        if "market_seed" in options:
            market_seed = int(options["market_seed"])
        elif seed is not None:
            market_seed = int(seed)
        else:
            seeds = self.config.train_seeds
            market_seed = int(seeds[self._episode % len(seeds)])
        self._episode += 1
        self.market_seed = market_seed
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._features = {}
        self._pending = None
        self._skip_agent_selection = None
        self._apply(self.core.reset(market_seed))

    def observe(self, agent: str):
        """Latest features of ``agent`` (from its last decision) and its mask: the real one at its turn, else only ``hold``."""
        features = self._features.get(agent)
        if features is None:
            features = np.zeros(self.core.encoder.size, dtype=np.float32)
        if not self.mask_in_observation:
            return features
        mask = self._pending_mask if agent == self._pending else self._hold_only()
        return {"observation": features, "action_mask": mask}

    def step(self, action: Optional[int]) -> None:
        """Send the selected agent's action and run the market to the next decision of any learner."""
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0.0
        self.rewards = {other: 0.0 for other in self.agents}
        self.infos[agent].pop("action_mask", None)
        self._apply(self.core.step(agent, int(action)))

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def env(config: Optional[MultiAgentConfig] = None, **kwargs: Any) -> GlassAECEnv:
    """PettingZoo-style constructor: ``glass_sim.rl.pettingzoo_env.env(config)``."""
    return GlassAECEnv(config, **kwargs)
