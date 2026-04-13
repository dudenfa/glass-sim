from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from marl_trading.agents.base import MarketObservation, OrderIntent, ScriptedAgent
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.core.config import SimulationConfig
from marl_trading.market.processes import NewsEvent
from marl_trading.market.simulator import SyntheticMarketSimulator

from .boundary import RLAction, RLActionType, action_to_order_intent, compute_reward, feature_vector


@dataclass(frozen=True)
class SingleAgentEnvConfig:
    learning_agent_id: str
    reward_inventory_penalty: float = 0.0
    terminate_on_ruin: bool = True


class _LearningAgentProxy(ScriptedAgent):
    def __init__(self, agent_id: str, max_resting_orders: int = 1) -> None:
        super().__init__(agent_id=agent_id, agent_type="rl_agent", max_resting_orders=max_resting_orders)
        self._pending_action = RLAction(RLActionType.HOLD)

    def set_action(self, action: RLAction) -> None:
        self._pending_action = action

    def clear_action(self) -> None:
        self._pending_action = RLAction(RLActionType.HOLD)

    def decide(self, observation: MarketObservation, rng) -> OrderIntent | None:  # noqa: ARG002
        intent = action_to_order_intent(self._pending_action, observation)
        self.clear_action()
        return intent


class SingleAgentMarketEnv:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        env_config: SingleAgentEnvConfig | None = None,
        horizon: int | None = None,
    ) -> None:
        self.config = config or default_simulation_config()
        self.env_config = env_config or SingleAgentEnvConfig(learning_agent_id=self._default_learning_agent_id())
        self.horizon = int(horizon if horizon is not None else self.config.market.event_horizon)
        self._proxy = _LearningAgentProxy(self.env_config.learning_agent_id)
        self.simulator: SyntheticMarketSimulator | None = None
        self._done = False
        self._current_observation: MarketObservation | None = None
        self._last_reward: float = 0.0
        self._last_info: dict[str, Any] = {}

        self.reset()

    def _default_learning_agent_id(self) -> str:
        if not self.config.agents:
            raise ValueError("Simulation config must define at least one agent.")
        return str(self.config.agents[-1].agent_id)

    def _build_simulator(self, *, seed: int | None = None, horizon: int | None = None) -> SyntheticMarketSimulator:
        updated = self.config
        if seed is not None:
            updated = replace(updated, seed=int(seed))
        simulator = SyntheticMarketSimulator(updated, horizon=horizon if horizon is not None else self.horizon)
        self._attach_proxy(simulator)
        return simulator

    def _attach_proxy(self, simulator: SyntheticMarketSimulator) -> None:
        if self.env_config.learning_agent_id not in simulator.agents:
            raise KeyError(f"Unknown learning agent id: {self.env_config.learning_agent_id}")
        original = simulator.agents[self.env_config.learning_agent_id]
        self._proxy = _LearningAgentProxy(
            self.env_config.learning_agent_id,
            max_resting_orders=getattr(original, "max_resting_orders", 1),
        )
        simulator.agents[self.env_config.learning_agent_id] = self._proxy

    def _active_agent_ids(self) -> list[str]:
        if self.simulator is None:
            return []
        return [agent_id for agent_id, portfolio in self.simulator.portfolios.portfolios.items() if portfolio.active]

    def _is_learning_turn(self) -> bool:
        if self.simulator is None:
            return False
        active_ids = self._active_agent_ids()
        if not active_ids:
            return False
        if self.env_config.learning_agent_id not in active_ids:
            return False
        next_index = self.simulator.current_step_index % len(active_ids)
        return active_ids[next_index] == self.env_config.learning_agent_id

    def _build_observation(self) -> MarketObservation:
        if self.simulator is None:
            raise RuntimeError("Environment is not initialized.")
        step_index = int(self.simulator.current_step_index)
        timestamp_ns = step_index
        snapshot = self.simulator._current_book_snapshot(timestamp_ns)  # noqa: SLF001
        portfolio = self.simulator.portfolios.get(self.env_config.learning_agent_id)
        news_headline, news_severity = self.simulator.recent_news
        news = None if news_headline is None else NewsEvent(headline=str(news_headline), severity=float(news_severity or 0.0), impact=0.0)
        return self.simulator._make_observation(  # noqa: SLF001
            agent_id=self.env_config.learning_agent_id,
            step_index=step_index,
            timestamp_ns=timestamp_ns,
            news=news,
            portfolio=portfolio,
            snapshot=snapshot,
        )

    def _advance_until_learning_turn(self) -> MarketObservation:
        if self.simulator is None:
            raise RuntimeError("Environment is not initialized.")
        while not self._done and not self._is_learning_turn():
            self._proxy.set_action(RLAction(RLActionType.HOLD))
            self.simulator.step()
            if self._learning_agent_inactive():
                self._done = True
                break
        observation = self._build_observation()
        self._current_observation = observation
        return observation

    def _learning_agent_inactive(self) -> bool:
        if self.simulator is None:
            return True
        portfolio = self.simulator.portfolios.get(self.env_config.learning_agent_id)
        return not portfolio.active

    def get_observation(self) -> MarketObservation:
        if self._current_observation is None:
            return self._advance_until_learning_turn()
        return self._current_observation

    def reset(self, *, seed: int | None = None, horizon: int | None = None) -> tuple[float, ...]:
        self.horizon = int(self.horizon if horizon is None else horizon)
        self.simulator = self._build_simulator(seed=seed, horizon=self.horizon)
        self._done = False
        self._last_reward = 0.0
        self._last_info = {}
        observation = self._advance_until_learning_turn()
        return feature_vector(observation)

    def step(self, action: RLAction) -> tuple[tuple[float, ...], float, bool, dict[str, Any]]:
        if self.simulator is None:
            raise RuntimeError("Environment is not initialized.")
        if self._done:
            observation = self.get_observation()
            return feature_vector(observation), 0.0, True, dict(self._last_info)

        previous_observation = self.get_observation()
        previous_equity = float(previous_observation.agent_equity)

        cancelled_order_id: str | None = None
        if action.action_type is RLActionType.CANCEL_OLDEST:
            open_orders = list(self.simulator.open_orders.get(self.env_config.learning_agent_id, []))
            if open_orders:
                cancelled_order_id = str(open_orders[0])
                self.simulator._cancel_oldest_order(self.env_config.learning_agent_id, self.simulator.current_step_index)  # noqa: SLF001

        self._proxy.set_action(action)
        self.simulator.step()

        if self._learning_agent_inactive():
            self._done = True

        while not self._done and not self._is_learning_turn():
            self._proxy.set_action(RLAction(RLActionType.HOLD))
            self.simulator.step()
            if self._learning_agent_inactive():
                self._done = True
                break
            if self.simulator.current_step_index >= self.horizon:
                self._done = True
                break

        observation = self._build_observation()
        reward_breakdown = compute_reward(
            previous_equity=previous_equity,
            current_equity=float(observation.agent_equity),
            current_inventory=float(observation.agent_inventory),
            inventory_penalty_coefficient=float(self.env_config.reward_inventory_penalty),
        )
        done = bool(self._done or self.simulator.current_step_index >= self.horizon)
        if done:
            self._done = True

        info: dict[str, Any] = {
            "previous_equity": previous_equity,
            "current_equity": float(observation.agent_equity),
            "current_inventory": float(observation.agent_inventory),
            "reward_breakdown": reward_breakdown,
            "applied_action": action.action_type.value,
            "cancelled_order_id": cancelled_order_id,
            "step_index": int(self.simulator.current_step_index),
            "portfolio_active": bool(observation.portfolio_active),
        }
        self._current_observation = observation
        self._last_reward = reward_breakdown.total_reward
        self._last_info = info
        return feature_vector(observation), reward_breakdown.total_reward, done, info


__all__ = [
    "SingleAgentEnvConfig",
    "SingleAgentMarketEnv",
]
