from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised through the fallback surface in tests
    gym = None
    spaces = None

from marl_trading.agents.base import MarketObservation, OrderIntent, ScriptedAgent
from marl_trading.analysis import summarize_event_log
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.core.config import SimulationConfig
from marl_trading.market.processes import NewsEvent
from marl_trading.market.simulator import MarketRunResult, SyntheticMarketSimulator

from .boundary import (
    PHASE_A_ACTION_TYPE_ORDER,
    RLAction,
    RLActionType,
    action_to_order_intent,
    build_action_mask,
    compute_reward,
    feature_vector,
    mask_invalid_action,
)

_FEATURE_DIMENSION = 18
_ACTION_TYPE_ORDER = tuple(RLActionType)


class _FallbackEnvBase:
    metadata: dict[str, object] = {}

    def reset(self, *args, **kwargs):  # pragma: no cover - interface placeholder only
        raise NotImplementedError

    def step(self, *args, **kwargs):  # pragma: no cover - interface placeholder only
        raise NotImplementedError

    def close(self) -> None:
        return None


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class _FallbackMultiDiscrete:
    def __init__(self, nvec: list[int] | tuple[int, ...] | np.ndarray) -> None:
        self.nvec = np.asarray(nvec, dtype=np.int64)
        self.shape = tuple(self.nvec.shape)


class _FallbackDiscrete:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.shape = ()


_GymEnvBase = gym.Env if gym is not None else _FallbackEnvBase
_BoxSpace = spaces.Box if spaces is not None else _FallbackBox
_MultiDiscreteSpace = spaces.MultiDiscrete if spaces is not None else _FallbackMultiDiscrete
_DiscreteSpace = spaces.Discrete if spaces is not None else _FallbackDiscrete


@dataclass(frozen=True)
class SingleAgentEnvConfig:
    learning_agent_id: str
    learning_agent_starting_inventory: float = 0.0
    phase_a_action_space: bool = True
    include_cancel_action: bool = False
    fixed_order_quantity: int = 1
    fixed_price_offset_ticks: int = 1
    reward_realized_pnl_delta_coefficient: float = 0.0
    reward_inventory_penalty: float = 0.0
    reward_inventory_risk_penalty: float = 0.0
    reward_equity_delta_coefficient: float = 1.0
    reward_inactivity_penalty: float = 0.0
    terminate_on_ruin: bool = True
    auto_increment_seed_on_reset: bool = False
    seed_stride: int = 1


@dataclass
class _PnlTracker:
    inventory: float
    cost_basis: float
    realized_pnl: float = 0.0


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
        self._reset_count = 0
        self._last_seed = int(self.config.seed)
        self._processed_event_count = 0
        self._pnl_trackers: dict[str, _PnlTracker] = {}
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
        portfolio = simulator.portfolios.get(self.env_config.learning_agent_id)
        overridden_inventory = float(self.env_config.learning_agent_starting_inventory)
        portfolio.starting_inventory = overridden_inventory
        portfolio.inventory = overridden_inventory
        portfolio.reserved_inventory = 0.0

    def _bootstrap_pnl_trackers(self) -> None:
        if self.simulator is None:
            return
        self._processed_event_count = 0
        self._pnl_trackers = {}
        for agent_id, portfolio in self.simulator.portfolios.portfolios.items():
            starting_inventory = float(portfolio.inventory)
            self._pnl_trackers[agent_id] = _PnlTracker(
                inventory=starting_inventory,
                cost_basis=starting_inventory * float(self.simulator.start_midpoint),
            )

    def _ingest_new_trade_events(self) -> int:
        if self.simulator is None:
            return 0
        events = self.simulator.event_log.events
        if self._processed_event_count >= len(events):
            return 0
        learning_agent_trade_count = 0
        for event in events[self._processed_event_count :]:
            self._processed_event_count += 1
            event_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            if event_type != "trade":
                continue
            payload = dict(event.payload)
            buy_agent_id = payload.get("buy_agent_id")
            sell_agent_id = payload.get("sell_agent_id")
            if buy_agent_id is None or sell_agent_id is None or event.price is None or event.quantity is None:
                continue
            price = float(event.price)
            quantity = float(event.quantity)

            buyer = self._pnl_trackers.setdefault(str(buy_agent_id), _PnlTracker(inventory=0.0, cost_basis=0.0))
            seller = self._pnl_trackers.setdefault(str(sell_agent_id), _PnlTracker(inventory=0.0, cost_basis=0.0))
            learning_agent_id = self.env_config.learning_agent_id
            if str(buy_agent_id) == learning_agent_id or str(sell_agent_id) == learning_agent_id:
                learning_agent_trade_count += 1

            buyer.inventory += quantity
            buyer.cost_basis += quantity * price

            average_cost = seller.cost_basis / seller.inventory if seller.inventory > 1e-12 else price
            seller.realized_pnl += quantity * (price - average_cost)
            seller.cost_basis = max(seller.cost_basis - average_cost * quantity, 0.0)
            seller.inventory = max(seller.inventory - quantity, 0.0)
        return learning_agent_trade_count

    def _current_realized_pnl(self) -> float:
        tracker = self._pnl_trackers.get(self.env_config.learning_agent_id)
        if tracker is None:
            return 0.0
        return float(tracker.realized_pnl)

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
        runtime_seed = seed
        if runtime_seed is None and self.env_config.auto_increment_seed_on_reset:
            runtime_seed = int(self.config.seed + self._reset_count * self.env_config.seed_stride)
        if runtime_seed is None:
            runtime_seed = int(self.config.seed)
        self._last_seed = int(runtime_seed)
        self.simulator = self._build_simulator(seed=self._last_seed, horizon=self.horizon)
        self._done = False
        self._last_reward = 0.0
        self._last_info = {}
        self._reset_count += 1
        self._bootstrap_pnl_trackers()
        observation = self._advance_until_learning_turn()
        return feature_vector(observation)

    def reset_info(self) -> dict[str, Any]:
        observation = self.get_observation()
        return {
            "seed": int(self._last_seed),
            "horizon": int(self.horizon),
            "learning_agent_id": self.env_config.learning_agent_id,
            "step_index": int(self.simulator.current_step_index) if self.simulator is not None else 0,
            "portfolio_active": bool(observation.portfolio_active),
        }

    def _done_reason(self, done: bool, observation: MarketObservation) -> str | None:
        if not done:
            return None
        if self.simulator is not None and self.simulator.current_step_index >= self.horizon:
            return "horizon"
        if not observation.portfolio_active:
            return "ruin"
        return "terminated"

    def step(self, action: RLAction) -> tuple[tuple[float, ...], float, bool, dict[str, Any]]:
        if self.simulator is None:
            raise RuntimeError("Environment is not initialized.")
        if self._done:
            observation = self.get_observation()
            return feature_vector(observation), 0.0, True, dict(self._last_info)

        previous_observation = self.get_observation()
        previous_equity = float(previous_observation.agent_equity)
        previous_realized_pnl = self._current_realized_pnl()
        effective_action, invalid_action_reason = mask_invalid_action(action, previous_observation)
        intent_submitted = effective_action.action_type not in {RLActionType.HOLD, RLActionType.CANCEL_OLDEST}

        cancelled_order_id: str | None = None
        if effective_action.action_type is RLActionType.CANCEL_OLDEST:
            open_orders = list(self.simulator.open_orders.get(self.env_config.learning_agent_id, []))
            if open_orders:
                cancelled_order_id = str(open_orders[0])
                self.simulator._cancel_oldest_order(self.env_config.learning_agent_id, self.simulator.current_step_index)  # noqa: SLF001

        self._proxy.set_action(effective_action)
        self.simulator.step()
        learning_agent_trade_count = self._ingest_new_trade_events()

        if self._learning_agent_inactive():
            self._done = True

        while not self._done and not self._is_learning_turn():
            self._proxy.set_action(RLAction(RLActionType.HOLD))
            self.simulator.step()
            learning_agent_trade_count += self._ingest_new_trade_events()
            if self._learning_agent_inactive():
                self._done = True
                break
            if self.simulator.current_step_index >= self.horizon:
                self._done = True
                break

        observation = self._build_observation()
        current_realized_pnl = self._current_realized_pnl()
        reward_breakdown = compute_reward(
            previous_equity=previous_equity,
            current_equity=float(observation.agent_equity),
            current_inventory=float(observation.agent_inventory),
            previous_realized_pnl=previous_realized_pnl,
            current_realized_pnl=current_realized_pnl,
            realized_pnl_delta_coefficient=float(self.env_config.reward_realized_pnl_delta_coefficient),
            equity_delta_coefficient=float(self.env_config.reward_equity_delta_coefficient),
            inactivity_penalty_applied=learning_agent_trade_count == 0,
            inactivity_penalty_coefficient=float(self.env_config.reward_inactivity_penalty),
            absolute_inventory_penalty_coefficient=float(self.env_config.reward_inventory_penalty),
            inventory_risk_penalty_coefficient=float(self.env_config.reward_inventory_risk_penalty),
        )
        done = bool(self._done or self.simulator.current_step_index >= self.horizon)
        if done:
            self._done = True
        done_reason = self._done_reason(done, observation)

        info: dict[str, Any] = {
            "previous_equity": previous_equity,
            "current_equity": float(observation.agent_equity),
            "previous_realized_pnl": previous_realized_pnl,
            "current_realized_pnl": current_realized_pnl,
            "realized_pnl_delta": reward_breakdown.realized_pnl_delta,
            "current_inventory": float(observation.agent_inventory),
            "reward_breakdown": reward_breakdown,
            "requested_action": action.action_type.value,
            "applied_action": effective_action.action_type.value,
            "cancelled_order_id": cancelled_order_id,
            "intent_submitted": bool(intent_submitted),
            "invalid_action_reason": invalid_action_reason,
            "action_masked": bool(invalid_action_reason is not None),
            "learning_agent_trade_count": int(learning_agent_trade_count),
            "learning_agent_had_trade": bool(learning_agent_trade_count > 0),
            "inactivity_penalty_applied": bool(learning_agent_trade_count == 0),
            "step_index": int(self.simulator.current_step_index),
            "portfolio_active": bool(observation.portfolio_active),
            "done_reason": done_reason,
            "seed": int(self._last_seed),
            "horizon": int(self.horizon),
        }
        self._current_observation = observation
        self._last_reward = reward_breakdown.total_reward
        self._last_info = info
        return feature_vector(observation), reward_breakdown.total_reward, done, info

    def build_run_result(self) -> MarketRunResult:
        if self.simulator is None:
            raise RuntimeError("Environment is not initialized.")
        summary = summarize_event_log(self.simulator.event_log)
        summary.update(
            {
                "horizon": self.horizon,
                "final_fundamental": self.simulator.fundamental.current_value,
                "active_agent_count": len(self.simulator.portfolios.active_portfolios()),
                "final_midpoint": summary.get("final_midpoint") or self.simulator.fundamental.current_value,
            }
        )
        final_portfolios = {
            agent_id: portfolio.summary(self.simulator.fundamental.current_value)
            for agent_id, portfolio in self.simulator.portfolios.portfolios.items()
        }
        return MarketRunResult(
            event_log=self.simulator.event_log,
            step_records=list(self.simulator.step_records),
            summary=summary,
            final_portfolios=final_portfolios,
            final_fundamental=self.simulator.fundamental.current_value,
        )


class GymSingleAgentMarketEnv(_GymEnvBase):
    metadata = {"render_modes": []}

    def __init__(
        self,
        core_env: SingleAgentMarketEnv,
        *,
        max_quantity: int = 3,
        max_price_offset_ticks: int = 3,
    ) -> None:
        self.core_env = core_env
        self.max_quantity = max(int(max_quantity), 1)
        self.max_price_offset_ticks = max(int(max_price_offset_ticks), 1)
        env_config = self.core_env.env_config
        self.phase_a_action_space = bool(env_config.phase_a_action_space)
        self.fixed_order_quantity = max(int(env_config.fixed_order_quantity), 1)
        self.fixed_price_offset_ticks = max(int(env_config.fixed_price_offset_ticks), 1)
        self.action_types = self._resolve_action_types(include_cancel=bool(env_config.include_cancel_action))
        if self.phase_a_action_space:
            self.action_space = _DiscreteSpace(len(self.action_types))
        else:
            self.action_space = _MultiDiscreteSpace(
                [
                    len(_ACTION_TYPE_ORDER),
                    self.max_quantity,
                    self.max_price_offset_ticks,
                ]
            )
        self.observation_space = _BoxSpace(
            low=-np.inf,
            high=np.inf,
            shape=(_FEATURE_DIMENSION,),
            dtype=np.float32,
        )

    def _resolve_action_types(self, *, include_cancel: bool) -> tuple[RLActionType, ...]:
        if self.phase_a_action_space:
            if include_cancel:
                return PHASE_A_ACTION_TYPE_ORDER + (RLActionType.CANCEL_OLDEST,)
            return PHASE_A_ACTION_TYPE_ORDER
        return _ACTION_TYPE_ORDER

    def _coerce_action(self, action: RLAction | np.ndarray | list[int] | tuple[int, ...]) -> RLAction:
        if isinstance(action, RLAction):
            return action
        if self.phase_a_action_space:
            action_index = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
            if action_index < 0 or action_index >= len(self.action_types):
                raise ValueError(f"Unsupported action index: {action_index}")
            return RLAction(
                action_type=self.action_types[action_index],
                quantity=self.fixed_order_quantity,
                price_offset_ticks=self.fixed_price_offset_ticks,
            )
        raw = np.asarray(action, dtype=np.int64).reshape(-1)
        if raw.size != 3:
            raise ValueError("Gym action must contain exactly 3 integers: type, quantity, price offset.")
        action_index = int(raw[0])
        if action_index < 0 or action_index >= len(_ACTION_TYPE_ORDER):
            raise ValueError(f"Unsupported action index: {action_index}")
        quantity = int(raw[1]) + 1
        price_offset_ticks = int(raw[2]) + 1
        return RLAction(
            action_type=_ACTION_TYPE_ORDER[action_index],
            quantity=quantity,
            price_offset_ticks=price_offset_ticks,
        )

    def action_masks(self) -> np.ndarray:
        observation = self.core_env.get_observation()
        if self.phase_a_action_space:
            mask = build_action_mask(
                observation,
                action_types=self.action_types,
                quantity=self.fixed_order_quantity,
                price_offset_ticks=self.fixed_price_offset_ticks,
            )
        else:
            mask = tuple(True for _ in range(len(_ACTION_TYPE_ORDER)))
        return np.asarray(mask, dtype=bool)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        horizon = None if not options else options.get("horizon")
        observation = self.core_env.reset(seed=seed, horizon=horizon)
        info = self.core_env.reset_info()
        info["action_mask"] = self.action_masks().astype(bool).tolist()
        info["action_types"] = [action_type.value for action_type in self.action_types]
        info["phase_a_action_space"] = bool(self.phase_a_action_space)
        return np.asarray(observation, dtype=np.float32), info

    def step(
        self,
        action: RLAction | np.ndarray | list[int] | tuple[int, ...],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        decoded_action = self._coerce_action(action)
        observation, reward, done, info = self.core_env.step(decoded_action)
        terminated = bool(done and info.get("done_reason") != "horizon")
        truncated = bool(done and info.get("done_reason") == "horizon")
        enriched_info = dict(info)
        enriched_info["rl_action"] = decoded_action.action_type.value
        enriched_info["action_mask"] = self.action_masks().astype(bool).tolist()
        enriched_info["action_types"] = [action_type.value for action_type in self.action_types]
        enriched_info["phase_a_action_space"] = bool(self.phase_a_action_space)
        return np.asarray(observation, dtype=np.float32), float(reward), terminated, truncated, enriched_info

    def close(self) -> None:
        return None

    @property
    def learning_agent_id(self) -> str:
        return self.core_env.env_config.learning_agent_id

    def build_run_result(self) -> MarketRunResult:
        return self.core_env.build_run_result()


__all__ = [
    "GymSingleAgentMarketEnv",
    "SingleAgentEnvConfig",
    "SingleAgentMarketEnv",
]
