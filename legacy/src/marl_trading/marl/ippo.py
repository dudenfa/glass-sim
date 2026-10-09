from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from marl_trading.agents.base import MarketObservation, OrderIntent, ScriptedAgent
from marl_trading.configs import build_preset_config
from marl_trading.core.config import SimulationConfig
from marl_trading.market.processes import NewsEvent
from marl_trading.market.simulator import SyntheticMarketSimulator
from marl_trading.rl.boundary import (
    PHASE_A_ACTION_TYPE_ORDER,
    RLAction,
    RLActionType,
    action_to_order_intent,
    build_action_mask,
    compute_reward,
    feature_vector,
)
from marl_trading.rl.live import PPOPolicyAdapter, RuntimePolicyControlledAgent
from marl_trading.rl.scenario import (
    FrozenAgentSpec,
    exclude_agents_from_config,
    prepare_frozen_agent_configs,
    prepare_learning_agent_config,
)


@dataclass(frozen=True)
class IPPOTrainingConfig:
    learner_ids: tuple[str, str, str]
    learner_starting_inventory: float = 16.0
    frozen_agents: tuple[FrozenAgentSpec, ...] = ()
    horizon: int = 10_000
    train_seeds: tuple[int, ...] = ()
    total_timesteps: int = 100_000
    n_steps: int = 512
    batch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_equity_delta_coefficient: float = 0.1
    reward_inactivity_penalty: float = 0.01
    reward_inventory_penalty: float = 0.002
    reward_inventory_risk_penalty: float = 0.00005
    device: str = "auto"
    seed: int = 1

    def __post_init__(self) -> None:
        normalized_ids = tuple(str(agent_id).strip() for agent_id in self.learner_ids)
        if len(normalized_ids) != 3 or not all(normalized_ids):
            raise ValueError("IPPO requires exactly three non-empty learner ids.")
        if len(set(normalized_ids)) != 3:
            raise ValueError("IPPO learner ids must be unique.")
        if self.total_timesteps <= 0 or self.n_steps <= 0:
            raise ValueError("total_timesteps and n_steps must be positive.")
        if self.batch_size <= 0 or self.batch_size > self.n_steps:
            raise ValueError("batch_size must be positive and no greater than n_steps.")
        object.__setattr__(self, "learner_ids", normalized_ids)
        object.__setattr__(self, "train_seeds", tuple(int(seed) for seed in self.train_seeds))
        object.__setattr__(self, "frozen_agents", tuple(self.frozen_agents))


@dataclass
class _PnlTracker:
    inventory: float
    cost_basis: float
    realized_pnl: float = 0.0


@dataclass(frozen=True)
class _PendingTransition:
    observation: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    value: Any
    log_prob: Any
    previous_equity: float
    previous_realized_pnl: float
    trade_count: int
    episode_start: bool


class _MARLAgentProxy(ScriptedAgent):
    def __init__(self, agent_id: str, max_resting_orders: int) -> None:
        super().__init__(agent_id=agent_id, agent_type="marl_agent", max_resting_orders=max_resting_orders)
        self._pending_action = RLAction(RLActionType.HOLD)

    def set_action(self, action: RLAction) -> None:
        self._pending_action = action

    def decide(self, observation: MarketObservation, rng) -> OrderIntent | None:  # noqa: ARG002
        action = self._pending_action
        self._pending_action = RLAction(RLActionType.HOLD)
        return action_to_order_intent(action, observation)


def build_joint_training_config(
    *,
    preset_name: str,
    learner_ids: Sequence[str],
    frozen_agents: Sequence[FrozenAgentSpec] = (),
    seed: int = 1,
    excluded_agent_ids: Sequence[str] = (),
    preserve_excluded_turn_slots: bool = False,
) -> SimulationConfig:
    config = replace(build_preset_config(preset_name), seed=int(seed))
    config = exclude_agents_from_config(
        config,
        excluded_agent_ids=tuple(excluded_agent_ids),
        preserve_excluded_turn_slots=bool(preserve_excluded_turn_slots),
    )
    if frozen_agents:
        config = prepare_frozen_agent_configs(config, frozen_agents=tuple(frozen_agents))
    for learner_id in learner_ids:
        config = prepare_learning_agent_config(
            config,
            learning_agent_id=str(learner_id),
            add_learning_agent=True,
            learning_agent_template_id="trend_01",
        )
    return config


class JointIPPORunner:
    """Collect shared-market rollouts and update three MaskablePPO policies together."""

    def __init__(self, simulation_config: SimulationConfig, training_config: IPPOTrainingConfig) -> None:
        self.simulation_config = simulation_config
        self.training_config = training_config
        self._seed_index = 0
        self._processed_event_count = 0
        self._pnl_trackers: dict[str, _PnlTracker] = {}
        self._trade_counts: dict[str, int] = {}
        self._frozen_policies = self._load_frozen_policies()
        self.models = self._build_models()
        self.simulator: SyntheticMarketSimulator | None = None

    def _build_models(self) -> dict[str, Any]:
        try:
            import gymnasium as gym
            from gymnasium import spaces
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.logger import configure
        except ImportError as exc:  # pragma: no cover - depends on optional install state
            raise RuntimeError(
                "Joint IPPO training requires gymnasium, stable-baselines3, and sb3-contrib."
            ) from exc

        class _PolicySpaceEnv(gym.Env):
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
            action_space = spaces.Discrete(len(PHASE_A_ACTION_TYPE_ORDER))

            def reset(self, *, seed=None, options=None):  # noqa: ANN001
                super().reset(seed=seed)
                return np.zeros(18, dtype=np.float32), {}

            def step(self, action):  # noqa: ANN001
                return np.zeros(18, dtype=np.float32), 0.0, True, False, {}

        models: dict[str, Any] = {}
        for index, learner_id in enumerate(self.training_config.learner_ids):
            model = MaskablePPO(
                "MlpPolicy",
                _PolicySpaceEnv(),
                verbose=0,
                n_steps=self.training_config.n_steps,
                batch_size=self.training_config.batch_size,
                learning_rate=self.training_config.learning_rate,
                gamma=self.training_config.gamma,
                gae_lambda=self.training_config.gae_lambda,
                device=self.training_config.device,
                seed=self.training_config.seed + index,
            )
            model.set_logger(configure(folder=None, format_strings=[]))
            models[learner_id] = model
        return models

    def _load_frozen_policies(self) -> dict[str, PPOPolicyAdapter]:
        policies: dict[str, PPOPolicyAdapter] = {}
        for spec in self.training_config.frozen_agents:
            adapter, status = PPOPolicyAdapter.try_load(spec.checkpoint_path, device="cpu", deterministic=True)
            if adapter is None:
                raise RuntimeError(status.reason or f"Unable to load frozen PPO checkpoint: {spec.checkpoint_path}")
            policies[spec.agent_id] = adapter
        return policies

    def _next_seed(self) -> int:
        if self.training_config.train_seeds:
            seed = self.training_config.train_seeds[self._seed_index % len(self.training_config.train_seeds)]
            self._seed_index += 1
            return int(seed)
        seed = int(self.training_config.seed + self._seed_index)
        self._seed_index += 1
        return seed

    def _build_simulator(self) -> SyntheticMarketSimulator:
        simulator = SyntheticMarketSimulator(
            replace(self.simulation_config, seed=self._next_seed()),
            horizon=self.training_config.horizon,
        )
        for spec in self.training_config.frozen_agents:
            original = simulator.agents[spec.agent_id]
            simulator.agents[spec.agent_id] = RuntimePolicyControlledAgent(
                agent_id=spec.agent_id,
                policy=self._frozen_policies[spec.agent_id],
                fallback_agent=original,
                agent_type="rl_agent",
                max_resting_orders=original.max_resting_orders,
                delegate_bootstrap=False,
            )
            if spec.starting_inventory is not None:
                portfolio = simulator.portfolios.get(spec.agent_id)
                portfolio.starting_inventory = float(spec.starting_inventory)
                portfolio.inventory = float(spec.starting_inventory)
                portfolio.reserved_inventory = 0.0
        for learner_id in self.training_config.learner_ids:
            original = simulator.agents[learner_id]
            simulator.agents[learner_id] = _MARLAgentProxy(learner_id, original.max_resting_orders)
            portfolio = simulator.portfolios.get(learner_id)
            portfolio.starting_inventory = float(self.training_config.learner_starting_inventory)
            portfolio.inventory = float(self.training_config.learner_starting_inventory)
            portfolio.reserved_inventory = 0.0
        self._processed_event_count = 0
        self._pnl_trackers = {
            agent_id: _PnlTracker(
                inventory=float(portfolio.inventory),
                cost_basis=float(portfolio.inventory) * simulator.start_midpoint,
            )
            for agent_id, portfolio in simulator.portfolios.portfolios.items()
        }
        self._trade_counts = {agent_id: 0 for agent_id in simulator.portfolios.portfolios}
        return simulator

    def _ingest_new_trade_events(self) -> None:
        if self.simulator is None:
            return
        events = self.simulator.event_log.events
        for event in events[self._processed_event_count :]:
            self._processed_event_count += 1
            event_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            if event_type != "trade":
                continue
            payload = dict(event.payload)
            buyer_id = payload.get("buy_agent_id")
            seller_id = payload.get("sell_agent_id")
            if buyer_id is None or seller_id is None or event.price is None or event.quantity is None:
                continue
            buyer_id, seller_id = str(buyer_id), str(seller_id)
            price, quantity = float(event.price), float(event.quantity)
            buyer = self._pnl_trackers[buyer_id]
            seller = self._pnl_trackers[seller_id]
            buyer.inventory += quantity
            buyer.cost_basis += quantity * price
            average_cost = seller.cost_basis / seller.inventory if seller.inventory > 1e-12 else price
            seller.realized_pnl += quantity * (price - average_cost)
            seller.cost_basis = max(seller.cost_basis - average_cost * quantity, 0.0)
            seller.inventory = max(seller.inventory - quantity, 0.0)
            self._trade_counts[buyer_id] = self._trade_counts.get(buyer_id, 0) + 1
            self._trade_counts[seller_id] = self._trade_counts.get(seller_id, 0) + 1

    def _observation_for(self, agent_id: str) -> MarketObservation:
        if self.simulator is None:
            raise RuntimeError("Simulator is not initialized.")
        snapshot = self.simulator._current_book_snapshot(self.simulator.current_step_index)  # noqa: SLF001
        headline, severity = self.simulator.recent_news
        news = None if headline is None else NewsEvent(headline=str(headline), severity=float(severity or 0.0), impact=0.0)
        return self.simulator._make_observation(  # noqa: SLF001
            agent_id=agent_id,
            step_index=self.simulator.current_step_index,
            timestamp_ns=self.simulator.current_step_index,
            news=news,
            portfolio=self.simulator.portfolios.get(agent_id),
            snapshot=snapshot,
        )

    def _select_action(self, learner_id: str, observation: MarketObservation) -> tuple[RLAction, _PendingTransition]:
        try:
            import torch as th
            from stable_baselines3.common.utils import obs_as_tensor
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("Joint IPPO training requires stable-baselines3 and torch.") from exc

        model = self.models[learner_id]
        features = np.asarray(feature_vector(observation), dtype=np.float32)
        action_mask = np.asarray(
            build_action_mask(
                observation,
                action_types=PHASE_A_ACTION_TYPE_ORDER,
                quantity=1,
                price_offset_ticks=1,
            ),
            dtype=bool,
        )
        with th.no_grad():
            actions, values, log_prob = model.policy(
                obs_as_tensor(features[None, :], model.device),
                deterministic=False,
                action_masks=action_mask[None, :],
            )
        action_value = int(actions.detach().cpu().numpy().reshape(-1)[0])
        action = RLAction(PHASE_A_ACTION_TYPE_ORDER[action_value])
        pending = _PendingTransition(
            observation=features,
            action=np.asarray([action_value]),
            action_mask=action_mask,
            value=values,
            log_prob=log_prob,
            previous_equity=float(observation.agent_equity),
            previous_realized_pnl=float(self._pnl_trackers[learner_id].realized_pnl),
            trade_count=int(self._trade_counts.get(learner_id, 0)),
            episode_start=False,
        )
        return action, pending

    def _resolve_pending_transition(
        self,
        learner_id: str,
        pending: _PendingTransition,
        observation: MarketObservation,
        *,
        done: bool,
    ) -> None:
        model = self.models[learner_id]
        tracker = self._pnl_trackers[learner_id]
        reward = compute_reward(
            previous_equity=pending.previous_equity,
            current_equity=float(observation.agent_equity),
            current_inventory=float(observation.agent_inventory),
            previous_realized_pnl=pending.previous_realized_pnl,
            current_realized_pnl=float(tracker.realized_pnl),
            realized_pnl_delta_coefficient=1.0,
            equity_delta_coefficient=self.training_config.reward_equity_delta_coefficient,
            inactivity_penalty_applied=self._trade_counts.get(learner_id, 0) == pending.trade_count,
            inactivity_penalty_coefficient=self.training_config.reward_inactivity_penalty,
            absolute_inventory_penalty_coefficient=self.training_config.reward_inventory_penalty,
            inventory_risk_penalty_coefficient=self.training_config.reward_inventory_risk_penalty,
        )
        model.rollout_buffer.add(
            pending.observation[None, :],
            pending.action,
            np.asarray([reward.total_reward], dtype=np.float32),
            np.asarray([pending.episode_start], dtype=bool),
            pending.value,
            pending.log_prob,
            action_masks=pending.action_mask[None, :],
        )

    def _finish_buffers(self, last_observations: Mapping[str, MarketObservation], dones: Mapping[str, bool]) -> None:
        try:
            import torch as th
            from stable_baselines3.common.utils import obs_as_tensor
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("Joint IPPO training requires stable-baselines3 and torch.") from exc
        for learner_id, model in self.models.items():
            observation = last_observations[learner_id]
            features = np.asarray(feature_vector(observation), dtype=np.float32)
            with th.no_grad():
                values = model.policy.predict_values(obs_as_tensor(features[None, :], model.device))
            model.rollout_buffer.compute_returns_and_advantage(
                last_values=values,
                dones=np.asarray([bool(dones.get(learner_id, False))]),
            )

    def collect_rollout(self) -> dict[str, int]:
        for model in self.models.values():
            model.rollout_buffer.reset()

        pending: dict[str, _PendingTransition] = {}
        episode_starts = {learner_id: True for learner_id in self.training_config.learner_ids}
        last_observations: dict[str, MarketObservation] = {}
        final_dones = {learner_id: False for learner_id in self.training_config.learner_ids}
        self.simulator = self._build_simulator()

        while True:
            if self.simulator.current_step_index >= self.training_config.horizon:
                for learner_id, transition in list(pending.items()):
                    observation = self._observation_for(learner_id)
                    self._resolve_pending_transition(learner_id, transition, observation, done=True)
                    last_observations[learner_id] = observation
                    final_dones[learner_id] = True
                pending.clear()
                if all(model.rollout_buffer.full for model in self.models.values()):
                    break
                self.simulator = self._build_simulator()
                episode_starts = {learner_id: True for learner_id in self.training_config.learner_ids}
                continue

            scheduled_agent_id = self.simulator.next_scheduled_agent_id()
            if scheduled_agent_id in self.models:
                learner_id = str(scheduled_agent_id)
                observation = self._observation_for(learner_id)
                if learner_id in pending:
                    self._resolve_pending_transition(learner_id, pending.pop(learner_id), observation, done=False)
                    last_observations[learner_id] = observation
                    final_dones[learner_id] = False
                if all(model.rollout_buffer.full for model in self.models.values()):
                    break
                if not self.models[learner_id].rollout_buffer.full:
                    action, transition = self._select_action(learner_id, observation)
                    pending[learner_id] = replace(transition, episode_start=episode_starts[learner_id])
                    episode_starts[learner_id] = False
                    proxy = self.simulator.agents[learner_id]
                    if not isinstance(proxy, _MARLAgentProxy):
                        raise TypeError(f"Expected MARL proxy for {learner_id}")
                    proxy.set_action(action)

            self.simulator.step()
            self._ingest_new_trade_events()
            any_inactive = any(not self.simulator.portfolios.get(agent_id).active for agent_id in self.models)
            if any_inactive:
                for learner_id, transition in list(pending.items()):
                    observation = self._observation_for(learner_id)
                    self._resolve_pending_transition(learner_id, transition, observation, done=True)
                    last_observations[learner_id] = observation
                    final_dones[learner_id] = True
                pending.clear()
                if all(model.rollout_buffer.full for model in self.models.values()):
                    break
                self.simulator = self._build_simulator()
                episode_starts = {learner_id: True for learner_id in self.training_config.learner_ids}

        missing = [learner_id for learner_id, model in self.models.items() if not model.rollout_buffer.full]
        if missing:
            raise RuntimeError(f"Unable to fill MARL rollout buffers for: {', '.join(missing)}")
        self._finish_buffers(last_observations, final_dones)
        return {learner_id: int(model.rollout_buffer.buffer_size) for learner_id, model in self.models.items()}

    def train(self, progress_callback: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        completed = 0
        updates = 0
        while completed < self.training_config.total_timesteps:
            self.collect_rollout()
            completed = min(completed + self.training_config.n_steps, self.training_config.total_timesteps)
            for model in self.models.values():
                model._update_current_progress_remaining(completed, self.training_config.total_timesteps)
                model.num_timesteps = completed
            for model in self.models.values():
                model.train()
            updates += 1
            if progress_callback is not None:
                progress_callback(completed, self.training_config.total_timesteps)
        return {
            "timesteps_per_agent": completed,
            "rollout_timesteps_per_agent": updates * self.training_config.n_steps,
            "updates": updates,
            "learner_ids": list(self.training_config.learner_ids),
        }

    def save(self, checkpoint_paths: Mapping[str, Path]) -> None:
        for learner_id, model in self.models.items():
            path = checkpoint_paths[learner_id]
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(path))
