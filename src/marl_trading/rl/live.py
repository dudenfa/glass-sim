from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from marl_trading.agents.base import MarketObservation, OrderIntent, ScriptedAgent

from .boundary import RLAction, RLActionType, action_to_order_intent, feature_vector

_ACTION_TYPE_ORDER = tuple(RLActionType)


class PolicyPredictor(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        ...


class RuntimePolicyAdapter(Protocol):
    def action_for(self, observation: MarketObservation) -> "RuntimePolicyDecision":
        ...


@dataclass(frozen=True)
class RuntimePolicyDecision:
    features: tuple[float, ...]
    raw_action: tuple[int, ...]
    rl_action: RLAction


@dataclass(frozen=True)
class PolicyLoadStatus:
    available: bool
    checkpoint_path: Path
    reason: str | None = None


def _normalize_checkpoint_load_path(checkpoint_path: Path) -> str:
    resolved = checkpoint_path.resolve()
    if resolved.suffix == ".zip":
        return str(resolved.with_suffix(""))
    return str(resolved)


def decode_policy_action(
    raw_action: RLAction | np.ndarray | list[int] | tuple[int, ...] | int,
) -> RLAction:
    if isinstance(raw_action, RLAction):
        return raw_action

    values = np.asarray(raw_action, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("Policy action cannot be empty.")

    action_index = int(values[0])
    if action_index < 0 or action_index >= len(_ACTION_TYPE_ORDER):
        raise ValueError(f"Unsupported policy action index: {action_index}")

    quantity = int(values[1]) + 1 if values.size >= 2 else 1
    price_offset_ticks = int(values[2]) + 1 if values.size >= 3 else 1
    return RLAction(
        action_type=_ACTION_TYPE_ORDER[action_index],
        quantity=max(quantity, 1),
        price_offset_ticks=max(price_offset_ticks, 1),
    )


class ModelPolicyAdapter:
    def __init__(
        self,
        predictor: PolicyPredictor,
        *,
        deterministic: bool = True,
    ) -> None:
        self.predictor = predictor
        self.deterministic = bool(deterministic)

    def action_for(self, observation: MarketObservation) -> RuntimePolicyDecision:
        features = feature_vector(observation)
        model_input = np.asarray(features, dtype=np.float32)
        raw_action, _state = self.predictor.predict(model_input, deterministic=self.deterministic)
        normalized = tuple(int(value) for value in np.asarray(raw_action, dtype=np.int64).reshape(-1).tolist())
        return RuntimePolicyDecision(
            features=features,
            raw_action=normalized,
            rl_action=decode_policy_action(normalized),
        )


class PPOPolicyAdapter(ModelPolicyAdapter):
    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = True,
    ) -> "PPOPolicyAdapter":
        stable_baselines3 = importlib.import_module("stable_baselines3")
        model_class = getattr(stable_baselines3, "PPO")
        normalized_path = _normalize_checkpoint_load_path(Path(checkpoint_path))
        model = model_class.load(normalized_path, device=device)
        return cls(model, deterministic=deterministic)

    @classmethod
    def try_load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = True,
    ) -> tuple["PPOPolicyAdapter | None", PolicyLoadStatus]:
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {path}")
        try:
            adapter = cls.load(path, device=device, deterministic=deterministic)
        except ImportError:
            return None, PolicyLoadStatus(
                available=False,
                checkpoint_path=path,
                reason="stable-baselines3 is not installed.",
            )
        return adapter, PolicyLoadStatus(available=True, checkpoint_path=path, reason=None)


class RuntimePolicyControlledAgent(ScriptedAgent):
    def __init__(
        self,
        agent_id: str,
        *,
        policy: RuntimePolicyAdapter | PolicyPredictor | None,
        fallback_agent: ScriptedAgent | None = None,
        agent_type: str | None = None,
        max_resting_orders: int | None = None,
        delegate_bootstrap: bool = True,
    ) -> None:
        resolved_type = str(agent_type or getattr(fallback_agent, "agent_type", "rl_agent"))
        resolved_orders = int(max_resting_orders if max_resting_orders is not None else getattr(fallback_agent, "max_resting_orders", 1))
        super().__init__(agent_id=agent_id, agent_type=resolved_type, max_resting_orders=resolved_orders)
        self.policy = self._coerce_policy(policy)
        self.fallback_agent = fallback_agent
        self.delegate_bootstrap = bool(delegate_bootstrap)
        self.last_decision: RuntimePolicyDecision | None = None
        self.last_failure_reason: str | None = None
        self.decision_count = 0
        self.action_counts = {action_type.value: 0 for action_type in _ACTION_TYPE_ORDER}

    def _coerce_policy(
        self,
        policy: RuntimePolicyAdapter | PolicyPredictor | None,
    ) -> RuntimePolicyAdapter | None:
        if policy is None:
            return None
        if hasattr(policy, "action_for"):
            return policy  # type: ignore[return-value]
        if hasattr(policy, "predict"):
            return ModelPolicyAdapter(policy)  # type: ignore[arg-type]
        raise TypeError("policy must expose either action_for(...) or predict(...).")

    def bootstrap(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        if self.delegate_bootstrap and self.fallback_agent is not None:
            return self.fallback_agent.bootstrap(observation, rng)
        return ()

    def decide(self, observation: MarketObservation, rng) -> OrderIntent | None:  # noqa: ARG002
        self.last_decision = None
        self.last_failure_reason = None
        if self.policy is None:
            self.last_failure_reason = "policy_unavailable"
            self.decision_count += 1
            self.action_counts[RLActionType.HOLD.value] += 1
            return self._fallback_decide(observation, rng)
        try:
            decision = self.policy.action_for(observation)
            self.last_decision = decision
            self.decision_count += 1
            self.action_counts[decision.rl_action.action_type.value] += 1
            return action_to_order_intent(decision.rl_action, observation)
        except Exception as exc:  # pragma: no cover - guarded by focused tests
            self.last_failure_reason = str(exc)
            self.decision_count += 1
            self.action_counts[RLActionType.HOLD.value] += 1
            return self._fallback_decide(observation, rng)

    def _fallback_decide(self, observation: MarketObservation, rng) -> OrderIntent | None:
        if self.fallback_agent is None:
            return None
        return self.fallback_agent.decide(observation, rng)

    def diagnostics(self) -> dict[str, Any]:
        latest_action = self.last_decision.rl_action.action_type.value if self.last_decision is not None else None
        return {
            "decision_count": int(self.decision_count),
            "action_counts": dict(self.action_counts),
            "last_action_type": latest_action,
            "last_failure_reason": self.last_failure_reason,
        }


__all__ = [
    "ModelPolicyAdapter",
    "PPOPolicyAdapter",
    "PolicyLoadStatus",
    "PolicyPredictor",
    "RuntimePolicyControlledAgent",
    "RuntimePolicyDecision",
    "decode_policy_action",
]
