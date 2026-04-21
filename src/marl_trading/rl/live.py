from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from marl_trading.agents.base import MarketObservation, OrderIntent, ScriptedAgent

from .boundary import (
    PHASE_A_ACTION_TYPE_ORDER,
    RLAction,
    RLActionType,
    action_to_order_intent,
    build_action_mask,
    feature_vector,
    mask_invalid_action,
)

_ACTION_TYPE_ORDER = tuple(RLActionType)


def _checkpoint_sidecar_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = checkpoint_path.resolve().with_suffix(".json")
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
    effective_rl_action: RLAction | None = None
    invalid_reason: str | None = None


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
    *,
    action_types: tuple[RLActionType, ...] = _ACTION_TYPE_ORDER,
    fixed_quantity: int = 1,
    fixed_price_offset_ticks: int = 1,
) -> RLAction:
    if isinstance(raw_action, RLAction):
        return raw_action

    values = np.asarray(raw_action, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("Policy action cannot be empty.")

    action_index = int(values[0])
    if action_index < 0 or action_index >= len(action_types):
        raise ValueError(f"Unsupported policy action index: {action_index}")

    quantity = int(values[1]) + 1 if values.size >= 2 else int(fixed_quantity)
    price_offset_ticks = int(values[2]) + 1 if values.size >= 3 else int(fixed_price_offset_ticks)
    return RLAction(
        action_type=action_types[action_index],
        quantity=max(quantity, 1),
        price_offset_ticks=max(price_offset_ticks, 1),
    )


class ModelPolicyAdapter:
    def __init__(
        self,
        predictor: PolicyPredictor,
        *,
        deterministic: bool = True,
        action_types: tuple[RLActionType, ...] = PHASE_A_ACTION_TYPE_ORDER,
        fixed_quantity: int = 1,
        fixed_price_offset_ticks: int = 1,
    ) -> None:
        self.predictor = predictor
        self.deterministic = bool(deterministic)
        self.action_types = tuple(action_types)
        self.fixed_quantity = max(int(fixed_quantity), 1)
        self.fixed_price_offset_ticks = max(int(fixed_price_offset_ticks), 1)

    def action_for(self, observation: MarketObservation) -> RuntimePolicyDecision:
        features = feature_vector(observation)
        model_input = np.asarray(features, dtype=np.float32)
        action_mask = np.asarray(
            build_action_mask(
                observation,
                action_types=self.action_types,
                quantity=self.fixed_quantity,
                price_offset_ticks=self.fixed_price_offset_ticks,
            ),
            dtype=bool,
        )
        try:
            raw_action, _state = self.predictor.predict(
                model_input,
                deterministic=self.deterministic,
                action_masks=action_mask,
            )
        except TypeError:
            raw_action, _state = self.predictor.predict(model_input, deterministic=self.deterministic)
        normalized = tuple(int(value) for value in np.asarray(raw_action, dtype=np.int64).reshape(-1).tolist())
        return RuntimePolicyDecision(
            features=features,
            raw_action=normalized,
            rl_action=decode_policy_action(
                normalized,
                action_types=self.action_types,
                fixed_quantity=self.fixed_quantity,
                fixed_price_offset_ticks=self.fixed_price_offset_ticks,
            ),
        )


class PPOPolicyAdapter(ModelPolicyAdapter):
    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = True,
        action_types: tuple[RLActionType, ...] = PHASE_A_ACTION_TYPE_ORDER,
        fixed_quantity: int = 1,
        fixed_price_offset_ticks: int = 1,
    ) -> "PPOPolicyAdapter":
        resolved_path = Path(checkpoint_path).resolve()
        metadata = _checkpoint_sidecar_metadata(resolved_path)
        algorithm = str(metadata.get("algorithm") or "ppo")
        if metadata.get("phase_a_action_space") is False:
            action_types = _ACTION_TYPE_ORDER
        elif bool(metadata.get("include_cancel_action")):
            action_types = PHASE_A_ACTION_TYPE_ORDER + (RLActionType.CANCEL_OLDEST,)
        if "fixed_order_quantity" in metadata:
            fixed_quantity = int(metadata.get("fixed_order_quantity") or fixed_quantity)
        if "fixed_price_offset_ticks" in metadata:
            fixed_price_offset_ticks = int(metadata.get("fixed_price_offset_ticks") or fixed_price_offset_ticks)

        if algorithm == "maskable_ppo":
            algo_module = importlib.import_module("sb3_contrib.ppo_mask")
            model_class = getattr(algo_module, "MaskablePPO")
        else:
            stable_baselines3 = importlib.import_module("stable_baselines3")
            model_class = getattr(stable_baselines3, "PPO")
        normalized_path = _normalize_checkpoint_load_path(resolved_path)
        model = model_class.load(normalized_path, device=device)
        return cls(
            model,
            deterministic=deterministic,
            action_types=action_types,
            fixed_quantity=fixed_quantity,
            fixed_price_offset_ticks=fixed_price_offset_ticks,
        )

    @classmethod
    def try_load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = True,
        action_types: tuple[RLActionType, ...] = PHASE_A_ACTION_TYPE_ORDER,
        fixed_quantity: int = 1,
        fixed_price_offset_ticks: int = 1,
    ) -> tuple["PPOPolicyAdapter | None", PolicyLoadStatus]:
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {path}")
        try:
            adapter = cls.load(
                path,
                device=device,
                deterministic=deterministic,
                action_types=action_types,
                fixed_quantity=fixed_quantity,
                fixed_price_offset_ticks=fixed_price_offset_ticks,
            )
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
        self.requested_action_counts = {action_type.value: 0 for action_type in _ACTION_TYPE_ORDER}
        self.invalid_action_count = 0
        self.invalid_action_reasons: dict[str, int] = {}

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
            effective_action, invalid_reason = mask_invalid_action(decision.rl_action, observation)
            decision = replace(
                decision,
                effective_rl_action=effective_action,
                invalid_reason=invalid_reason,
            )
            self.last_decision = decision
            self.last_failure_reason = invalid_reason
            self.decision_count += 1
            self.requested_action_counts[decision.rl_action.action_type.value] += 1
            self.action_counts[effective_action.action_type.value] += 1
            if invalid_reason is not None:
                self.invalid_action_count += 1
                self.invalid_action_reasons[invalid_reason] = self.invalid_action_reasons.get(invalid_reason, 0) + 1
            return action_to_order_intent(effective_action, observation)
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
        latest_action = (
            self.last_decision.effective_rl_action.action_type.value
            if self.last_decision is not None and self.last_decision.effective_rl_action is not None
            else None
        )
        latest_requested_action = self.last_decision.rl_action.action_type.value if self.last_decision is not None else None
        return {
            "decision_count": int(self.decision_count),
            "action_counts": dict(self.action_counts),
            "requested_action_counts": dict(self.requested_action_counts),
            "last_action_type": latest_action,
            "last_requested_action_type": latest_requested_action,
            "last_failure_reason": self.last_failure_reason,
            "invalid_action_count": int(self.invalid_action_count),
            "invalid_action_reasons": dict(self.invalid_action_reasons),
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
