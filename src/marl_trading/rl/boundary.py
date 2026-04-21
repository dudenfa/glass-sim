from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from marl_trading.agents.base import MarketObservation, OrderIntent, _clamp_price, _midpoint_or_fallback
from marl_trading.exchange.models import OrderType, Side


class RLActionType(str, Enum):
    HOLD = "hold"
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    CANCEL_OLDEST = "cancel_oldest"


PHASE_A_ACTION_TYPE_ORDER = (
    RLActionType.HOLD,
    RLActionType.MARKET_BUY,
    RLActionType.MARKET_SELL,
    RLActionType.LIMIT_BUY,
    RLActionType.LIMIT_SELL,
)


@dataclass(frozen=True)
class RLAction:
    action_type: RLActionType
    quantity: int = 1
    price_offset_ticks: int = 1


@dataclass(frozen=True)
class RewardBreakdown:
    realized_pnl_delta: float
    equity_delta: float
    signal_reward: float
    inactivity_penalty: float
    inventory_penalty: float
    inventory_risk_penalty: float
    total_reward: float


def observation_to_feature_dict(observation: MarketObservation) -> dict[str, float]:
    midpoint = observation.midpoint if observation.midpoint is not None else observation.latent_fundamental
    best_bid = observation.best_bid if observation.best_bid is not None else midpoint
    best_ask = observation.best_ask if observation.best_ask is not None else midpoint
    spread = observation.spread if observation.spread is not None else max(best_ask - best_bid, 0.0)
    recent_returns = list(observation.recent_returns_bps[-3:])
    while len(recent_returns) < 3:
        recent_returns.insert(0, 0.0)

    return {
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "has_best_bid": 1.0 if observation.best_bid is not None else 0.0,
        "has_best_ask": 1.0 if observation.best_ask is not None else 0.0,
        "midpoint": float(midpoint),
        "spread": float(spread),
        "fundamental": float(observation.latent_fundamental),
        "fundamental_gap": float(observation.latent_fundamental - midpoint),
        "return_bps_1": float(recent_returns[-1]),
        "return_bps_2": float(recent_returns[-2]),
        "return_bps_3": float(recent_returns[-3]),
        "news_severity": float(observation.news_severity or 0.0),
        "agent_cash": float(observation.agent_cash),
        "agent_inventory": float(observation.agent_inventory),
        "agent_equity": float(observation.agent_equity),
        "open_orders": float(observation.open_orders),
        "active_agents": float(observation.active_agents),
        "portfolio_active": 1.0 if observation.portfolio_active else 0.0,
    }


def feature_vector(observation: MarketObservation) -> tuple[float, ...]:
    features = observation_to_feature_dict(observation)
    return tuple(features[key] for key in (
        "best_bid",
        "best_ask",
        "has_best_bid",
        "has_best_ask",
        "midpoint",
        "spread",
        "fundamental",
        "fundamental_gap",
        "return_bps_1",
        "return_bps_2",
        "return_bps_3",
        "news_severity",
        "agent_cash",
        "agent_inventory",
        "agent_equity",
        "open_orders",
        "active_agents",
        "portfolio_active",
    ))


def action_to_order_intent(action: RLAction, observation: MarketObservation) -> OrderIntent | None:
    quantity = max(int(action.quantity), 1)
    offset_ticks = max(int(action.price_offset_ticks), 1)
    tick = float(observation.tick_size)
    midpoint = _midpoint_or_fallback(observation)

    if action.action_type is RLActionType.HOLD:
        return None
    if action.action_type is RLActionType.CANCEL_OLDEST:
        return None
    if action.action_type is RLActionType.MARKET_BUY:
        return OrderIntent(side=Side.BUY, order_type=OrderType.MARKET, quantity=quantity, annotation="rl_market_buy")
    if action.action_type is RLActionType.MARKET_SELL:
        return OrderIntent(side=Side.SELL, order_type=OrderType.MARKET, quantity=quantity, annotation="rl_market_sell")
    if action.action_type is RLActionType.LIMIT_BUY:
        anchor = observation.best_bid if observation.best_bid is not None else midpoint
        price = _clamp_price(anchor - offset_ticks * tick, tick)
        return OrderIntent(side=Side.BUY, order_type=OrderType.LIMIT, quantity=quantity, limit_price=price, annotation="rl_limit_buy")
    if action.action_type is RLActionType.LIMIT_SELL:
        anchor = observation.best_ask if observation.best_ask is not None else midpoint
        price = _clamp_price(anchor + offset_ticks * tick, tick)
        return OrderIntent(side=Side.SELL, order_type=OrderType.LIMIT, quantity=quantity, limit_price=price, annotation="rl_limit_sell")
    raise ValueError(f"Unsupported RL action type: {action.action_type}")


def mask_invalid_action(action: RLAction, observation: MarketObservation) -> tuple[RLAction, str | None]:
    quantity = max(int(action.quantity), 1)
    available_inventory = max(float(observation.agent_inventory), 0.0)
    open_orders = max(int(observation.open_orders), 0)

    if action.action_type is RLActionType.MARKET_BUY and observation.best_ask is None:
        return RLAction(RLActionType.HOLD), "no_ask_liquidity_for_market_buy"
    if action.action_type is RLActionType.MARKET_SELL and observation.best_bid is None:
        return RLAction(RLActionType.HOLD), "no_bid_liquidity_for_market_sell"
    if action.action_type in {RLActionType.MARKET_SELL, RLActionType.LIMIT_SELL} and available_inventory + 1e-9 < quantity:
        return RLAction(RLActionType.HOLD), "insufficient_inventory_for_sell"
    if action.action_type is RLActionType.CANCEL_OLDEST and open_orders <= 0:
        return RLAction(RLActionType.HOLD), "no_open_orders_to_cancel"
    return action, None


def is_action_valid(action: RLAction, observation: MarketObservation) -> tuple[bool, str | None]:
    effective_action, invalid_reason = mask_invalid_action(action, observation)
    return effective_action == action, invalid_reason


def build_action_mask(
    observation: MarketObservation,
    *,
    action_types: tuple[RLActionType, ...],
    quantity: int = 1,
    price_offset_ticks: int = 1,
) -> tuple[bool, ...]:
    quantity = max(int(quantity), 1)
    price_offset_ticks = max(int(price_offset_ticks), 1)
    mask: list[bool] = []
    for action_type in action_types:
        valid, _reason = is_action_valid(
            RLAction(
                action_type=action_type,
                quantity=quantity,
                price_offset_ticks=price_offset_ticks,
            ),
            observation,
        )
        mask.append(valid)
    return tuple(mask)


def compute_reward(
    *,
    previous_equity: float,
    current_equity: float,
    current_inventory: float,
    previous_realized_pnl: float = 0.0,
    current_realized_pnl: float = 0.0,
    realized_pnl_delta_coefficient: float = 0.0,
    equity_delta_coefficient: float = 1.0,
    inactivity_penalty_applied: bool = False,
    inactivity_penalty_coefficient: float = 0.0,
    absolute_inventory_penalty_coefficient: float = 0.0,
    inventory_penalty_coefficient: float | None = None,
    inventory_risk_penalty_coefficient: float = 0.0,
) -> RewardBreakdown:
    realized_pnl_delta = float(current_realized_pnl - previous_realized_pnl)
    equity_delta = float(current_equity - previous_equity)
    signal_reward = (
        realized_pnl_delta * float(realized_pnl_delta_coefficient)
        + equity_delta * float(equity_delta_coefficient)
    )
    absolute_penalty_coefficient = float(
        absolute_inventory_penalty_coefficient
        if inventory_penalty_coefficient is None
        else inventory_penalty_coefficient
    )
    inactivity_penalty = float(inactivity_penalty_coefficient) if inactivity_penalty_applied else 0.0
    inventory_penalty = abs(float(current_inventory)) * absolute_penalty_coefficient
    inventory_risk_penalty = (float(current_inventory) ** 2) * float(inventory_risk_penalty_coefficient)
    total_reward = signal_reward - inactivity_penalty - inventory_penalty - inventory_risk_penalty
    return RewardBreakdown(
        realized_pnl_delta=realized_pnl_delta,
        equity_delta=equity_delta,
        signal_reward=signal_reward,
        inactivity_penalty=inactivity_penalty,
        inventory_penalty=inventory_penalty,
        inventory_risk_penalty=inventory_risk_penalty,
        total_reward=total_reward,
    )


__all__ = [
    "RLAction",
    "RLActionType",
    "PHASE_A_ACTION_TYPE_ORDER",
    "RewardBreakdown",
    "action_to_order_intent",
    "build_action_mask",
    "compute_reward",
    "feature_vector",
    "is_action_valid",
    "mask_invalid_action",
    "observation_to_feature_dict",
]
