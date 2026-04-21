from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from marl_trading.exchange.models import OrderType, Side


@dataclass(frozen=True)
class MarketObservation:
    timestamp_ns: int
    symbol: str
    tick_size: float
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    spread: float | None
    latent_fundamental: float
    recent_midpoints: tuple[float, ...]
    recent_returns_bps: tuple[float, ...]
    news_headline: str | None
    news_severity: float | None
    agent_cash: float
    agent_inventory: float
    agent_equity: float
    open_orders: int
    active_agents: int
    portfolio_active: bool
    agent_type: str
    public_note: str = ""


@dataclass(frozen=True)
class OrderIntent:
    side: Side
    order_type: OrderType
    quantity: int
    limit_price: float | None = None
    annotation: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive.")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit intents require a limit_price.")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("Market intents must not define a limit_price.")


class ScriptedAgent(ABC):
    def __init__(self, agent_id: str, agent_type: str, max_resting_orders: int = 3) -> None:
        self.agent_id = str(agent_id)
        self.agent_type = str(agent_type)
        self.max_resting_orders = int(max_resting_orders)

    def bootstrap(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        return ()

    def decide(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        return ()


def _midpoint_or_fallback(observation: MarketObservation) -> float:
    if observation.midpoint is not None:
        return float(observation.midpoint)
    return float(observation.latent_fundamental)


def _clamp_price(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive.")
    rounded = round(price / tick_size) * tick_size
    return max(tick_size, float(rounded))
