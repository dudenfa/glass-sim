from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from marl_trading.core.domain import AgentId, AssetSymbol, OrderId


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    RESTING = "resting"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Order:
    order_id: OrderId
    agent_id: AgentId
    symbol: AssetSymbol
    side: Side
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    timestamp_ns: int = 0
    status: OrderStatus = field(default=OrderStatus.PENDING)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require a limit_price.")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("Market orders must not define a limit_price.")


@dataclass(frozen=True)
class Trade:
    trade_id: str
    symbol: AssetSymbol
    price: float
    quantity: float
    taker_order_id: OrderId
    maker_order_id: Optional[OrderId]
    taker_agent_id: AgentId
    maker_agent_id: Optional[AgentId]
    timestamp_ns: int
