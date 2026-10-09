from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .errors import InvalidOrderError


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass
class Order:
    order_id: str
    agent_id: str
    side: Side
    order_type: OrderType
    quantity: int
    price: Optional[int] = None
    timestamp: int = 0
    sequence: int = 0
    remaining_quantity: int = field(init=False)
    status: OrderStatus = field(init=False, default=OrderStatus.PENDING)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise InvalidOrderError("Order quantity must be positive.")
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise InvalidOrderError("Limit orders require a price.")
        if self.order_type is OrderType.MARKET and self.price is not None:
            raise InvalidOrderError("Market orders must not define a price.")
        self.remaining_quantity = self.quantity


@dataclass(frozen=True)
class BookLevel:
    price: int
    quantity: int


@dataclass(frozen=True)
class OrderBookSnapshot:
    timestamp: int
    best_bid: Optional[int]
    best_ask: Optional[int]
    spread: Optional[int]
    mid_price: Optional[float]
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]


@dataclass(frozen=True)
class Trade:
    trade_id: str
    timestamp: int
    price: int
    quantity: int
    buy_order_id: str
    sell_order_id: str
    buy_agent_id: str
    sell_agent_id: str
    taker_order_id: str
    maker_order_id: str
    aggressor_side: Side
