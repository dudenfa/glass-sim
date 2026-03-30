from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import OrderType, Side, Trade


class EventType(str, Enum):
    ORDER_ACCEPTED = "order_accepted"
    ORDER_CANCELED = "order_canceled"
    TRADE = "trade"


@dataclass(frozen=True)
class OrderAcceptedEvent:
    event_id: int
    timestamp: int
    order_id: str
    agent_id: str
    side: Side
    order_type: OrderType
    price: int | None
    quantity: int


@dataclass(frozen=True)
class OrderCanceledEvent:
    event_id: int
    timestamp: int
    order_id: str
    agent_id: str


@dataclass(frozen=True)
class TradeEvent:
    event_id: int
    trade: Trade
