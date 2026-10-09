from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from marl_trading.core.domain import AgentId, AssetSymbol
from marl_trading.core.orders import Order, Trade


class EventType(str, Enum):
    ORDER_SUBMITTED = "order_submitted"
    ORDER_CANCELED = "order_canceled"
    ORDER_FILLED = "order_filled"
    TRADE = "trade"
    NEWS = "news"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True)
class MarketEvent:
    event_type: EventType
    timestamp_ns: int
    payload: Mapping[str, object]


@dataclass(frozen=True)
class MarketNews:
    news_id: str
    timestamp_ns: int
    headline: str
    impact: float = 0.0
    symbol: Optional[AssetSymbol] = None


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp_ns: int
    symbol: AssetSymbol
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    total_bid_depth: float
    total_ask_depth: float


@dataclass(frozen=True)
class EventBundle:
    """Convenience container for a market event and its derived payloads."""

    event: MarketEvent
    order: Optional[Order] = None
    trade: Optional[Trade] = None
    news: Optional[MarketNews] = None
