from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import List, Optional

from .book import LimitOrderBook
from .events import OrderAcceptedEvent, OrderCanceledEvent, TradeEvent
from .errors import OrderNotFoundError
from .models import Order, OrderBookSnapshot, Trade


@dataclass
class ExchangeKernel:
    book: LimitOrderBook = field(default_factory=LimitOrderBook)
    event_log: List[object] = field(default_factory=list)
    _event_counter: count = field(default_factory=lambda: count(start=1), init=False, repr=False)

    def _next_event_id(self) -> int:
        return next(self._event_counter)

    def submit_order(self, order: Order) -> List[Trade]:
        event_id = self._next_event_id()
        self.event_log.append(
            OrderAcceptedEvent(
                event_id=event_id,
                timestamp=order.timestamp,
                order_id=order.order_id,
                agent_id=order.agent_id,
                side=order.side,
                order_type=order.order_type,
                price=order.price,
                quantity=order.quantity,
            )
        )

        trades = self.book.submit_order(order)
        for trade in trades:
            self.event_log.append(TradeEvent(event_id=self._next_event_id(), trade=trade))
        return trades

    def cancel_order(self, order_id: str, timestamp: int = 0) -> Order:
        try:
            order = self.book.cancel_order(order_id)
        except OrderNotFoundError:
            raise

        self.event_log.append(
            OrderCanceledEvent(
                event_id=self._next_event_id(),
                timestamp=timestamp,
                order_id=order.order_id,
                agent_id=order.agent_id,
            )
        )
        return order

    def snapshot(self, depth: int = 5, timestamp: int = 0) -> OrderBookSnapshot:
        return self.book.snapshot(depth=depth, timestamp=timestamp)

    def best_bid(self) -> Optional[int]:
        return self.book.best_bid()

    def best_ask(self) -> Optional[int]:
        return self.book.best_ask()

    def clear(self) -> None:
        self.book = LimitOrderBook()
        self.event_log.clear()
        self._event_counter = count(start=1)
