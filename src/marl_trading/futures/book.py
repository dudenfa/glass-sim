"""A price-time priority limit order book on integer ticks.

Design notes:
- each price level keeps a FIFO deque of orders and a running total quantity;
- canceled orders are removed lazily (flagged inactive and skipped when they
  reach the front of the queue), so cancel is O(1);
- the sorted price ladders are maintained incrementally with ``bisect``, so
  best bid/ask is O(1);
- self-trade prevention: an incoming order never trades with a resting order
  of the same owner; the resting order is canceled instead ("cancel resting").

The book knows nothing about accounts, margin or fees. The market decides
whether an order may be submitted; the book only matches.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterator, List, Optional, Tuple

from .types import Side, TimeInForce


class RestingOrder:
    __slots__ = ("order_id", "agent_id", "side", "price", "quantity", "remaining", "time", "seq", "tif", "active", "tag")

    def __init__(
        self,
        order_id: str,
        agent_id: str,
        side: Side,
        price: int,
        quantity: int,
        time: float,
        seq: int,
        tif: TimeInForce = TimeInForce.GTC,
        tag: str = "",
    ) -> None:
        self.order_id = order_id
        self.agent_id = agent_id
        self.side = side
        self.price = price
        self.quantity = quantity
        self.remaining = quantity
        self.time = time
        self.seq = seq
        self.tif = tif
        self.active = True
        self.tag = tag

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RestingOrder({self.order_id}, {self.agent_id}, {self.side.value}, {self.price}, {self.remaining}/{self.quantity})"


@dataclass
class _Level:
    orders: Deque[RestingOrder] = field(default_factory=deque)
    total: int = 0
    count: int = 0


@dataclass(frozen=True)
class Match:
    """A fill between an incoming (taker) order and a resting (maker) order."""

    maker: RestingOrder
    price: int
    quantity: int


@dataclass
class MatchResult:
    matches: List[Match]
    self_trade_canceled: List[RestingOrder]
    filled: int


class OrderBook:
    def __init__(self) -> None:
        self._levels: Dict[Side, Dict[int, _Level]] = {Side.BUY: {}, Side.SELL: {}}
        # Both ladders are ascending; best bid is the last bid price, best ask the first ask price.
        self._prices: Dict[Side, List[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: Dict[str, RestingOrder] = {}

    # ------------------------------------------------------------------ queries

    def __len__(self) -> int:
        return len(self._orders)

    def __contains__(self, order_id: str) -> bool:
        return order_id in self._orders

    def get(self, order_id: str) -> Optional[RestingOrder]:
        return self._orders.get(order_id)

    def best_bid(self) -> Optional[int]:
        prices = self._prices[Side.BUY]
        return prices[-1] if prices else None

    def best_ask(self) -> Optional[int]:
        prices = self._prices[Side.SELL]
        return prices[0] if prices else None

    def best(self, side: Side) -> Optional[int]:
        return self.best_bid() if side is Side.BUY else self.best_ask()

    def mid(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread(self) -> Optional[int]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def _ladder(self, side: Side) -> Iterator[int]:
        prices = self._prices[side]
        return reversed(prices) if side is Side.BUY else iter(prices)

    def levels(self, side: Side, depth: int = 10) -> List[Tuple[int, int, int]]:
        """Aggregated depth: ``[(price, total_quantity, order_count), ...]`` best first."""
        out: List[Tuple[int, int, int]] = []
        book = self._levels[side]
        for price in self._ladder(side):
            level = book[price]
            out.append((price, level.total, level.count))
            if len(out) >= depth:
                break
        return out

    def orders(self, side: Side, depth: int = 10) -> List[RestingOrder]:
        """Order-level (L3) view of the best ``depth`` price levels, in priority order."""
        out: List[RestingOrder] = []
        book = self._levels[side]
        seen = 0
        for price in self._ladder(side):
            for order in book[price].orders:
                if order.active:
                    out.append(order)
            seen += 1
            if seen >= depth:
                break
        return out

    def depth_within(self, side: Side, ticks_from_best: int) -> int:
        best = self.best(side)
        if best is None:
            return 0
        total = 0
        book = self._levels[side]
        for price in self._ladder(side):
            if abs(price - best) > ticks_from_best:
                break
            total += book[price].total
        return total

    def orders_of(self, agent_id: str) -> List[RestingOrder]:
        return [order for order in self._orders.values() if order.agent_id == agent_id]

    def estimate_fill(self, side: Side, quantity: int, agent_id: str, limit: Optional[int] = None) -> Tuple[int, int]:
        """Walk the opposite side without mutating it.

        Returns ``(fillable_quantity, notional_in_ticks)`` ignoring the agent's own
        orders (which self-trade prevention would cancel).
        """
        remaining = quantity
        notional = 0
        opposite = side.opposite
        book = self._levels[opposite]
        for price in self._ladder(opposite):
            if limit is not None and ((side is Side.BUY and price > limit) or (side is Side.SELL and price < limit)):
                break
            for order in book[price].orders:
                if not order.active or order.agent_id == agent_id:
                    continue
                take = min(remaining, order.remaining)
                remaining -= take
                notional += take * price
                if remaining == 0:
                    return quantity, notional
        return quantity - remaining, notional

    def would_cross(self, side: Side, price: int) -> bool:
        best_opposite = self.best(side.opposite)
        if best_opposite is None:
            return False
        return price >= best_opposite if side is Side.BUY else price <= best_opposite

    # ---------------------------------------------------------------- mutation

    def _add_level_price(self, side: Side, price: int) -> None:
        insort(self._prices[side], price)

    def _remove_level_price(self, side: Side, price: int) -> None:
        prices = self._prices[side]
        index = bisect_left(prices, price)
        if index < len(prices) and prices[index] == price:
            del prices[index]

    def add(self, order: RestingOrder) -> None:
        """Rest an order in the book. The caller must ensure it does not cross."""
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order id {order.order_id}")
        book = self._levels[order.side]
        level = book.get(order.price)
        if level is None:
            level = _Level()
            book[order.price] = level
            self._add_level_price(order.side, order.price)
        level.orders.append(order)
        level.total += order.remaining
        level.count += 1
        self._orders[order.order_id] = order

    def cancel(self, order_id: str) -> Optional[RestingOrder]:
        order = self._orders.pop(order_id, None)
        if order is None:
            return None
        order.active = False
        book = self._levels[order.side]
        level = book[order.price]
        level.total -= order.remaining
        level.count -= 1
        if level.count == 0:
            del book[order.price]
            self._remove_level_price(order.side, order.price)
        return order

    def match(self, side: Side, quantity: int, agent_id: str, limit: Optional[int] = None) -> MatchResult:
        """Match an incoming order against the opposite side.

        ``limit`` is the worst acceptable price (``None`` for a market order).
        Resting orders owned by ``agent_id`` are canceled, never traded against.
        """
        remaining = quantity
        matches: List[Match] = []
        stp: List[RestingOrder] = []
        opposite = side.opposite
        book = self._levels[opposite]
        prices = self._prices[opposite]
        while remaining > 0 and prices:
            price = prices[0] if opposite is Side.SELL else prices[-1]
            if limit is not None and ((side is Side.BUY and price > limit) or (side is Side.SELL and price < limit)):
                break
            level = book[price]
            queue = level.orders
            while remaining > 0 and queue:
                maker = queue[0]
                if not maker.active:
                    queue.popleft()
                    continue
                if maker.agent_id == agent_id:
                    queue.popleft()
                    self._orders.pop(maker.order_id, None)
                    maker.active = False
                    level.total -= maker.remaining
                    level.count -= 1
                    stp.append(maker)
                    continue
                take = min(remaining, maker.remaining)
                remaining -= take
                maker.remaining -= take
                level.total -= take
                matches.append(Match(maker=maker, price=price, quantity=take))
                if maker.remaining == 0:
                    queue.popleft()
                    maker.active = False
                    level.count -= 1
                    self._orders.pop(maker.order_id, None)
            if level.count == 0:
                del book[price]
                self._remove_level_price(opposite, price)
        return MatchResult(matches=matches, self_trade_canceled=stp, filled=quantity - remaining)
