from __future__ import annotations

from collections import deque
from itertools import count
from typing import Deque, Dict, List, Optional

from .errors import OrderNotFoundError
from .models import BookLevel, Order, OrderBookSnapshot, OrderStatus, OrderType, Side, Trade


class LimitOrderBook:
    def __init__(self) -> None:
        self._bids: Dict[int, Deque[Order]] = {}
        self._asks: Dict[int, Deque[Order]] = {}
        self._orders_by_id: Dict[str, Order] = {}
        self._trade_counter = count(start=1)

    @property
    def resting_orders(self) -> Dict[str, Order]:
        return self._orders_by_id

    def _book_side(self, side: Side) -> Dict[int, Deque[Order]]:
        return self._bids if side is Side.BUY else self._asks

    def _opposite_side(self, side: Side) -> Dict[int, Deque[Order]]:
        return self._asks if side is Side.BUY else self._bids

    def _sorted_prices(self, side: Side) -> List[int]:
        prices = list(self._book_side(side).keys())
        return sorted(prices, reverse=(side is Side.BUY))

    def _best_price(self, side: Side) -> Optional[int]:
        prices = self._sorted_prices(side)
        return prices[0] if prices else None

    def best_bid(self) -> Optional[int]:
        return self._best_price(Side.BUY)

    def best_ask(self) -> Optional[int]:
        return self._best_price(Side.SELL)

    def spread(self) -> Optional[int]:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def mid_price(self) -> Optional[float]:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def _is_crossing(self, order: Order, best_opposite_price: Optional[int]) -> bool:
        if best_opposite_price is None:
            return False
        if order.order_type is OrderType.MARKET:
            return True
        assert order.price is not None
        if order.side is Side.BUY:
            return order.price >= best_opposite_price
        return order.price <= best_opposite_price

    def _remove_order_from_level(self, side: Side, price: int, order_id: str) -> None:
        levels = self._book_side(side)
        queue = levels.get(price)
        if queue is None:
            return
        for index, resting_order in enumerate(queue):
            if resting_order.order_id == order_id:
                del queue[index]
                break
        if not queue:
            del levels[price]

    def cancel_order(self, order_id: str) -> Order:
        order = self._orders_by_id.pop(order_id, None)
        if order is None:
            raise OrderNotFoundError(f"Unknown resting order: {order_id}")
        if order.price is None:
            raise OrderNotFoundError(f"Order {order_id} is not resting in the book.")

        self._remove_order_from_level(order.side, order.price, order.order_id)
        order.status = OrderStatus.CANCELED
        return order

    def _match_against_level(self, incoming: Order, opposite_price: int, timestamp: int) -> List[Trade]:
        trades: List[Trade] = []
        queue = self._opposite_side(incoming.side)[opposite_price]

        while incoming.remaining_quantity > 0 and queue:
            resting_order = queue[0]
            trade_quantity = min(incoming.remaining_quantity, resting_order.remaining_quantity)
            if trade_quantity <= 0:
                break

            incoming.remaining_quantity -= trade_quantity
            resting_order.remaining_quantity -= trade_quantity

            if incoming.side is Side.BUY:
                buy_order = incoming
                sell_order = resting_order
            else:
                buy_order = resting_order
                sell_order = incoming

            trade = Trade(
                trade_id=str(next(self._trade_counter)),
                timestamp=timestamp,
                price=opposite_price,
                quantity=trade_quantity,
                buy_order_id=buy_order.order_id,
                sell_order_id=sell_order.order_id,
                buy_agent_id=buy_order.agent_id,
                sell_agent_id=sell_order.agent_id,
                taker_order_id=incoming.order_id,
                maker_order_id=resting_order.order_id,
                aggressor_side=incoming.side,
            )
            trades.append(trade)

            if resting_order.remaining_quantity == 0:
                queue.popleft()
                del self._orders_by_id[resting_order.order_id]
                resting_order.status = OrderStatus.FILLED
            else:
                resting_order.status = OrderStatus.PARTIALLY_FILLED

        if not queue:
            del self._opposite_side(incoming.side)[opposite_price]

        return trades

    def submit_order(self, order: Order) -> List[Trade]:
        if order.order_id in self._orders_by_id:
            raise ValueError(f"Duplicate order_id: {order.order_id}")

        if order.order_type is OrderType.LIMIT:
            best_opposite_price = self.best_ask() if order.side is Side.BUY else self.best_bid()
        else:
            best_opposite_price = self.best_ask() if order.side is Side.BUY else self.best_bid()

        trades: List[Trade] = []
        while order.remaining_quantity > 0 and self._is_crossing(order, best_opposite_price):
            if order.side is Side.BUY:
                best_opposite_price = self.best_ask()
            else:
                best_opposite_price = self.best_bid()
            if best_opposite_price is None:
                break
            trades.extend(self._match_against_level(order, best_opposite_price, order.timestamp))

        if order.order_type is OrderType.LIMIT and order.remaining_quantity > 0:
            order.status = OrderStatus.OPEN if order.remaining_quantity == order.quantity else OrderStatus.PARTIALLY_FILLED
            levels = self._book_side(order.side)
            queue = levels.setdefault(order.price, deque())  # type: ignore[arg-type]
            queue.append(order)
            self._orders_by_id[order.order_id] = order
        else:
            if order.remaining_quantity == 0:
                order.status = OrderStatus.FILLED
            elif trades:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.EXPIRED

        return trades

    def snapshot(self, depth: int = 5, timestamp: int = 0) -> OrderBookSnapshot:
        bid_prices = self._sorted_prices(Side.BUY)[:depth]
        ask_prices = self._sorted_prices(Side.SELL)[:depth]
        bids = tuple(
            BookLevel(price=price, quantity=sum(order.remaining_quantity for order in self._bids[price]))
            for price in bid_prices
        )
        asks = tuple(
            BookLevel(price=price, quantity=sum(order.remaining_quantity for order in self._asks[price]))
            for price in ask_prices
        )
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        spread = self.spread()
        mid_price = self.mid_price()
        return OrderBookSnapshot(
            timestamp=timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            mid_price=mid_price,
            bids=bids,
            asks=asks,
        )
