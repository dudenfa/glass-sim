from __future__ import annotations

import pytest

from marl_trading.exchange import LimitOrderBook, Order, OrderStatus, OrderType, Side
from marl_trading.exchange.errors import OrderNotFoundError


def test_limit_orders_rest_and_snapshot() -> None:
    book = LimitOrderBook()

    buy = Order(
        order_id="b1",
        agent_id="agent_a",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=99,
        timestamp=1,
    )
    sell = Order(
        order_id="s1",
        agent_id="agent_b",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=12,
        price=101,
        timestamp=2,
    )

    assert book.submit_order(buy) == []
    assert book.submit_order(sell) == []

    snapshot = book.snapshot(depth=5, timestamp=3)
    assert snapshot.best_bid == 99
    assert snapshot.best_ask == 101
    assert snapshot.spread == 2
    assert snapshot.mid_price == 100.0
    assert snapshot.bids[0].price == 99
    assert snapshot.bids[0].quantity == 10
    assert snapshot.asks[0].price == 101
    assert snapshot.asks[0].quantity == 12
    assert buy.status is OrderStatus.OPEN
    assert sell.status is OrderStatus.OPEN


def test_price_time_priority_and_partial_fill_order() -> None:
    book = LimitOrderBook()

    first_sell = Order(
        order_id="s1",
        agent_id="maker_1",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=5,
        price=101,
        timestamp=1,
    )
    second_sell = Order(
        order_id="s2",
        agent_id="maker_2",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=7,
        price=101,
        timestamp=2,
    )
    book.submit_order(first_sell)
    book.submit_order(second_sell)

    aggressive_buy = Order(
        order_id="b1",
        agent_id="taker",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
        timestamp=3,
    )
    trades = book.submit_order(aggressive_buy)

    assert [trade.sell_order_id for trade in trades] == ["s1", "s2"]
    assert [trade.quantity for trade in trades] == [5, 5]
    assert first_sell.status is OrderStatus.FILLED
    assert second_sell.status is OrderStatus.PARTIALLY_FILLED
    assert second_sell.remaining_quantity == 2
    assert aggressive_buy.status is OrderStatus.FILLED
    assert book.best_ask() == 101
    assert book.snapshot(depth=5).asks[0].quantity == 2


def test_aggressive_limit_order_executes_then_rests_remainder() -> None:
    book = LimitOrderBook()

    ask_1 = Order(
        order_id="s1",
        agent_id="maker_1",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=3,
        price=101,
        timestamp=1,
    )
    ask_2 = Order(
        order_id="s2",
        agent_id="maker_2",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=3,
        price=102,
        timestamp=2,
    )
    book.submit_order(ask_1)
    book.submit_order(ask_2)

    aggressive_buy = Order(
        order_id="b1",
        agent_id="taker",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=105,
        timestamp=3,
    )
    trades = book.submit_order(aggressive_buy)

    assert [trade.price for trade in trades] == [101, 102]
    assert [trade.quantity for trade in trades] == [3, 3]
    assert aggressive_buy.remaining_quantity == 4
    assert aggressive_buy.status is OrderStatus.OPEN
    snapshot = book.snapshot(depth=5)
    assert snapshot.best_bid == 105
    assert snapshot.best_ask is None
    assert snapshot.bids[0].price == 105
    assert snapshot.bids[0].quantity == 4


def test_cancel_order_removes_resting_liquidity() -> None:
    book = LimitOrderBook()

    order = Order(
        order_id="b1",
        agent_id="agent_a",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=8,
        price=99,
        timestamp=1,
    )
    book.submit_order(order)
    assert book.best_bid() == 99

    canceled = book.cancel_order("b1")
    assert canceled.status is OrderStatus.CANCELED
    assert book.best_bid() is None
    assert book.snapshot().bids == ()


def test_cancel_unknown_order_raises() -> None:
    book = LimitOrderBook()

    with pytest.raises(OrderNotFoundError):
        book.cancel_order("missing")
