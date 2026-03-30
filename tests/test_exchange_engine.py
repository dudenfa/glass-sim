from __future__ import annotations

from marl_trading.exchange import ExchangeKernel, Order, OrderType, Side
from marl_trading.exchange.events import OrderAcceptedEvent, OrderCanceledEvent, TradeEvent


def test_exchange_kernel_logs_accept_trade_and_cancel_events() -> None:
    exchange = ExchangeKernel()

    sell = Order(
        order_id="s1",
        agent_id="maker",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=5,
        price=101,
        timestamp=1,
    )
    buy = Order(
        order_id="b1",
        agent_id="taker",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=5,
        timestamp=2,
    )

    exchange.submit_order(sell)
    exchange.submit_order(buy)

    resting_buy = Order(
        order_id="b2",
        agent_id="agent_b",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=7,
        price=99,
        timestamp=3,
    )
    exchange.submit_order(resting_buy)
    exchange.cancel_order("b2", timestamp=4)

    assert isinstance(exchange.event_log[0], OrderAcceptedEvent)
    assert isinstance(exchange.event_log[1], OrderAcceptedEvent)
    assert isinstance(exchange.event_log[2], TradeEvent)
    assert isinstance(exchange.event_log[-1], OrderCanceledEvent)
    assert exchange.snapshot().best_bid is None
    assert exchange.snapshot().best_ask is None
