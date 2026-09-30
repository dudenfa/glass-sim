import math

import pytest

from marl_trading.futures.account import MarginAccount
from marl_trading.futures.book import OrderBook, RestingOrder
from marl_trading.futures.types import LimitOrder, MarketOrder, Side, TimeInForce


def _order(order_id, agent, side, price, qty, seq):
    return RestingOrder(order_id, agent, side, price, qty, time=0.0, seq=seq)


# ------------------------------------------------------------------ order book


def test_best_prices_and_levels():
    book = OrderBook()
    book.add(_order("b1", "a", Side.BUY, 99, 5, 1))
    book.add(_order("b2", "b", Side.BUY, 100, 3, 2))
    book.add(_order("s1", "c", Side.SELL, 102, 4, 3))
    book.add(_order("s2", "d", Side.SELL, 101, 2, 4))
    assert book.best_bid() == 100
    assert book.best_ask() == 101
    assert book.mid() == 100.5
    assert book.levels(Side.BUY) == [(100, 3, 1), (99, 5, 1)]
    assert book.levels(Side.SELL) == [(101, 2, 1), (102, 4, 1)]


def test_price_time_priority_and_partial_fill():
    book = OrderBook()
    book.add(_order("s1", "first", Side.SELL, 101, 3, 1))
    book.add(_order("s2", "second", Side.SELL, 101, 3, 2))
    book.add(_order("s3", "third", Side.SELL, 102, 5, 3))
    result = book.match(Side.BUY, 5, "taker")
    assert [(m.maker.order_id, m.price, m.quantity) for m in result.matches] == [("s1", 101, 3), ("s2", 101, 2)]
    assert book.get("s2").remaining == 1
    assert "s1" not in book
    assert book.levels(Side.SELL) == [(101, 1, 1), (102, 5, 1)]


def test_limit_price_stops_matching():
    book = OrderBook()
    book.add(_order("s1", "m", Side.SELL, 101, 3, 1))
    book.add(_order("s2", "m", Side.SELL, 105, 3, 2))
    result = book.match(Side.BUY, 6, "taker", limit=103)
    assert result.filled == 3
    assert book.best_ask() == 105


def test_cancel_is_lazy_but_invisible():
    book = OrderBook()
    book.add(_order("s1", "m", Side.SELL, 101, 3, 1))
    book.add(_order("s2", "n", Side.SELL, 101, 2, 2))
    book.cancel("s1")
    assert book.levels(Side.SELL) == [(101, 2, 1)]
    result = book.match(Side.BUY, 2, "taker")
    assert [m.maker.order_id for m in result.matches] == ["s2"]
    assert book.best_ask() is None


def test_self_trade_prevention_cancels_resting():
    book = OrderBook()
    book.add(_order("own", "me", Side.SELL, 101, 3, 1))
    book.add(_order("other", "x", Side.SELL, 101, 3, 2))
    result = book.match(Side.BUY, 3, "me")
    assert [o.order_id for o in result.self_trade_canceled] == ["own"]
    assert [m.maker.order_id for m in result.matches] == ["other"]
    assert len(book) == 0


def test_estimate_fill_does_not_mutate_and_skips_own_orders():
    book = OrderBook()
    book.add(_order("own", "me", Side.SELL, 100, 10, 1))
    book.add(_order("s1", "x", Side.SELL, 101, 2, 2))
    book.add(_order("s2", "y", Side.SELL, 102, 5, 3))
    assert book.estimate_fill(Side.BUY, 4, "me") == (4, 2 * 101 + 2 * 102)
    assert book.estimate_fill(Side.BUY, 4, "me", limit=101) == (2, 202)
    assert len(book) == 3


# --------------------------------------------------------------------- account


def test_open_increase_reduce_close_long():
    acc = MarginAccount("a", 1_000.0)
    acc.apply_fill(Side.BUY, 10, 100.0, fee=0.0)
    acc.apply_fill(Side.BUY, 10, 110.0, fee=0.0)
    assert acc.position == 20
    assert acc.entry_price == pytest.approx(105.0)
    realized = acc.apply_fill(Side.SELL, 5, 115.0, fee=0.0)
    assert realized == pytest.approx(5 * 10.0)
    assert acc.position == 15
    assert acc.entry_price == pytest.approx(105.0)
    acc.apply_fill(Side.SELL, 15, 100.0, fee=0.0)
    assert acc.position == 0
    assert acc.entry_price == 0.0
    assert acc.realized_pnl == pytest.approx(50.0 - 75.0)
    assert acc.collateral == pytest.approx(1_000.0 - 25.0)


def test_short_pnl_and_flip_through_zero():
    acc = MarginAccount("a", 1_000.0)
    acc.apply_fill(Side.SELL, 10, 100.0, fee=0.0)
    assert acc.unrealized_pnl(90.0) == pytest.approx(100.0)
    realized = acc.apply_fill(Side.BUY, 15, 90.0, fee=1.0)
    assert realized == pytest.approx(100.0)
    assert acc.position == 5
    assert acc.entry_price == pytest.approx(90.0)
    assert acc.collateral == pytest.approx(1_000.0 + 100.0 - 1.0)


def test_equity_includes_unrealized_and_fees():
    acc = MarginAccount("a", 500.0, contract_size=2.0)
    acc.apply_fill(Side.BUY, 3, 100.0, fee=0.5)
    assert acc.equity(101.0) == pytest.approx(500.0 - 0.5 + 3 * 1.0 * 2.0)
    assert acc.notional(101.0) == pytest.approx(3 * 101.0 * 2.0)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_liquidation_price_is_where_equity_hits_maintenance(side):
    mmr = 0.0125
    acc = MarginAccount("a", 1_000.0)
    acc.apply_fill(side, 300, 100.0, fee=0.0)
    price = acc.liquidation_price(mmr)
    assert price is not None
    assert acc.equity(price) == pytest.approx(acc.maintenance_margin(price, mmr))
    assert (price < 100.0) == (side is Side.BUY)


def test_worst_case_margin_counts_open_orders():
    acc = MarginAccount("a", 1_000.0)
    acc.apply_fill(Side.BUY, 5, 100.0, fee=0.0)
    acc.reserve_order(Side.BUY, 10)
    acc.reserve_order(Side.SELL, 3)
    assert acc.worst_case_position() == 15
    assert acc.initial_margin(100.0, 0.05) == pytest.approx(0.05 * 15 * 100.0)
    acc.reduce_reservation(Side.BUY, 10, closes_order=True)
    assert acc.worst_case_position() == 5
    assert acc.open_orders == 1


# --------------------------------------------------------------------- actions


def test_action_validation():
    with pytest.raises(ValueError):
        MarketOrder(Side.BUY, 0)
    with pytest.raises(ValueError):
        LimitOrder(Side.BUY, 1, -1.0)
    with pytest.raises(ValueError):
        LimitOrder(Side.BUY, 1, 100.0, reduce_only=True)
    LimitOrder(Side.BUY, 1, 100.0, tif=TimeInForce.IOC, reduce_only=True)
