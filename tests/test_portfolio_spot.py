from __future__ import annotations

import pytest

from marl_trading.exchange.models import Side, Trade
from marl_trading.portfolio import (
    InsufficientCashError,
    InsufficientInventoryError,
    PortfolioInactiveError,
    PortfolioManager,
    PortfolioStatus,
    SpotPortfolio,
)


def test_buy_reservation_fill_and_release() -> None:
    portfolio = SpotPortfolio(
        agent_id="buyer",
        symbol="XYZ",
        starting_cash=10_000.0,
        ruin_threshold=1_000.0,
    )

    reservation = portfolio.reserve_order(
        order_id="buy-1",
        side=Side.BUY,
        quantity=4,
        reservation_price=1_000,
    )

    assert reservation.reserved_cash == pytest.approx(4_000.0)
    assert portfolio.available_cash == pytest.approx(6_000.0)

    portfolio.apply_fill(
        order_id="buy-1",
        side=Side.BUY,
        quantity=1,
        execution_price=990,
    )

    assert portfolio.cash == pytest.approx(9_010.0)
    assert portfolio.inventory == pytest.approx(1.0)
    assert portfolio.reserved_cash == pytest.approx(3_000.0)
    assert portfolio.available_cash == pytest.approx(6_010.0)

    portfolio.release_order("buy-1")

    assert portfolio.reserved_cash == pytest.approx(0.0)
    assert portfolio.available_cash == pytest.approx(9_010.0)
    assert portfolio.equity(995) == pytest.approx(10_005.0)


def test_sell_reservation_fill_and_cancel() -> None:
    portfolio = SpotPortfolio(
        agent_id="seller",
        symbol="XYZ",
        starting_cash=500.0,
        starting_inventory=10.0,
        ruin_threshold=100.0,
    )

    reservation = portfolio.reserve_order(
        order_id="sell-1",
        side=Side.SELL,
        quantity=3,
        reservation_price=1_000,
    )

    assert reservation.reserved_inventory == pytest.approx(3.0)
    assert portfolio.available_inventory == pytest.approx(7.0)

    portfolio.apply_fill(
        order_id="sell-1",
        side=Side.SELL,
        quantity=2,
        execution_price=1_010,
    )

    assert portfolio.cash == pytest.approx(2_520.0)
    assert portfolio.inventory == pytest.approx(8.0)
    assert portfolio.reserved_inventory == pytest.approx(1.0)

    portfolio.release_order("sell-1")

    assert portfolio.reserved_inventory == pytest.approx(0.0)
    assert portfolio.available_inventory == pytest.approx(8.0)
    assert portfolio.equity(1_000) == pytest.approx(10_520.0)


def test_manager_applies_exchange_trade_to_both_sides() -> None:
    buyer = SpotPortfolio(
        agent_id="buyer",
        symbol="XYZ",
        starting_cash=10_000.0,
        ruin_threshold=1_000.0,
    )
    seller = SpotPortfolio(
        agent_id="seller",
        symbol="XYZ",
        starting_cash=0.0,
        starting_inventory=5.0,
        ruin_threshold=100.0,
    )

    manager = PortfolioManager()
    manager.register_many([buyer, seller])

    manager.reserve_order(
        agent_id="buyer",
        order_id="buy-1",
        side=Side.BUY,
        quantity=2,
        reservation_price=101,
    )
    manager.reserve_order(
        agent_id="seller",
        order_id="sell-1",
        side=Side.SELL,
        quantity=2,
        reservation_price=99,
    )

    trade = Trade(
        trade_id="t1",
        timestamp=1,
        price=100,
        quantity=2,
        buy_order_id="buy-1",
        sell_order_id="sell-1",
        buy_agent_id="buyer",
        sell_agent_id="seller",
        taker_order_id="buy-1",
        maker_order_id="sell-1",
        aggressor_side=Side.BUY,
    )
    manager.apply_trade(trade)

    assert buyer.cash == pytest.approx(9_800.0)
    assert buyer.inventory == pytest.approx(2.0)
    assert buyer.reserved_cash == pytest.approx(0.0)

    assert seller.cash == pytest.approx(200.0)
    assert seller.inventory == pytest.approx(3.0)
    assert seller.reserved_inventory == pytest.approx(0.0)


def test_manager_applies_tick_priced_trade_using_real_execution_price() -> None:
    buyer = SpotPortfolio(
        agent_id="buyer",
        symbol="XYZ",
        starting_cash=10_000.0,
        ruin_threshold=1_000.0,
    )
    seller = SpotPortfolio(
        agent_id="seller",
        symbol="XYZ",
        starting_cash=0.0,
        starting_inventory=5.0,
        ruin_threshold=100.0,
    )

    manager = PortfolioManager()
    manager.register_many([buyer, seller])

    manager.reserve_order(
        agent_id="buyer",
        order_id="buy-2",
        side=Side.BUY,
        quantity=1,
        reservation_price=100.00,
    )
    manager.reserve_order(
        agent_id="seller",
        order_id="sell-2",
        side=Side.SELL,
        quantity=1,
        reservation_price=99.00,
    )

    trade = Trade(
        trade_id="t2",
        timestamp=2,
        price=9_999,
        quantity=1,
        buy_order_id="buy-2",
        sell_order_id="sell-2",
        buy_agent_id="buyer",
        sell_agent_id="seller",
        taker_order_id="buy-2",
        maker_order_id="sell-2",
        aggressor_side=Side.BUY,
    )
    manager.apply_trade(trade, execution_price=99.99)

    assert buyer.cash == pytest.approx(9_900.01)
    assert seller.cash == pytest.approx(99.99)
    assert buyer.inventory == pytest.approx(1.0)
    assert seller.inventory == pytest.approx(4.0)


def test_ruin_cancels_orders_and_deactivates_portfolio() -> None:
    portfolio = SpotPortfolio(
        agent_id="ruined",
        symbol="XYZ",
        starting_cash=1_000.0,
        ruin_threshold=1_500.0,
    )
    portfolio.reserve_order(
        order_id="buy-1",
        side=Side.BUY,
        quantity=3,
        reservation_price=100,
    )

    ruined = portfolio.deactivate_if_ruined(mark_price=100.0, timestamp_ns=42)

    assert ruined is True
    assert portfolio.status is PortfolioStatus.DEACTIVATED
    assert portfolio.reserved_cash == pytest.approx(0.0)
    assert portfolio.reservations == {}
    assert portfolio.deactivated_reason == "ruin_threshold_breached"
    assert portfolio.deactivated_at_ns == 42

    with pytest.raises(PortfolioInactiveError):
        portfolio.reserve_order(
            order_id="buy-2",
            side=Side.BUY,
            quantity=1,
            reservation_price=100,
        )


def test_portfolio_rejects_overcommitment() -> None:
    buy_portfolio = SpotPortfolio(
        agent_id="buyer",
        symbol="XYZ",
        starting_cash=100.0,
        ruin_threshold=10.0,
    )
    sell_portfolio = SpotPortfolio(
        agent_id="seller",
        symbol="XYZ",
        starting_cash=0.0,
        starting_inventory=1.0,
        ruin_threshold=10.0,
    )

    with pytest.raises(InsufficientCashError):
        buy_portfolio.reserve_order(
            order_id="buy-1",
            side=Side.BUY,
            quantity=2,
            reservation_price=60,
        )

    with pytest.raises(InsufficientInventoryError):
        sell_portfolio.reserve_order(
            order_id="sell-1",
            side=Side.SELL,
            quantity=2,
            reservation_price=60,
        )
