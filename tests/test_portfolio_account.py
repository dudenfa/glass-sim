from __future__ import annotations

import pytest

from marl_trading.core.domain import AgentId, AssetSymbol
from marl_trading.exchange.models import Side
from marl_trading.portfolio import (
    FillResult,
    InsufficientCashError,
    InsufficientInventoryError,
    PortfolioInactiveError,
    ReservationNotFoundError,
    SpotPortfolio,
)


@pytest.fixture()
def portfolio() -> SpotPortfolio:
    return SpotPortfolio(
        agent_id=AgentId("agent-1"),
        symbol=AssetSymbol("BTC"),
        starting_cash=10_000.0,
        starting_inventory=0.0,
        ruin_threshold=4_000.0,
    )


def test_buy_reservation_tracks_reserved_cash(portfolio: SpotPortfolio) -> None:
    reservation = portfolio.reserve_buy("order-1", quantity=4.0, price_per_unit=1_000.0)

    assert reservation.side is Side.BUY
    assert portfolio.reserved_cash == pytest.approx(4_000.0)
    assert portfolio.available_cash == pytest.approx(6_000.0)


def test_sell_reservation_tracks_reserved_inventory(portfolio: SpotPortfolio) -> None:
    portfolio.inventory = 5.0
    reservation = portfolio.reserve_sell("order-2", quantity=3.0)

    assert reservation.side is Side.SELL
    assert portfolio.reserved_inventory == pytest.approx(3.0)
    assert portfolio.available_inventory == pytest.approx(2.0)


def test_buy_fill_releases_unused_cash_and_updates_inventory(portfolio: SpotPortfolio) -> None:
    portfolio.reserve_buy("order-3", quantity=4.0, price_per_unit=1_000.0)

    result = portfolio.apply_fill("order-3", execution_price=900.0, fill_quantity=2.0)

    assert isinstance(result, FillResult)
    assert portfolio.cash == pytest.approx(8_200.0)
    assert portfolio.inventory == pytest.approx(2.0)
    assert portfolio.reserved_cash == pytest.approx(2_000.0)
    assert result.released_cash == pytest.approx(200.0)
    assert result.remaining_quantity == pytest.approx(2.0)


def test_sell_fill_updates_cash_and_inventory(portfolio: SpotPortfolio) -> None:
    portfolio.inventory = 5.0
    portfolio.reserve_sell("order-4", quantity=3.0)

    result = portfolio.apply_fill("order-4", execution_price=1_100.0, fill_quantity=2.0)

    assert portfolio.cash == pytest.approx(10_400.0)
    assert portfolio.inventory == pytest.approx(3.0)
    assert portfolio.reserved_inventory == pytest.approx(1.0)
    assert result.inventory_delta == pytest.approx(-2.0)


def test_cancel_releases_remaining_reservation(portfolio: SpotPortfolio) -> None:
    portfolio.reserve_buy("order-5", quantity=2.0, price_per_unit=1_500.0)

    reservation = portfolio.cancel_order("order-5")

    assert reservation.order_id == "order-5"
    assert portfolio.reserved_cash == pytest.approx(0.0)
    assert portfolio.available_cash == pytest.approx(10_000.0)


def test_rejects_overcommitted_buy(portfolio: SpotPortfolio) -> None:
    with pytest.raises(InsufficientCashError):
        portfolio.reserve_buy("order-6", quantity=11.0, price_per_unit=1_000.0)


def test_rejects_overcommitted_sell(portfolio: SpotPortfolio) -> None:
    with pytest.raises(InsufficientInventoryError):
        portfolio.reserve_sell("order-7", quantity=1.0)


def test_ruin_deactivates_portfolio(portfolio: SpotPortfolio) -> None:
    ruin_portfolio = SpotPortfolio(
        agent_id=portfolio.agent_id,
        symbol=portfolio.symbol,
        starting_cash=10_000.0,
        starting_inventory=0.0,
        ruin_threshold=4_000.0,
    )
    ruin_portfolio.reserve_buy("ruin-order", quantity=10.0, price_per_unit=1_000.0)
    ruin_portfolio.apply_fill("ruin-order", execution_price=1_000.0, fill_quantity=10.0)

    assert ruin_portfolio.deactivate_if_ruined(300.0)
    assert ruin_portfolio.deactivated is True
    assert ruin_portfolio.deactivation_reason == "ruin_threshold_breached"


def test_deactivated_portfolio_rejects_new_reservations(portfolio: SpotPortfolio) -> None:
    portfolio.deactivate("manual_shutdown")

    with pytest.raises(PortfolioInactiveError):
        portfolio.reserve_buy("order-8", quantity=1.0, price_per_unit=100.0)


def test_missing_reservation_raises(portfolio: SpotPortfolio) -> None:
    with pytest.raises(ReservationNotFoundError):
        portfolio.apply_fill("missing", execution_price=1_000.0, fill_quantity=1.0)
