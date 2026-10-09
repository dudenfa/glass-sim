from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from marl_trading.portfolio.errors import PortfolioInactiveError
from marl_trading.portfolio.models import SpotPortfolio, _as_str


@dataclass
class PortfolioManager:
    portfolios: Dict[str, SpotPortfolio] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.portfolios, dict):
            self.portfolios = {str(agent_id): portfolio for agent_id, portfolio in self.portfolios.items()}
            return

        portfolios = list(self.portfolios)
        self.portfolios = {}
        self.register_many(portfolios)

    def register(self, portfolio: SpotPortfolio) -> None:
        self.portfolios[portfolio.agent_id] = portfolio

    def register_many(self, portfolios: Iterable[SpotPortfolio]) -> None:
        for portfolio in portfolios:
            self.register(portfolio)

    def get(self, agent_id: Any) -> SpotPortfolio:
        key = _as_str(agent_id)
        try:
            return self.portfolios[key]
        except KeyError as exc:
            raise KeyError(f"Unknown portfolio for agent_id={key}.") from exc

    def reserve_order(self, *, agent_id: Any, order_id: Any, side: Any, quantity: Any, reservation_price: Any):
        return self.get(agent_id).reserve_order(
            order_id=order_id,
            side=side,
            quantity=quantity,
            reservation_price=reservation_price,
        )

    def release_order(self, *, agent_id: Any, order_id: Any):
        return self.get(agent_id).release_order(order_id)

    def apply_fill(self, *, agent_id: Any, order_id: Any, side: Any, quantity: Any, execution_price: Any) -> None:
        self.get(agent_id).apply_fill(
            order_id=order_id,
            side=side,
            quantity=quantity,
            execution_price=execution_price,
        )

    def apply_trade(self, trade: Any, *, execution_price: Any | None = None) -> None:
        buyer_agent_id = getattr(trade, "buy_agent_id", None)
        seller_agent_id = getattr(trade, "sell_agent_id", None)
        buy_order_id = getattr(trade, "buy_order_id", None)
        sell_order_id = getattr(trade, "sell_order_id", None)
        quantity = getattr(trade, "quantity", None)
        # The exchange model stores trade.price in integer ticks; callers should pass the
        # real execution_price in currency units when they have the market tick size.
        price = execution_price if execution_price is not None else getattr(trade, "price", None)
        timestamp_ns = getattr(trade, "timestamp_ns", None)
        if buyer_agent_id is None or seller_agent_id is None:
            raise TypeError("Trade record must expose buy_agent_id and sell_agent_id.")
        if buy_order_id is None or sell_order_id is None:
            raise TypeError("Trade record must expose buy_order_id and sell_order_id.")
        if price is None:
            raise TypeError("Trade record must expose a price or execution_price.")

        self.apply_fill(
            agent_id=buyer_agent_id,
            order_id=buy_order_id,
            side="buy",
            quantity=quantity,
            execution_price=price,
        )
        self.apply_fill(
            agent_id=seller_agent_id,
            order_id=sell_order_id,
            side="sell",
            quantity=quantity,
            execution_price=price,
        )

    def deactivate_ruined(self, *, mark_price: Any, timestamp_ns: int | None = None) -> list[SpotPortfolio]:
        deactivated: list[SpotPortfolio] = []
        for portfolio in self.portfolios.values():
            if portfolio.deactivate_if_ruined(mark_price=mark_price, timestamp_ns=timestamp_ns):
                deactivated.append(portfolio)
        return deactivated

    def active_portfolios(self) -> list[SpotPortfolio]:
        return [portfolio for portfolio in self.portfolios.values() if portfolio.active]

    def require_active(self, agent_id: Any) -> SpotPortfolio:
        portfolio = self.get(agent_id)
        if not portfolio.active:
            raise PortfolioInactiveError(f"Portfolio {portfolio.agent_id} is deactivated.")
        return portfolio
