from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from marl_trading.core.domain import AgentId, AssetSymbol
from marl_trading.exchange.models import Side

from .errors import (
    InsufficientCashError,
    InsufficientInventoryError,
    PortfolioInactiveError,
    ReservationNotFoundError,
)


@dataclass(frozen=True)
class OrderReservation:
    order_id: str
    side: Side
    quantity: float
    price_per_unit: Optional[float]
    reserved_cash: float = 0.0
    reserved_inventory: float = 0.0
    filled_quantity: float = 0.0

    @property
    def remaining_quantity(self) -> float:
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def remaining_reserved_cash(self) -> float:
        if self.side is Side.BUY and self.price_per_unit is not None:
            return max(self.remaining_quantity * self.price_per_unit, 0.0)
        return 0.0

    @property
    def remaining_reserved_inventory(self) -> float:
        if self.side is Side.SELL:
            return max(self.remaining_quantity, 0.0)
        return 0.0


@dataclass(frozen=True)
class FillResult:
    order_id: str
    side: Side
    fill_quantity: float
    execution_price: float
    cash_delta: float
    inventory_delta: float
    released_cash: float
    released_inventory: float
    remaining_quantity: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    agent_id: AgentId
    symbol: AssetSymbol
    mark_price: float
    starting_cash: float
    cash: float
    inventory: float
    reserved_cash: float
    reserved_inventory: float
    available_cash: float
    available_inventory: float
    equity: float
    ruin_threshold: float
    deactivated: bool
    deactivation_reason: Optional[str]


@dataclass
class SpotPortfolio:
    agent_id: AgentId
    symbol: AssetSymbol
    starting_cash: float
    ruin_threshold: float
    starting_inventory: float = 0.0
    cash: float = field(init=False)
    inventory: float = field(init=False)
    reserved_cash: float = field(init=False, default=0.0)
    reserved_inventory: float = field(init=False, default=0.0)
    deactivated: bool = field(init=False, default=False)
    deactivation_reason: Optional[str] = field(init=False, default=None)
    _reservations: Dict[str, OrderReservation] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.starting_cash < 0:
            raise ValueError("starting_cash must be non-negative.")
        if self.starting_inventory < 0:
            raise ValueError("starting_inventory must be non-negative.")
        if self.ruin_threshold < 0:
            raise ValueError("ruin_threshold must be non-negative.")
        self.cash = float(self.starting_cash)
        self.inventory = float(self.starting_inventory)

    @property
    def available_cash(self) -> float:
        return max(self.cash - self.reserved_cash, 0.0)

    @property
    def available_inventory(self) -> float:
        return max(self.inventory - self.reserved_inventory, 0.0)

    def equity(self, mark_price: float) -> float:
        return float(self.cash + self.inventory * mark_price)

    def snapshot(self, mark_price: float) -> PortfolioSnapshot:
        equity = self.equity(mark_price)
        return PortfolioSnapshot(
            agent_id=self.agent_id,
            symbol=self.symbol,
            mark_price=float(mark_price),
            starting_cash=float(self.starting_cash),
            cash=float(self.cash),
            inventory=float(self.inventory),
            reserved_cash=float(self.reserved_cash),
            reserved_inventory=float(self.reserved_inventory),
            available_cash=float(self.available_cash),
            available_inventory=float(self.available_inventory),
            equity=equity,
            ruin_threshold=float(self.ruin_threshold),
            deactivated=bool(self.deactivated),
            deactivation_reason=self.deactivation_reason,
        )

    def is_ruined(self, mark_price: float) -> bool:
        return self.equity(mark_price) <= self.ruin_threshold

    def deactivate(self, reason: str) -> None:
        self.deactivated = True
        self.deactivation_reason = reason

    def deactivate_if_ruined(self, mark_price: float, reason: str = "ruin_threshold_breached") -> bool:
        if self.is_ruined(mark_price):
            self.deactivate(reason)
            return True
        return False

    def _ensure_active(self) -> None:
        if self.deactivated:
            raise PortfolioInactiveError("Portfolio is deactivated.")

    def _store_reservation(self, reservation: OrderReservation) -> OrderReservation:
        self._reservations[reservation.order_id] = reservation
        return reservation

    def reserve_buy(self, order_id: str, quantity: float, price_per_unit: float) -> OrderReservation:
        self._ensure_active()
        if quantity <= 0:
            raise ValueError("quantity must be positive.")
        if price_per_unit <= 0:
            raise ValueError("price_per_unit must be positive.")

        reserved_cash = quantity * price_per_unit
        if reserved_cash > self.available_cash + 1e-12:
            raise InsufficientCashError("Buy reservation exceeds available cash.")

        self.reserved_cash += reserved_cash
        return self._store_reservation(
            OrderReservation(
                order_id=order_id,
                side=Side.BUY,
                quantity=float(quantity),
                price_per_unit=float(price_per_unit),
                reserved_cash=float(reserved_cash),
            )
        )

    def reserve_sell(self, order_id: str, quantity: float) -> OrderReservation:
        self._ensure_active()
        if quantity <= 0:
            raise ValueError("quantity must be positive.")
        if quantity > self.available_inventory + 1e-12:
            raise InsufficientInventoryError("Sell reservation exceeds available inventory.")

        self.reserved_inventory += quantity
        return self._store_reservation(
            OrderReservation(
                order_id=order_id,
                side=Side.SELL,
                quantity=float(quantity),
                price_per_unit=None,
                reserved_inventory=float(quantity),
            )
        )

    def reserve_order(
        self,
        order_id: str,
        side: Side,
        quantity: float,
        price_per_unit: Optional[float] = None,
    ) -> OrderReservation:
        if side is Side.BUY:
            if price_per_unit is None:
                raise ValueError("Buy reservations require price_per_unit.")
            return self.reserve_buy(order_id, quantity, price_per_unit)
        return self.reserve_sell(order_id, quantity)

    def release_order(self, order_id: str) -> OrderReservation:
        reservation = self._reservations.pop(order_id, None)
        if reservation is None:
            raise ReservationNotFoundError(f"Unknown reservation: {order_id}")

        self.reserved_cash = max(self.reserved_cash - reservation.remaining_reserved_cash, 0.0)
        self.reserved_inventory = max(self.reserved_inventory - reservation.remaining_reserved_inventory, 0.0)
        return reservation

    def apply_fill(self, order_id: str, execution_price: float, fill_quantity: float) -> FillResult:
        self._ensure_active()
        if fill_quantity <= 0:
            raise ValueError("fill_quantity must be positive.")
        if execution_price <= 0:
            raise ValueError("execution_price must be positive.")

        reservation = self._reservations.get(order_id)
        if reservation is None:
            raise ReservationNotFoundError(f"Unknown reservation: {order_id}")
        if fill_quantity > reservation.remaining_quantity + 1e-12:
            raise ValueError("fill_quantity exceeds remaining reservation quantity.")

        released_cash = 0.0
        released_inventory = 0.0
        cash_delta = 0.0
        inventory_delta = 0.0

        if reservation.side is Side.BUY:
            if reservation.price_per_unit is None:
                raise ValueError("Buy reservation missing price_per_unit.")
            reserved_fill_cash = fill_quantity * reservation.price_per_unit
            actual_fill_cash = fill_quantity * execution_price
            released_cash = max(reserved_fill_cash - actual_fill_cash, 0.0)
            self.cash -= actual_fill_cash
            self.reserved_cash = max(self.reserved_cash - reserved_fill_cash, 0.0)
            cash_delta = -actual_fill_cash
            inventory_delta = fill_quantity
            self.inventory += fill_quantity
        else:
            self.cash += fill_quantity * execution_price
            self.inventory -= fill_quantity
            self.reserved_inventory = max(self.reserved_inventory - fill_quantity, 0.0)
            released_inventory = 0.0
            cash_delta = fill_quantity * execution_price
            inventory_delta = -fill_quantity

        updated_reservation = OrderReservation(
            order_id=reservation.order_id,
            side=reservation.side,
            quantity=reservation.quantity,
            price_per_unit=reservation.price_per_unit,
            reserved_cash=max(reservation.reserved_cash - (fill_quantity * reservation.price_per_unit if reservation.side is Side.BUY and reservation.price_per_unit is not None else 0.0), 0.0),
            reserved_inventory=max(reservation.reserved_inventory - (fill_quantity if reservation.side is Side.SELL else 0.0), 0.0),
            filled_quantity=reservation.filled_quantity + fill_quantity,
        )

        if updated_reservation.remaining_quantity <= 1e-12:
            self._reservations.pop(order_id, None)
        else:
            self._reservations[order_id] = updated_reservation

        return FillResult(
            order_id=order_id,
            side=reservation.side,
            fill_quantity=float(fill_quantity),
            execution_price=float(execution_price),
            cash_delta=float(cash_delta),
            inventory_delta=float(inventory_delta),
            released_cash=float(released_cash),
            released_inventory=float(released_inventory),
            remaining_quantity=float(updated_reservation.remaining_quantity),
        )

    def cancel_order(self, order_id: str) -> OrderReservation:
        return self.release_order(order_id)

