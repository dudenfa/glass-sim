from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from marl_trading.portfolio.errors import (
    InsufficientCashError,
    InsufficientInventoryError,
    PortfolioInactiveError,
    ReservationNotFoundError,
)


def _as_str(value: Any) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def _as_float(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric.") from exc
    if numeric < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return numeric


class PortfolioStatus(str, Enum):
    ACTIVE = "active"
    DEACTIVATED = "deactivated"


@dataclass
class OrderReservation:
    order_id: str
    side: str
    reservation_price: float
    remaining_quantity: float
    reserved_cash: float = 0.0
    reserved_inventory: float = 0.0

    def __post_init__(self) -> None:
        self.order_id = _as_str(self.order_id)
        self.side = _as_str(self.side).lower()
        self.reservation_price = float(self.reservation_price)
        self.remaining_quantity = _as_float(self.remaining_quantity, "remaining_quantity")
        self.reserved_cash = float(self.reserved_cash)
        self.reserved_inventory = float(self.reserved_inventory)
        if self.side not in {"buy", "sell"}:
            raise ValueError("Reservation side must be 'buy' or 'sell'.")
        if self.remaining_quantity <= 0.0:
            raise ValueError("remaining_quantity must be positive.")
        if self.reserved_cash < 0.0 or self.reserved_inventory < 0.0:
            raise ValueError("Reserved amounts must be non-negative.")

    @property
    def is_fully_funded(self) -> bool:
        return self.reserved_cash > 0.0 or self.reserved_inventory > 0.0


@dataclass(frozen=True)
class FillResult:
    order_id: str
    side: str
    fill_quantity: float
    execution_price: float
    cash_delta: float
    inventory_delta: float
    released_cash: float
    released_inventory: float
    remaining_quantity: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    agent_id: str
    symbol: str
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
    agent_id: str
    symbol: str
    starting_cash: float
    ruin_threshold: float
    starting_inventory: float = 0.0
    cash: float = field(init=False)
    inventory: float = field(init=False)
    reserved_cash: float = field(init=False, default=0.0)
    reserved_inventory: float = field(init=False, default=0.0)
    status: PortfolioStatus = field(init=False, default=PortfolioStatus.ACTIVE)
    deactivated_reason: str | None = field(init=False, default=None)
    deactivated_at_ns: int | None = field(init=False, default=None)
    reservations: Dict[str, OrderReservation] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_id = _as_str(self.agent_id)
        self.symbol = _as_str(self.symbol)
        self.starting_cash = _as_float(self.starting_cash, "starting_cash")
        self.starting_inventory = float(self.starting_inventory)
        self.ruin_threshold = _as_float(self.ruin_threshold, "ruin_threshold")
        self.cash = float(self.starting_cash)
        self.inventory = float(self.starting_inventory)
        if self.cash < 0.0:
            raise ValueError("starting_cash must be non-negative.")

    @property
    def active(self) -> bool:
        return self.status is PortfolioStatus.ACTIVE

    @property
    def deactivated(self) -> bool:
        return self.status is PortfolioStatus.DEACTIVATED

    @property
    def available_cash(self) -> float:
        return self.cash - self.reserved_cash

    @property
    def available_inventory(self) -> float:
        return self.inventory - self.reserved_inventory

    def equity(self, mark_price: float) -> float:
        price = float(mark_price)
        return self.cash + self.inventory * price

    def free_equity(self, mark_price: float) -> float:
        price = float(mark_price)
        return self.available_cash + self.available_inventory * price

    def can_reserve(self, *, side: Any, quantity: Any, reservation_price: Any) -> bool:
        side_value = _as_str(side).lower()
        amount = _as_float(quantity, "quantity")
        price = _as_float(reservation_price, "reservation_price")
        if side_value == "buy":
            return self.available_cash + 1e-12 >= amount * price
        if side_value == "sell":
            return self.available_inventory + 1e-12 >= amount
        raise ValueError("side must be buy or sell.")

    def reserve_order(
        self,
        order_id: Any,
        side: Any,
        quantity: Any,
        reservation_price: Any | None = None,
        *,
        price_per_unit: Any | None = None,
    ) -> OrderReservation:
        if not self.active:
            raise PortfolioInactiveError(f"Portfolio {self.agent_id} is deactivated.")

        order_key = _as_str(order_id)
        side_value = _as_str(side).lower()
        amount = _as_float(quantity, "quantity")
        if reservation_price is None:
            reservation_price = price_per_unit
        if reservation_price is None:
            raise ValueError("reservation_price or price_per_unit must be provided.")
        price = _as_float(reservation_price, "reservation_price")
        if order_key in self.reservations:
            raise ValueError(f"Duplicate reservation for order_id={order_key}.")
        if side_value not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell.")

        if side_value == "buy":
            required_cash = amount * price
            if self.available_cash + 1e-12 < required_cash:
                raise InsufficientCashError(
                    f"Agent {self.agent_id} requires {required_cash:.6f} cash but only has {self.available_cash:.6f} available."
                )
            reservation = OrderReservation(
                order_id=order_key,
                side=side_value,
                reservation_price=price,
                remaining_quantity=amount,
                reserved_cash=required_cash,
                reserved_inventory=0.0,
            )
            self.reserved_cash += required_cash
        else:
            if self.available_inventory + 1e-12 < amount:
                raise InsufficientInventoryError(
                    f"Agent {self.agent_id} requires {amount:.6f} inventory but only has {self.available_inventory:.6f} available."
                )
            reservation = OrderReservation(
                order_id=order_key,
                side=side_value,
                reservation_price=price,
                remaining_quantity=amount,
                reserved_cash=0.0,
                reserved_inventory=amount,
            )
            self.reserved_inventory += amount

        self.reservations[order_key] = reservation
        return reservation

    def reserve_buy(self, order_id: Any, quantity: Any, price_per_unit: Any) -> OrderReservation:
        return self.reserve_order(order_id, "buy", quantity, reservation_price=price_per_unit)

    def reserve_sell(self, order_id: Any, quantity: Any) -> OrderReservation:
        return self.reserve_order(order_id, "sell", quantity, reservation_price=0.0)

    def release_order(self, order_id: Any) -> OrderReservation:
        order_key = _as_str(order_id)
        reservation = self.reservations.pop(order_key, None)
        if reservation is None:
            raise ReservationNotFoundError(f"Unknown reservation for order_id={order_key}.")

        self.reserved_cash -= reservation.reserved_cash
        self.reserved_inventory -= reservation.reserved_inventory
        return reservation

    def cancel_order(self, order_id: Any) -> OrderReservation:
        return self.release_order(order_id)

    def apply_fill(
        self,
        order_id: Any,
        execution_price: Any | None = None,
        fill_quantity: Any | None = None,
        *,
        side: Any | None = None,
        quantity: Any | None = None,
    ) -> FillResult:
        if not self.active:
            raise PortfolioInactiveError(f"Portfolio {self.agent_id} is deactivated.")

        order_key = _as_str(order_id)
        if fill_quantity is None:
            fill_quantity = quantity
        if fill_quantity is None:
            raise ValueError("fill quantity must be provided.")
        if execution_price is None:
            raise ValueError("execution_price must be provided.")

        amount = _as_float(fill_quantity, "quantity")
        price = _as_float(execution_price, "execution_price")
        if amount <= 0.0:
            raise ValueError("quantity must be positive.")

        reservation = self.reservations.get(order_key)
        if reservation is None:
            raise ReservationNotFoundError(f"Unknown reservation for order_id={order_key}.")
        side_value = _as_str(side).lower() if side is not None else reservation.side
        if reservation.side != side_value:
            raise ValueError("Reservation side does not match fill side.")
        if amount > reservation.remaining_quantity + 1e-12:
            raise ValueError("Fill quantity exceeds reserved quantity.")

        released_cash = 0.0
        released_inventory = 0.0
        if side_value == "buy":
            reserved_release = amount * reservation.reservation_price
            actual_cost = amount * price
            self.reserved_cash -= reserved_release
            self.cash -= actual_cost
            self.inventory += amount
            reservation.reserved_cash -= reserved_release
            released_cash = max(reserved_release - actual_cost, 0.0)
        else:
            self.reserved_inventory -= amount
            self.cash += amount * price
            self.inventory -= amount
            reservation.reserved_inventory -= amount
            released_inventory = 0.0

        reservation.remaining_quantity -= amount
        if reservation.remaining_quantity <= 1e-12:
            self.reservations.pop(order_key, None)
        else:
            reservation.remaining_quantity = max(reservation.remaining_quantity, 0.0)
            if side_value == "buy":
                reservation.reserved_cash = max(reservation.reserved_cash, 0.0)
            else:
                reservation.reserved_inventory = max(reservation.reserved_inventory, 0.0)

        cash_delta = -amount * price if side_value == "buy" else amount * price
        inventory_delta = amount if side_value == "buy" else -amount
        return FillResult(
            order_id=order_key,
            side=side_value,
            fill_quantity=amount,
            execution_price=price,
            cash_delta=cash_delta,
            inventory_delta=inventory_delta,
            released_cash=released_cash,
            released_inventory=released_inventory,
            remaining_quantity=reservation.remaining_quantity if order_key in self.reservations else 0.0,
        )

    def deactivate(self, reason: str, timestamp_ns: int | None = None) -> None:
        self.status = PortfolioStatus.DEACTIVATED
        self.deactivated_reason = reason
        self.deactivated_at_ns = timestamp_ns

    def deactivate_if_ruined(self, mark_price: Any, timestamp_ns: int | None = None) -> bool:
        if self.equity(float(mark_price)) < self.ruin_threshold:
            self.cancel_all_orders()
            self.deactivate("ruin_threshold_breached", timestamp_ns=timestamp_ns)
            return True
        return False

    def snapshot(self, mark_price: Any) -> PortfolioSnapshot:
        price = float(mark_price)
        return PortfolioSnapshot(
            agent_id=self.agent_id,
            symbol=self.symbol,
            mark_price=price,
            starting_cash=self.starting_cash,
            cash=self.cash,
            inventory=self.inventory,
            reserved_cash=self.reserved_cash,
            reserved_inventory=self.reserved_inventory,
            available_cash=self.available_cash,
            available_inventory=self.available_inventory,
            equity=self.equity(price),
            ruin_threshold=self.ruin_threshold,
            deactivated=self.deactivated,
            deactivation_reason=self.deactivated_reason,
        )

    def cancel_all_orders(self) -> list[OrderReservation]:
        canceled: list[OrderReservation] = []
        for order_id in list(self.reservations):
            canceled.append(self.release_order(order_id))
        return canceled

    def summary(self, mark_price: float) -> dict[str, float | str | None]:
        return {
            "agent_id": self.agent_id,
            "symbol": self.symbol,
            "starting_cash": self.starting_cash,
            "starting_inventory": self.starting_inventory,
            "cash": self.cash,
            "inventory": self.inventory,
            "reserved_cash": self.reserved_cash,
            "reserved_inventory": self.reserved_inventory,
            "available_cash": self.available_cash,
            "available_inventory": self.available_inventory,
            "mark_price": float(mark_price),
            "equity": self.equity(mark_price),
            "free_equity": self.free_equity(mark_price),
            "status": self.status.value,
            "deactivated_reason": self.deactivated_reason,
            "deactivated_at_ns": self.deactivated_at_ns,
        }
