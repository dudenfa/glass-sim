"""Cross-margin account for a single perpetual contract.

Accounting conventions (all amounts in quote currency, e.g. USD):

- ``collateral`` is the account's cash balance: initial deposit plus realized
  PnL, minus fees, plus funding received. It is what remains when the position
  is flat.
- ``position`` is a signed integer number of contracts (long > 0, short < 0).
- ``entry_price`` is the volume-weighted average entry of the current position.
- ``equity(mark) = collateral + unrealized_pnl(mark)``.

Margin:

- the initial-margin requirement uses the *worst case* position the account
  could reach if all of its resting orders on one side were filled:
  ``IM = imr * max(|pos + open_buys|, |pos - open_sells|) * mark * contract_size``.
  An order that would push IM above equity is rejected.
- the maintenance requirement uses the current position only:
  ``MM = mmr * |pos| * mark * contract_size``. When equity falls below MM the
  account is liquidated.

With these definitions, the sum of all equities plus the insurance fund plus
collected fees is conserved exactly by trading and funding (every contract
bought is sold by someone), which the test-suite checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import Side


@dataclass
class AccountSnapshot:
    """Immutable copy of an account's state and statistics at one mark price."""

    agent_id: str
    public_id: str
    agent_type: str
    active: bool
    collateral: float
    position: int
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    equity: float
    notional: float
    leverage: float
    liquidation_price: Optional[float]
    fees_paid: float
    funding_paid: float
    initial_collateral: float
    total_pnl: float
    return_pct: float
    open_orders: int
    liquidations: int
    trade_count: int
    volume: float
    maker_quantity: int = 0
    taker_quantity: int = 0
    maker_share: float = 0.0
    orders_accepted: int = 0
    orders_rejected: int = 0
    orders_canceled: int = 0
    peak_equity: float = 0.0
    min_equity: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_abs_position: int = 0
    max_leverage: float = 0.0
    first_liquidation_time: Optional[float] = None


class MarginAccount:
    """Mutable cross-margin account; uses ``__slots__`` because every fill and valuation touches it."""

    __slots__ = (
        "agent_id",
        "public_id",
        "agent_type",
        "contract_size",
        "initial_collateral",
        "collateral",
        "position",
        "entry_price",
        "realized_pnl",
        "fees_paid",
        "funding_paid",
        "open_buy_qty",
        "open_sell_qty",
        "open_orders",
        "active",
        "liquidations",
        "trade_count",
        "volume",
        "deactivation_reason",
        "maker_quantity",
        "taker_quantity",
        "orders_accepted",
        "orders_rejected",
        "orders_canceled",
        "peak_equity",
        "min_equity",
        "max_drawdown",
        "max_drawdown_pct",
        "max_abs_position",
        "max_leverage",
        "first_liquidation_time",
    )

    def __init__(
        self,
        agent_id: str,
        collateral: float,
        *,
        public_id: Optional[str] = None,
        agent_type: str = "agent",
        contract_size: float = 1.0,
    ) -> None:
        if collateral < 0:
            raise ValueError("collateral must be non-negative")
        self.agent_id = agent_id
        self.public_id = public_id or agent_id
        self.agent_type = agent_type
        self.contract_size = float(contract_size)
        self.initial_collateral = float(collateral)
        self.collateral = float(collateral)
        self.position = 0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.open_buy_qty = 0
        self.open_sell_qty = 0
        self.open_orders = 0
        self.active = True
        self.liquidations = 0
        self.trade_count = 0
        self.volume = 0.0
        self.deactivation_reason = ""
        self.maker_quantity = 0
        self.taker_quantity = 0
        self.orders_accepted = 0
        self.orders_rejected = 0
        self.orders_canceled = 0
        self.peak_equity = float(collateral)
        self.min_equity = float(collateral)
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.max_abs_position = 0
        self.max_leverage = 0.0
        self.first_liquidation_time: Optional[float] = None

    # ----------------------------------------------------------------- values

    def unrealized_pnl(self, mark: float) -> float:
        """Position PnL at ``mark`` (0 when flat)."""
        if self.position == 0:
            return 0.0
        return self.position * (mark - self.entry_price) * self.contract_size

    def equity(self, mark: float) -> float:
        """Collateral plus unrealized PnL."""
        return self.collateral + self.unrealized_pnl(mark)

    def notional(self, mark: float) -> float:
        """Absolute position value at ``mark``."""
        return abs(self.position) * mark * self.contract_size

    def leverage(self, mark: float) -> float:
        """Notional over equity (infinite once equity is gone)."""
        equity = self.equity(mark)
        if equity <= 0:
            return float("inf") if self.position else 0.0
        return self.notional(mark) / equity

    def worst_case_position(self, extra_buy: int = 0, extra_sell: int = 0) -> int:
        """Largest absolute position reachable if all resting orders on one side filled (plus any extra order)."""
        return max(abs(self.position + self.open_buy_qty + extra_buy), abs(self.position - self.open_sell_qty - extra_sell))

    def initial_margin(self, mark: float, imr: float, extra_buy: int = 0, extra_sell: int = 0) -> float:
        """Margin required for the worst-case position."""
        return imr * self.worst_case_position(extra_buy, extra_sell) * mark * self.contract_size

    def maintenance_margin(self, mark: float, mmr: float) -> float:
        """Margin below which the current position is liquidated."""
        return mmr * abs(self.position) * mark * self.contract_size

    def available_margin(self, mark: float, imr: float) -> float:
        """Equity left after the initial-margin requirement."""
        return self.equity(mark) - self.initial_margin(mark, imr)

    def liquidation_price(self, mmr: float) -> Optional[float]:
        """Mark price at which equity equals the maintenance requirement."""
        if self.position == 0:
            return None
        cs = self.contract_size
        pos = self.position
        denominator = cs * (pos - mmr * abs(pos))
        if denominator == 0:
            return None
        price = (pos * self.entry_price * cs - self.collateral) / denominator
        if price <= 0:
            return None
        return price

    # -------------------------------------------------------------- mutations

    def apply_fill(self, side: Side, quantity: int, price: float, fee: float) -> float:
        """Apply a fill and return the realized PnL it produced (before fees)."""
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        signed = quantity * side.sign
        realized = 0.0
        pos = self.position
        if pos == 0 or (pos > 0) == (signed > 0):
            new_abs = abs(pos) + quantity
            self.entry_price = (abs(pos) * self.entry_price + quantity * price) / new_abs
            self.position = pos + signed
        else:
            closing = min(quantity, abs(pos))
            realized = closing * (price - self.entry_price) * (1 if pos > 0 else -1) * self.contract_size
            new_pos = pos + signed
            if new_pos == 0:
                self.entry_price = 0.0
            elif (new_pos > 0) != (pos > 0):
                # Flipped through zero: the remainder opens a new position at the fill price.
                self.entry_price = price
            self.position = new_pos
        self.realized_pnl += realized
        self.fees_paid += fee
        self.collateral += realized - fee
        self.trade_count += 1
        self.volume += quantity * price * self.contract_size
        return realized

    def update_path(self, mark: float) -> None:
        """Update path-risk statistics (peak, drawdown, worst equity, largest position, leverage)."""
        equity = self.equity(mark)
        if equity > self.peak_equity:
            self.peak_equity = equity
        if equity < self.min_equity:
            self.min_equity = equity
        drawdown = self.peak_equity - equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = 100.0 * drawdown / self.peak_equity if self.peak_equity > 0 else 0.0
        size = abs(self.position)
        if size > self.max_abs_position:
            self.max_abs_position = size
        if size:
            leverage = self.leverage(mark)
            if leverage > self.max_leverage and leverage != float("inf"):
                self.max_leverage = leverage

    def apply_funding(self, payment: float) -> None:
        """``payment`` > 0 means the account pays funding."""
        self.funding_paid += payment
        self.collateral -= payment

    def reserve_order(self, side: Side, quantity: int) -> None:
        """Count a resting order toward worst-case margin, so many small orders cannot over-commit the account."""
        if side is Side.BUY:
            self.open_buy_qty += quantity
        else:
            self.open_sell_qty += quantity
        self.open_orders += 1

    def reduce_reservation(self, side: Side, quantity: int, closes_order: bool) -> None:
        """Release reserved quantity after a fill or a cancel."""
        if side is Side.BUY:
            self.open_buy_qty = max(0, self.open_buy_qty - quantity)
        else:
            self.open_sell_qty = max(0, self.open_sell_qty - quantity)
        if closes_order:
            self.open_orders = max(0, self.open_orders - 1)

    def snapshot(self, mark: float, mmr: float) -> AccountSnapshot:
        """Freeze the account for results and the viewer."""
        equity = self.equity(mark)
        total_pnl = equity - self.initial_collateral
        return AccountSnapshot(
            agent_id=self.agent_id,
            public_id=self.public_id,
            agent_type=self.agent_type,
            active=self.active,
            collateral=self.collateral,
            position=self.position,
            entry_price=self.entry_price,
            mark_price=mark,
            unrealized_pnl=self.unrealized_pnl(mark),
            realized_pnl=self.realized_pnl,
            equity=equity,
            notional=self.notional(mark),
            leverage=self.leverage(mark) if self.position else 0.0,
            liquidation_price=self.liquidation_price(mmr),
            fees_paid=self.fees_paid,
            funding_paid=self.funding_paid,
            initial_collateral=self.initial_collateral,
            total_pnl=total_pnl,
            return_pct=100.0 * total_pnl / self.initial_collateral if self.initial_collateral else 0.0,
            open_orders=self.open_orders,
            liquidations=self.liquidations,
            trade_count=self.trade_count,
            volume=self.volume,
            maker_quantity=self.maker_quantity,
            taker_quantity=self.taker_quantity,
            maker_share=self.maker_quantity / (self.maker_quantity + self.taker_quantity) if (self.maker_quantity + self.taker_quantity) else 0.0,
            orders_accepted=self.orders_accepted,
            orders_rejected=self.orders_rejected,
            orders_canceled=self.orders_canceled,
            peak_equity=self.peak_equity,
            min_equity=self.min_equity,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=self.max_drawdown_pct,
            max_abs_position=self.max_abs_position,
            max_leverage=self.max_leverage,
            first_liquidation_time=self.first_liquidation_time,
        )
