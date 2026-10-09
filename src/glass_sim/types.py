"""Core value types shared by the futures exchange, accounts, agents and recorder.

Prices inside the exchange are integer ticks and quantities are integer
contracts. Everything that leaves the exchange (views, records, results) is
converted to float prices so that users never have to think about ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class Side(str, Enum):
    """Buy or sell; ``sign`` turns it into a position direction."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for buy, -1 for sell."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        """The other side."""
        return Side.SELL if self is Side.BUY else Side.BUY


class TimeInForce(str, Enum):
    """How long a limit order may rest.

    GTC: rests until filled or canceled.
    IOC: fills what it can immediately, the rest is canceled.
    POST_ONLY: rejected if it would trade immediately (guarantees maker).
    """

    GTC = "gtc"
    IOC = "ioc"
    POST_ONLY = "post_only"


class Transparency(str, Enum):
    """What the public market feed reveals to every agent.

    ANONYMOUS: aggregated order book and trade prices/sizes only, like a
        traditional exchange.
    ATTRIBUTED: every resting order and every trade carries the public id of
        its owner (buyer and seller), like an on-chain order book.
    FULL: additionally, every account's position, entry price, leverage,
        unrealized PnL, equity and liquidation price is public, like a fully
        transparent perpetual exchange such as Hyperliquid.
    """

    ANONYMOUS = "anonymous"
    ATTRIBUTED = "attributed"
    FULL = "full"

    @property
    def level(self) -> int:
        """Ordinal of the level, for comparisons."""
        return {"anonymous": 0, "attributed": 1, "full": 2}[self.value]

    def allows(self, required: Transparency) -> bool:
        """Whether this level reveals everything ``required`` reveals."""
        return self.level >= required.level


# ---------------------------------------------------------------------------
# Actions: what an agent can ask the exchange to do.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitOrder:
    """Place a limit order at ``price`` for ``quantity`` contracts."""

    side: Side
    quantity: int
    price: float
    tif: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    tag: str = ""

    def __post_init__(self) -> None:
        if int(self.quantity) <= 0:
            raise ValueError("quantity must be a positive integer number of contracts")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.reduce_only and self.tif is not TimeInForce.IOC:
            raise ValueError("reduce_only limit orders must use TimeInForce.IOC")


@dataclass(frozen=True)
class MarketOrder:
    """Take liquidity for ``quantity`` contracts at the best available prices."""

    side: Side
    quantity: int
    reduce_only: bool = False
    tag: str = ""

    def __post_init__(self) -> None:
        if int(self.quantity) <= 0:
            raise ValueError("quantity must be a positive integer number of contracts")


@dataclass(frozen=True)
class Cancel:
    """Cancel one of the agent's own resting orders."""

    order_id: str


@dataclass(frozen=True)
class CancelAll:
    """Cancel all of the agent's resting orders (optionally only one side)."""

    side: Optional[Side] = None


Action = Union[LimitOrder, MarketOrder, Cancel, CancelAll]


# ---------------------------------------------------------------------------
# Exchange records.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    """One side of a trade, as seen by the account that traded."""

    time: float
    order_id: str
    agent_id: str
    side: Side
    price: float
    quantity: int
    fee: float
    is_maker: bool
    realized_pnl: float
    liquidation: bool = False


@dataclass(frozen=True)
class TradeRecord:
    """A public trade. ``buyer``/``seller`` are internal agent ids; views hide
    them unless the transparency level allows it."""

    trade_id: int
    time: float
    price: float
    quantity: int
    buyer: str
    seller: str
    aggressor: Side
    liquidation: bool = False
    adl: bool = False


@dataclass(frozen=True)
class OrderEvent:
    """Lifecycle event of an order: accepted, rejected, canceled, expired."""

    time: float
    kind: str
    order_id: str
    agent_id: str
    side: Side
    order_type: str
    quantity: int
    price: Optional[float]
    reason: str = ""


@dataclass(frozen=True)
class LiquidationRecord:
    """One liquidation event, as recorded for results and the viewer."""

    time: float
    agent_id: str
    side_closed: Side
    quantity: int
    mark_price: float
    avg_fill_price: Optional[float]
    adl_quantity: int
    bad_debt: float
    penalty: float
    equity_before: float
    deactivated: bool


@dataclass(frozen=True)
class FundingRecord:
    """One funding settlement."""

    time: float
    rate: float
    premium: float
    mark_price: float
    index_price: float
    total_paid_by_longs: float


@dataclass(frozen=True)
class NewsRecord:
    """A news item as applied to the index, or an exogenous shock from ``shock_index``."""

    time: float
    headline: str
    log_jump: float
    public: bool = True
