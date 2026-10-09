"""What an agent can see: a read-only, transparency-aware view of the market.

Each agent receives a ``MarketView`` when it wakes up. The view reads the live
market state (nothing is copied until you ask for it) and enforces the
market's ``Transparency`` level:

============================  =========  ==========  ====
information                   ANONYMOUS  ATTRIBUTED  FULL
============================  =========  ==========  ====
aggregated book, trades       yes        yes         yes
mark / index / funding / OI   yes        yes         yes
owner of each resting order   no         yes         yes
buyer and seller of trades    no         yes         yes
every account's position,     no         no          yes
entry, PnL, leverage, liq px
============================  =========  ==========  ====

Asking for information the level does not allow raises ``TransparencyError``,
so an agent written for FULL transparency fails loudly in an anonymous market
instead of silently receiving empty data. Use ``view.can_see(...)`` to write
agents that adapt to the level.

Agents always see their own account, their own orders and their own private
signals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .types import Side, Transparency

if TYPE_CHECKING:  # pragma: no cover
    from .market import PerpetualMarket
    from .processes import PrivateSignal


class TransparencyError(PermissionError):
    """Raised when an agent requests information hidden at the current transparency level."""


@dataclass(frozen=True)
class Level:
    """One aggregated price level of the book."""

    price: float
    quantity: int
    orders: int


@dataclass(frozen=True)
class PublicOrder:
    """A resting order with its owner (ATTRIBUTED and up)."""

    price: float
    quantity: int
    side: Side
    owner: str


@dataclass(frozen=True)
class PublicTrade:
    """A trade on the public tape; ``buyer`` and ``seller`` are ``None`` in an anonymous market."""

    time: float
    price: float
    quantity: int
    aggressor: Side
    buyer: Optional[str]
    seller: Optional[str]
    liquidation: bool


@dataclass(frozen=True)
class PublicAccount:
    """Another account's state as revealed under FULL transparency."""

    public_id: str
    position: int
    entry_price: float
    notional: float
    equity: float
    unrealized_pnl: float
    leverage: float
    liquidation_price: Optional[float]
    total_pnl: float
    return_pct: float
    active: bool


@dataclass(frozen=True)
class OwnOrder:
    """One of the agent's own resting orders."""

    order_id: str
    side: Side
    price: float
    remaining: int
    quantity: int
    time: float
    tag: str


@dataclass(frozen=True)
class AccountView:
    """The agent's own account, always visible."""

    public_id: str
    collateral: float
    position: int
    entry_price: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    available_margin: float
    leverage: float
    liquidation_price: Optional[float]
    open_buy_quantity: int
    open_sell_quantity: int
    initial_collateral: float
    active: bool

    @property
    def max_additional_notional(self) -> float:
        """Rough headroom: extra notional that fits in available margin at max leverage."""
        return max(0.0, self.available_margin)


@dataclass(frozen=True)
class NewsView:
    """A published news item."""

    time: float
    headline: str
    log_jump: float


class MarketView:
    """The market as seen by one agent."""

    __slots__ = ("_market", "_agent_id")

    def __init__(self, market: PerpetualMarket, agent_id: str) -> None:
        self._market = market
        self._agent_id = agent_id

    # ------------------------------------------------------------ static info

    @property
    def transparency(self) -> Transparency:
        """The market's transparency level."""
        return self._market.config.transparency

    def can_see(self, required: Transparency) -> bool:
        """Whether the market reveals what ``required`` reveals; use it to write agents that adapt instead of failing."""
        return self.transparency.allows(required)

    def _require(self, required: Transparency, what: str) -> None:
        """Raise ``TransparencyError`` unless ``required`` is allowed."""
        if not self.transparency.allows(required):
            raise TransparencyError(f"{what} requires transparency >= {required.value}; market is {self.transparency.value}")

    @property
    def time(self) -> float:
        """Simulated time in seconds."""
        return self._market.time

    @property
    def tick_size(self) -> float:
        """Price increment."""
        return self._market.tick_size

    @property
    def contract_size(self) -> float:
        """Notional per contract per unit of price."""
        return self._market.config.contract.contract_size

    @property
    def max_leverage(self) -> float:
        """Maximum leverage of the contract."""
        return self._market.config.contract.max_leverage

    @property
    def maker_fee(self) -> float:
        """Maker fee as a fraction of notional."""
        return self._market.config.contract.maker_fee

    @property
    def taker_fee(self) -> float:
        """Taker fee as a fraction of notional."""
        return self._market.config.contract.taker_fee

    @property
    def mmr(self) -> float:
        """Maintenance-margin ratio."""
        return self._market.config.contract.mmr

    # ----------------------------------------------------------- public prices

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid (``None`` when empty)."""
        return self._market.best_bid_price()

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask (``None`` when empty)."""
        return self._market.best_ask_price()

    @property
    def mid(self) -> Optional[float]:
        """Book mid (``None`` when a side is empty)."""
        return self._market.mid_price()

    @property
    def spread(self) -> Optional[float]:
        """Ask minus bid (``None`` when a side is empty)."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def last_price(self) -> float:
        """Price of the last trade."""
        return self._market.last_price

    @property
    def mark_price(self) -> float:
        """The mark price used for margin and liquidations."""
        return self._market.mark_price

    @property
    def index_price(self) -> float:
        """The external index (oracle) price."""
        return self._market.index_price

    @property
    def funding_rate(self) -> float:
        """The rate applied at the last funding settlement."""
        return self._market.last_funding_rate

    @property
    def predicted_funding_rate(self) -> float:
        """Rate the next settlement would apply at the current premium."""
        return self._market.predicted_funding_rate()

    @property
    def next_funding_time(self) -> float:
        """Simulated time of the next funding settlement."""
        return self._market.next_funding_time

    @property
    def open_interest(self) -> int:
        """Total long (= total short) contracts."""
        return self._market.open_interest()

    def price_history(self, seconds: int = 60) -> list[float]:
        """Last-trade price sampled once per record interval, most recent last."""
        return self._market.recent_prices(seconds)

    def mark_history(self, seconds: int = 60) -> list[float]:
        """Mark price sampled once per record interval, most recent last."""
        return self._market.recent_marks(seconds)

    def open_interest_history(self, seconds: int = 60) -> list[float]:
        """Open interest sampled once per record interval, most recent last."""
        return self._market.recent_open_interest(seconds)

    def index_history(self, seconds: int = 60) -> list[float]:
        """Index price sampled once per record interval, most recent last."""
        return self._market.recent_index(seconds)

    def recent_news(self, n: int = 5) -> list[NewsView]:
        """The last ``n`` published news items."""
        return [NewsView(time=item.time, headline=item.headline, log_jump=item.log_jump) for item in self._market.public_news[-n:]]

    # ------------------------------------------------------------- order book

    def book(self, depth: int = 10) -> tuple[list[Level], list[Level]]:
        """Aggregated depth ``(bids, asks)``, best price first."""
        tick = self.tick_size
        bids = [Level(price * tick, qty, count) for price, qty, count in self._market.book.levels(Side.BUY, depth)]
        asks = [Level(price * tick, qty, count) for price, qty, count in self._market.book.levels(Side.SELL, depth)]
        return bids, asks

    def depth_near_touch(self, side: Side, ticks: int = 10) -> int:
        """Resting contracts within ``ticks`` of the best price on ``side``."""
        return self._market.book.depth_within(side, ticks)

    def book_orders(self, side: Side, depth: int = 10) -> list[PublicOrder]:
        """Order-level book with owners. Requires ATTRIBUTED transparency."""
        self._require(Transparency.ATTRIBUTED, "order ownership")
        tick = self.tick_size
        public = self._market.public_id
        return [PublicOrder(price=o.price * tick, quantity=o.remaining, side=o.side, owner=public(o.agent_id)) for o in self._market.book.orders(side, depth)]

    def recent_trades(self, n: int = 50) -> list[PublicTrade]:
        """The last ``n`` public trades; counterparties are shown only from ATTRIBUTED up."""
        attributed = self.can_see(Transparency.ATTRIBUTED)
        public = self._market.public_id
        trades = list(self._market.recent_trades)[-n:]
        return [
            PublicTrade(
                time=t.time,
                price=t.price,
                quantity=t.quantity,
                aggressor=t.aggressor,
                buyer=public(t.buyer) if attributed else None,
                seller=public(t.seller) if attributed else None,
                liquidation=t.liquidation,
            )
            for t in trades
        ]

    # --------------------------------------------------------------- accounts

    def accounts(self, include_self: bool = False) -> list[PublicAccount]:
        """Every account's public state. Requires FULL transparency."""
        self._require(Transparency.FULL, "account positions")
        return self._market.public_accounts(exclude=None if include_self else self._agent_id)

    # ------------------------------------------------------------ own private

    @property
    def me(self) -> AccountView:
        """The agent's own account."""
        return self._market.account_view(self._agent_id)

    @property
    def position(self) -> int:
        """The agent's signed position in contracts."""
        return self._market.accounts[self._agent_id].position

    def my_orders(self) -> list[OwnOrder]:
        """The agent's resting orders."""
        tick = self.tick_size
        return [
            OwnOrder(order_id=o.order_id, side=o.side, price=o.price * tick, remaining=o.remaining, quantity=o.quantity, time=o.time, tag=o.tag)
            for o in self._market.book.orders_of(self._agent_id)
        ]

    def signals(self) -> Sequence[PrivateSignal]:
        """Private signals received and not yet public (empty unless subscribed)."""
        return self._market.pending_signals(self._agent_id)

    # --------------------------------------------------------------- helpers

    def round_price(self, price: float) -> float:
        """Nearest valid price on the tick grid."""
        tick = self.tick_size
        return max(tick, round(price / tick) * tick)

    def max_position(self, price: Optional[float] = None, leverage: Optional[float] = None) -> int:
        """Largest absolute position (contracts) the current equity supports at ``leverage``."""
        me = self.me
        price = price or self.mark_price
        lev = min(leverage or self.max_leverage, self.max_leverage)
        if price <= 0 or me.equity <= 0:
            return 0
        return int(me.equity * lev / (price * self.contract_size))
