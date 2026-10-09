"""Stores everything a simulation produces, in plain Python lists.

Time series are sampled every ``record_interval`` seconds ("bars"). Trades,
liquidations, funding settlements and news are stored as full event lists.
``to_frames()`` converts everything to pandas DataFrames when pandas is
installed.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

from .types import FundingRecord, LiquidationRecord, NewsRecord, OrderEvent, TradeRecord

BAR_FIELDS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "mark",
    "index",
    "mid",
    "best_bid",
    "best_ask",
    "spread",
    "bid_depth",
    "ask_depth",
    "open_interest",
    "predicted_funding",
    "insurance_fund",
    "active_agents",
    "liquidations",
)


class Recorder:
    """Append-only store of bars, events and account samples, kept as plain lists so the hot loop never touches pandas."""

    def __init__(self, keep_order_events: bool = False, recent_orders: int = 500) -> None:
        self.bars: dict[str, list[Optional[float]]] = {name: [] for name in BAR_FIELDS}
        self.account_times: list[float] = []
        self.account_equity: dict[str, list[float]] = {}
        self.account_position: dict[str, list[int]] = {}
        self.trades: list[TradeRecord] = []
        self.liquidations: list[LiquidationRecord] = []
        self.funding: list[FundingRecord] = []
        self.news: list[NewsRecord] = []
        self.order_event_counts: Counter = Counter()
        self.keep_order_events = keep_order_events
        self.order_events: list[OrderEvent] = []
        self.recent_orders: deque[OrderEvent] = deque(maxlen=recent_orders)

    def add_bar(self, **values: Optional[float]) -> None:
        """Append one bar; missing fields are stored as ``None``."""
        for name in BAR_FIELDS:
            self.bars[name].append(values.get(name))

    def add_order_event(self, event: OrderEvent) -> None:
        """Count the event and keep a recent window; the full history is kept only on request because it dominates memory."""
        self.order_event_counts[event.kind] += 1
        self.recent_orders.append(event)
        if self.keep_order_events:
            self.order_events.append(event)

    def add_account_sample(self, time: float, equities: dict[str, float], positions: dict[str, int]) -> None:
        """Sample every account's equity and position at ``time``."""
        self.account_times.append(time)
        for agent_id, equity in equities.items():
            self.account_equity.setdefault(agent_id, []).append(equity)
            self.account_position.setdefault(agent_id, []).append(positions[agent_id])

    @property
    def bar_count(self) -> int:
        """Number of bars recorded so far."""
        return len(self.bars["time"])

    def to_frames(self):
        """Return a dict of pandas DataFrames: bars, trades, liquidations, funding, news, equity, positions."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_frames() requires pandas: pip install pandas") from exc
        from dataclasses import asdict

        def records(items):
            rows = [asdict(item) for item in items]
            for row in rows:
                for key, value in row.items():
                    if hasattr(value, "value"):
                        row[key] = value.value
            return pd.DataFrame(rows)

        equity = pd.DataFrame(self.account_equity, index=pd.Index(self.account_times, name="time"))
        positions = pd.DataFrame(self.account_position, index=pd.Index(self.account_times, name="time"))
        return {
            "bars": pd.DataFrame(self.bars).set_index("time"),
            "trades": records(self.trades),
            "liquidations": records(self.liquidations),
            "funding": records(self.funding),
            "news": records(self.news),
            "equity": equity,
            "positions": positions,
            "orders": records(self.order_events) if self.keep_order_events else pd.DataFrame(),
        }
