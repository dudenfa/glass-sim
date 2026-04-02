from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any, Iterable, Sequence, TYPE_CHECKING

from .events import EventLog, MarketEvent, OrderBookSnapshot
from .replay import summarize_event_log

if TYPE_CHECKING:
    from marl_trading.market.simulator import MarketRunResult


@dataclass(frozen=True)
class MarketHealthSummary:
    event_count: int
    step_count: int | None
    trade_count: int
    news_count: int
    snapshot_count: int
    unique_agent_count: int
    snapshot_coverage_ratio: float
    spread_availability_ratio: float
    mean_spread: float | None
    midpoint_return_volatility_bps: float | None
    top_of_book_occupancy_ratio: float
    mean_top_of_book_liquidity: float | None
    active_agent_mean: float | None
    mean_total_equity: float | None
    final_total_equity: float | None
    final_midpoint: float | None
    final_fundamental: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_metric(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def format_market_health_summary(
    summary: MarketHealthSummary,
    *,
    preset_name: str | None = None,
    seed: int | None = None,
    horizon: int | None = None,
) -> str:
    context_parts = []
    if preset_name:
        context_parts.append(f"preset={preset_name}")
    if seed is not None:
        context_parts.append(f"seed={seed}")
    if horizon is not None:
        context_parts.append(f"horizon={horizon}")

    header = " ".join(context_parts) if context_parts else "market-health"
    lines = [
        header,
        (
            "events={events} steps={steps} trades={trades} news={news} snapshots={snapshots} agents={agents}".format(
                events=summary.event_count,
                steps=_format_metric(summary.step_count),
                trades=summary.trade_count,
                news=summary.news_count,
                snapshots=summary.snapshot_count,
                agents=summary.unique_agent_count,
            )
        ),
        (
            "coverage={coverage} spread_availability={spread_availability} mean_spread={mean_spread} "
            "active_agents_mean={active_agents_mean} final_total_equity={final_total_equity} "
            "final_midpoint={final_midpoint} final_fundamental={final_fundamental}"
        ).format(
            coverage=_format_metric(summary.snapshot_coverage_ratio, 3),
            spread_availability=_format_metric(summary.spread_availability_ratio, 3),
            mean_spread=_format_metric(summary.mean_spread, 4),
            active_agents_mean=_format_metric(summary.active_agent_mean, 2),
            final_total_equity=_format_metric(summary.final_total_equity, 2),
            final_midpoint=_format_metric(summary.final_midpoint, 4),
            final_fundamental=_format_metric(summary.final_fundamental, 4),
        ),
    ]
    return "\n".join(lines)


def _as_event_list(source: EventLog | Sequence[MarketEvent] | Iterable[MarketEvent] | Any) -> list[MarketEvent]:
    if isinstance(source, EventLog):
        return list(source.events)
    if hasattr(source, "event_log"):
        event_log = getattr(source, "event_log")
        if isinstance(event_log, EventLog):
            return list(event_log.events)
        if hasattr(event_log, "events"):
            return list(getattr(event_log, "events"))
        return list(event_log)
    return list(source)


def _payload_numeric_series(events: Sequence[MarketEvent], *keys: str) -> list[float]:
    values: list[float] = []
    for event in events:
        payload = dict(event.payload)
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
                break
            except (TypeError, ValueError):
                continue
    return values


def _payload_last_number(events: Sequence[MarketEvent], *keys: str) -> float | None:
    for event in reversed(events):
        payload = dict(event.payload)
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _step_record_series(source: Any, *keys: str) -> list[float]:
    if not hasattr(source, "step_records"):
        return []
    try:
        records = list(getattr(source, "step_records"))
    except TypeError:
        return []
    values: list[float] = []
    for record in records:
        for key in keys:
            value = getattr(record, key, None)
            if value is None:
                continue
            try:
                values.append(float(value))
                break
            except (TypeError, ValueError):
                continue
    return values


def _midpoint_series(events: Sequence[MarketEvent]) -> list[float]:
    midpoints: list[float] = []
    for event in events:
        book: OrderBookSnapshot | None = event.order_book
        if book is None:
            continue
        midpoint = book.midpoint()
        if midpoint is not None:
            midpoints.append(float(midpoint))
    return midpoints


def _spread_series(events: Sequence[MarketEvent]) -> list[float]:
    spreads: list[float] = []
    for event in events:
        book: OrderBookSnapshot | None = event.order_book
        if book is None:
            continue
        spread = book.spread()
        if spread is not None:
            spreads.append(float(spread))
    return spreads


def _top_of_book_liquidity_series(events: Sequence[MarketEvent]) -> list[float]:
    values: list[float] = []
    for event in events:
        book = event.order_book
        if book is None or not book.bids or not book.asks:
            continue
        values.append(float(book.bids[0].quantity + book.asks[0].quantity))
    return values


def _return_volatility_bps(midpoints: Sequence[float]) -> float | None:
    returns: list[float] = []
    for previous, current in zip(midpoints[:-1], midpoints[1:]):
        if previous <= 0:
            continue
        returns.append(((current - previous) / previous) * 10_000.0)
    if len(returns) < 2:
        return None
    return float(pstdev(returns))


def summarize_market_health(
    source: EventLog | Sequence[MarketEvent] | Iterable[MarketEvent] | Any,
) -> MarketHealthSummary:
    events = _as_event_list(source)
    event_summary = summarize_event_log(events)
    source_final_fundamental = getattr(source, "final_fundamental", None)

    step_count: int | None = None
    if hasattr(source, "step_records"):
        try:
            step_count = len(getattr(source, "step_records"))
        except TypeError:
            step_count = None

    midpoints = _midpoint_series(events)
    spreads = _spread_series(events)
    top_of_book_liquidity = _top_of_book_liquidity_series(events)
    active_agent_series = _step_record_series(source, "active_agents")
    if not active_agent_series:
        active_agent_series = _payload_numeric_series(events, "active_agents")
    total_equity_series = _step_record_series(source, "total_equity")
    if not total_equity_series:
        total_equity_series = _payload_numeric_series(events, "total_equity")

    event_count = int(event_summary["event_count"])
    snapshot_count = int(event_summary["snapshot_count"])

    return MarketHealthSummary(
        event_count=event_count,
        step_count=step_count,
        trade_count=int(event_summary["trade_count"]),
        news_count=int(event_summary["news_count"]),
        snapshot_count=snapshot_count,
        unique_agent_count=int(event_summary["unique_agent_count"]),
        snapshot_coverage_ratio=(snapshot_count / event_count) if event_count else 0.0,
        spread_availability_ratio=(len(spreads) / snapshot_count) if snapshot_count else 0.0,
        mean_spread=float(fmean(spreads)) if spreads else None,
        midpoint_return_volatility_bps=_return_volatility_bps(midpoints),
        top_of_book_occupancy_ratio=(len(top_of_book_liquidity) / snapshot_count) if snapshot_count else 0.0,
        mean_top_of_book_liquidity=float(fmean(top_of_book_liquidity)) if top_of_book_liquidity else None,
        active_agent_mean=float(fmean(active_agent_series)) if active_agent_series else None,
        mean_total_equity=float(fmean(total_equity_series)) if total_equity_series else None,
        final_total_equity=(
            total_equity_series[-1]
            if total_equity_series
            else _payload_last_number(events, "total_equity")
        ),
        final_midpoint=event_summary["final_midpoint"],
        final_fundamental=float(source_final_fundamental)
        if source_final_fundamental is not None
        else _payload_last_number(events, "final_fundamental", "latent_fundamental", "fundamental", "fair_value", "reference_value", "mark_price"),
    )


__all__ = [
    "MarketHealthSummary",
    "format_market_health_summary",
    "summarize_market_health",
]
