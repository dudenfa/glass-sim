from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

from .events import EventLog, EventType, MarketEvent, OrderBookSnapshot
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


@dataclass(frozen=True)
class PortfolioHealthRow:
    agent_id: str
    agent_type: str
    status: str
    active: bool
    starting_cash: float
    ending_cash: float
    starting_inventory: float
    ending_inventory: float
    starting_equity: float
    ending_equity: float
    starting_free_equity: float | None
    ending_free_equity: float | None
    cash_delta: float
    inventory_delta: float
    equity_delta: float
    total_pnl: float
    realized_pnl: float | None = None
    unrealized_pnl: float | None = None
    open_orders: int | None = None
    ruin_threshold: float | None = None
    deactivated_reason: str | None = None
    deactivated_at_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PnlTracker:
    starting_cash: float
    starting_inventory: float
    starting_midpoint: float
    inventory: float
    cost_basis: float
    realized_pnl: float = 0.0


def _format_metric(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _format_delta(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    prefix = "+" if value >= 0 else "-"
    return f"{prefix}{_format_metric(abs(value), digits)}"


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


def format_portfolio_health_breakdown(rows: Sequence[PortfolioHealthRow]) -> str:
    if not rows:
        return "portfolio_breakdown: none"

    lines = ["portfolio_breakdown:"]
    for row in rows:
        status = "active" if row.active else row.status
        free_equity_text = (
            f"{_format_metric(row.starting_free_equity, 2)} -> {_format_metric(row.ending_free_equity, 2)}"
            if row.starting_free_equity is not None or row.ending_free_equity is not None
            else "n/a"
        )
        lines.append(
            (
                f"- {row.agent_id} ({row.agent_type}, {status}): "
                f"equity {_format_metric(row.starting_equity, 2)} -> {_format_metric(row.ending_equity, 2)} "
                f"({_format_delta(row.equity_delta, 2)}); "
                f"cash {_format_metric(row.starting_cash, 2)} -> {_format_metric(row.ending_cash, 2)} "
                f"({_format_delta(row.cash_delta, 2)}); "
                f"inventory {_format_metric(row.starting_inventory, 2)} -> {_format_metric(row.ending_inventory, 2)} "
                f"({_format_delta(row.inventory_delta, 2)}); "
                f"free equity {free_equity_text}; "
                f"realized {_format_metric(row.realized_pnl, 2) if row.realized_pnl is not None else 'n/a'}; "
                f"unrealized {_format_metric(row.unrealized_pnl, 2) if row.unrealized_pnl is not None else 'n/a'}; "
                f"open orders {_format_optional_int(row.open_orders)}"
            )
        )
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


def _default_starting_inventory(agent_type: str) -> float:
    inventory_map = {
        "market_maker": 40.0,
        "noise_trader": 20.0,
        "trend_follower": 16.0,
        "informed_trader": 18.0,
    }
    return float(inventory_map.get(agent_type, 10.0))


def _agent_id_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def build_portfolio_health_rows(
    final_portfolios: Mapping[str, Mapping[str, Any]],
    agent_configs: Sequence[Any],
    *,
    starting_midpoint: float,
    agent_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    starting_inventory_overrides: Mapping[str, float] | None = None,
    starting_cash_overrides: Mapping[str, float] | None = None,
) -> list[PortfolioHealthRow]:
    config_by_id = {
        _agent_id_text(getattr(agent_cfg, "agent_id", "")): agent_cfg
        for agent_cfg in agent_configs
        if getattr(agent_cfg, "agent_id", None) is not None
    }
    ordered_agent_ids: list[str] = [_agent_id_text(getattr(agent_cfg, "agent_id")) for agent_cfg in agent_configs if getattr(agent_cfg, "agent_id", None) is not None]
    for agent_id in final_portfolios:
        if agent_id not in config_by_id and agent_id not in ordered_agent_ids:
            ordered_agent_ids.append(agent_id)

    rows: list[PortfolioHealthRow] = []
    for agent_id in ordered_agent_ids:
        agent_cfg = config_by_id.get(agent_id)
        final_summary = dict(final_portfolios.get(agent_id, {}))
        extra_metrics = dict((agent_metrics or {}).get(agent_id, {}))
        agent_type = str(
            final_summary.get(
                "agent_type",
                extra_metrics.get("agent_type", getattr(agent_cfg, "agent_type", "unknown")),
            )
        )
        starting_cash = float(
            (starting_cash_overrides or {}).get(
                agent_id,
                final_summary.get(
                    "starting_cash",
                    getattr(agent_cfg, "starting_cash", final_summary.get("cash", 0.0)),
                ),
            )
        )
        starting_inventory = float(
            (starting_inventory_overrides or {}).get(
                agent_id,
                final_summary.get(
                    "starting_inventory",
                    getattr(agent_cfg, "starting_inventory", _default_starting_inventory(agent_type)),
                ),
            )
        )
        starting_equity = starting_cash + starting_inventory * float(starting_midpoint)
        starting_free_equity = starting_equity
        ending_cash = float(final_summary.get("cash", starting_cash))
        ending_inventory = float(final_summary.get("inventory", starting_inventory))
        ending_equity = float(final_summary.get("equity", ending_cash + ending_inventory * float(starting_midpoint)))
        ending_free_equity_value = final_summary.get("free_equity")
        ending_free_equity = float(ending_free_equity_value) if ending_free_equity_value is not None else None
        status = str(final_summary.get("status", "unknown"))
        active = status.lower() == "active"
        ruin_threshold = final_summary.get("ruin_threshold", extra_metrics.get("ruin_threshold"))
        rows.append(
            PortfolioHealthRow(
                agent_id=agent_id,
                agent_type=agent_type,
                status=status,
                active=active,
                starting_cash=starting_cash,
                ending_cash=ending_cash,
                starting_inventory=starting_inventory,
                ending_inventory=ending_inventory,
                starting_equity=starting_equity,
                ending_equity=ending_equity,
                starting_free_equity=starting_free_equity,
                ending_free_equity=ending_free_equity,
                cash_delta=ending_cash - starting_cash,
                inventory_delta=ending_inventory - starting_inventory,
                equity_delta=ending_equity - starting_equity,
                total_pnl=ending_equity - starting_equity,
                realized_pnl=extra_metrics.get("realized_pnl", final_summary.get("realized_pnl")),
                unrealized_pnl=extra_metrics.get("unrealized_pnl", final_summary.get("unrealized_pnl")),
                open_orders=extra_metrics.get("open_orders", final_summary.get("open_orders")),
                ruin_threshold=float(ruin_threshold) if ruin_threshold is not None else None,
                deactivated_reason=final_summary.get("deactivated_reason"),
                deactivated_at_ns=(
                    int(final_summary["deactivated_at_ns"])
                    if final_summary.get("deactivated_at_ns") is not None
                    else None
                ),
            )
        )

    rows.sort(key=lambda row: row.ending_equity, reverse=True)
    return rows


def build_agent_health_metrics(
    events: Sequence[MarketEvent],
    agent_configs: Sequence[Any],
    *,
    starting_midpoint: float,
    final_mark_price: float,
    final_portfolios: Mapping[str, Mapping[str, Any]] | None = None,
    open_orders_by_agent: Mapping[str, int] | None = None,
    starting_inventory_overrides: Mapping[str, float] | None = None,
    starting_cash_overrides: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    config_by_id = {
        _agent_id_text(getattr(agent_cfg, "agent_id", "")): agent_cfg
        for agent_cfg in agent_configs
        if getattr(agent_cfg, "agent_id", None) is not None
    }
    trackers: dict[str, _PnlTracker] = {}
    for agent_id, agent_cfg in config_by_id.items():
        starting_cash = float(
            (starting_cash_overrides or {}).get(
                agent_id,
                getattr(agent_cfg, "starting_cash", 0.0),
            )
        )
        starting_inventory = float(
            (starting_inventory_overrides or {}).get(
                agent_id,
                getattr(agent_cfg, "starting_inventory", _default_starting_inventory(getattr(agent_cfg, "agent_type", "unknown"))),
            )
        )
        trackers[agent_id] = _PnlTracker(
            starting_cash=starting_cash,
            starting_inventory=starting_inventory,
            starting_midpoint=float(starting_midpoint),
            inventory=starting_inventory,
            cost_basis=starting_inventory * float(starting_midpoint),
        )

    for event in events:
        event_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        if event_type != EventType.TRADE.value:
            continue
        payload = dict(event.payload)
        buy_agent_id = payload.get("buy_agent_id")
        sell_agent_id = payload.get("sell_agent_id")
        if buy_agent_id is None or sell_agent_id is None or event.price is None or event.quantity is None:
            continue
        price = float(event.price)
        quantity = float(event.quantity)
        buy_key = _agent_id_text(buy_agent_id)
        sell_key = _agent_id_text(sell_agent_id)

        for agent_key in (buy_key, sell_key):
            if agent_key not in trackers:
                trackers[agent_key] = _PnlTracker(
                    starting_cash=0.0,
                    starting_inventory=0.0,
                    starting_midpoint=float(starting_midpoint),
                    inventory=0.0,
                    cost_basis=0.0,
                )

        buyer = trackers[buy_key]
        seller = trackers[sell_key]
        buyer.inventory += quantity
        buyer.cost_basis += quantity * price

        average_cost = seller.cost_basis / seller.inventory if seller.inventory > 1e-12 else price
        seller.realized_pnl += quantity * (price - average_cost)
        seller.cost_basis = max(seller.cost_basis - average_cost * quantity, 0.0)
        seller.inventory = max(seller.inventory - quantity, 0.0)

    metrics: dict[str, dict[str, Any]] = {}
    for agent_id, tracker in trackers.items():
        final_summary = dict((final_portfolios or {}).get(agent_id, {}))
        portfolio_mark_price = final_summary.get("mark_price")
        if portfolio_mark_price is None and abs(tracker.inventory) > 1e-12:
            ending_cash = final_summary.get("cash")
            ending_equity = final_summary.get("equity")
            if ending_cash is not None and ending_equity is not None:
                portfolio_mark_price = (float(ending_equity) - float(ending_cash)) / float(tracker.inventory)
        agent_final_mark_price = float(portfolio_mark_price) if portfolio_mark_price is not None else float(final_mark_price)
        unrealized_pnl = tracker.inventory * agent_final_mark_price - tracker.cost_basis
        agent_cfg = config_by_id.get(agent_id)
        metrics[agent_id] = {
            "agent_type": getattr(agent_cfg, "agent_type", "unknown"),
            "realized_pnl": float(tracker.realized_pnl),
            "unrealized_pnl": float(unrealized_pnl),
            "total_pnl": float(tracker.realized_pnl + unrealized_pnl),
            "open_orders": int((open_orders_by_agent or {}).get(agent_id, 0)),
            "ruin_threshold": getattr(agent_cfg, "ruin_threshold", None),
        }
    return metrics


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
    "PortfolioHealthRow",
    "build_agent_health_metrics",
    "build_portfolio_health_rows",
    "format_market_health_summary",
    "format_portfolio_health_breakdown",
    "summarize_market_health",
]
