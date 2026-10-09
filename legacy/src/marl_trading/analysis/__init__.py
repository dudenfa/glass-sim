from .events import (
    EventLog,
    EventType,
    MarketEvent,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderSide,
    OrderType,
)
from .comparison import (
    AGENT_METRIC_SPECS,
    MetricComparison,
    RunComparison,
    RunSnapshot,
    SUMMARY_METRIC_SPECS,
    compare_market_runs,
    format_market_run_comparison,
    load_market_run,
)
from .health import (
    MarketHealthSummary,
    PortfolioHealthRow,
    build_agent_health_metrics,
    build_portfolio_health_rows,
    format_market_health_summary,
    format_portfolio_health_breakdown,
    summarize_market_health,
)
from .replay import ReplayAnnotation, ReplaySeries, build_replay_series, summarize_event_log


def plot_market_replay(*args, **kwargs):
    from .plotting import plot_market_replay as _plot_market_replay

    return _plot_market_replay(*args, **kwargs)

__all__ = [
    "EventLog",
    "EventType",
    "MarketEvent",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "AGENT_METRIC_SPECS",
    "MarketHealthSummary",
    "MetricComparison",
    "PortfolioHealthRow",
    "RunComparison",
    "RunSnapshot",
    "SUMMARY_METRIC_SPECS",
    "build_agent_health_metrics",
    "compare_market_runs",
    "build_portfolio_health_rows",
    "format_market_health_summary",
    "format_portfolio_health_breakdown",
    "format_market_run_comparison",
    "OrderSide",
    "OrderType",
    "ReplayAnnotation",
    "ReplaySeries",
    "build_replay_series",
    "load_market_run",
    "plot_market_replay",
    "summarize_market_health",
    "summarize_event_log",
]
