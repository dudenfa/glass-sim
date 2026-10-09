"""Human-readable text reports of a run: market summary, participant types, every agent.

The functions work on plain dictionaries (``summary()``, ``group_table()``,
``agent_table()``) so they format a live ``SimulationResult`` and a run
loaded back from disk in exactly the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

Fmt = Callable[[Any], str]


def _num(digits: int = 2) -> Fmt:
    """Formatter with ``digits`` decimals and thousands separators."""
    return lambda v: "—" if v is None else f"{v:,.{digits}f}"


def _int(v: Any) -> str:
    """Integer formatter with thousands separators."""
    return "—" if v is None else f"{int(v):,}"


def _pct(digits: int = 2) -> Fmt:
    """Signed percentage formatter."""
    return lambda v: "—" if v is None else f"{v:+.{digits}f}%"


def _rate(v: Any) -> str:
    """Funding-rate formatter (percent with four decimals)."""
    return "—" if v is None else f"{v * 100:+.4f}%"


# (key, label, formatter). The same keys are used by the comparison report.
SUMMARY_SECTIONS: list[tuple[str, list[tuple[str, str, Fmt]]]] = [
    (
        "Activity",
        [
            ("trades", "Trades", _int),
            ("volume_contracts", "Volume (contracts)", _int),
            ("trades_per_minute", "Trades per minute", _num(1)),
            ("mean_open_interest", "Mean open interest", _int),
            ("max_open_interest", "Max open interest", _int),
        ],
    ),
    (
        "Liquidity",
        [
            ("two_sided_book_fraction", "Two-sided book (fraction)", _num(4)),
            ("median_spread_bps", "Median spread (bps)", _num(2)),
            ("mean_spread", "Mean spread (price)", _num(4)),
            ("mean_bid_depth", "Mean bid depth <=50 ticks", _int),
            ("mean_ask_depth", "Mean ask depth <=50 ticks", _int),
        ],
    ),
    (
        "Prices",
        [
            ("return_vol_1m_bps", "1-minute return vol (bps)", _num(2)),
            ("last_index_mean_abs_bps", "|last - index| mean (bps)", _num(2)),
            ("last_index_max_abs_bps", "|last - index| max (bps)", _num(1)),
            ("mark_index_mean_abs_bps", "|mark - index| mean (bps)", _num(2)),
            ("final_last", "Final last price", _num(3)),
            ("final_mark", "Final mark price", _num(3)),
            ("final_index", "Final index price", _num(3)),
        ],
    ),
    (
        "Risk and funding",
        [
            ("liquidations", "Liquidations", _int),
            ("liquidated_contracts", "Liquidated contracts", _int),
            ("adl_contracts", "Auto-deleveraged contracts", _int),
            ("bad_debt", "Bad debt", _num(2)),
            ("insurance_fund_final", "Insurance fund (final)", _num(2)),
            ("funding_settlements", "Funding settlements", _int),
            ("mean_funding_rate", "Mean funding rate", _rate),
            ("fees_collected", "Fees collected", _num(2)),
        ],
    ),
    (
        "Population and checks",
        [
            ("agents", "Agents", _int),
            ("active_agents_final", "Active at end", _int),
            ("orders_accepted", "Orders accepted", _int),
            ("orders_rejected", "Orders rejected", _int),
            ("orders_canceled", "Orders canceled", _int),
            ("events_processed", "Events processed", _int),
            ("conservation_error", "Conservation error", lambda v: "—" if v is None else f"{v:.2e}"),
        ],
    ),
]

# (key, header, width, formatter) for the per-agent table.
AGENT_COLUMNS: list[tuple[str, str, int, Fmt]] = [
    ("agent_id", "agent", 14, str),
    ("agent_type", "type", 18, str),
    ("position", "pos", 8, _int),
    ("equity", "equity", 12, _num(0)),
    ("total_pnl", "pnl", 10, lambda v: f"{v:+,.0f}"),
    ("return_pct", "return", 9, _pct(2)),
    ("realized_pnl", "realized", 10, lambda v: f"{v:+,.0f}"),
    ("unrealized_pnl", "unreal", 9, lambda v: f"{v:+,.0f}"),
    ("fees_paid", "fees", 8, _num(0)),
    ("funding_paid", "funding", 8, lambda v: f"{v:+,.0f}"),
    ("max_drawdown_pct", "maxDD%", 7, _num(1)),
    ("max_abs_position", "max|pos|", 8, _int),
    ("max_leverage", "maxLev", 7, _num(1)),
    ("maker_share", "maker%", 7, lambda v: f"{100 * v:.0f}"),
    ("trade_count", "fills", 7, _int),
    ("liquidations", "liqs", 5, _int),
    ("active", "status", 7, lambda v: "active" if v else "OUT"),
]

GROUP_COLUMNS: list[tuple[str, str, int, Fmt]] = [
    ("agent_type", "type", 19, str),
    ("agents", "n", 4, _int),
    ("inactive", "out", 4, _int),
    ("return_pct", "return", 9, _pct(2)),
    ("median_return_pct", "median", 9, _pct(2)),
    ("total_pnl", "pnl", 12, lambda v: f"{v:+,.0f}"),
    ("fees_paid", "fees", 10, _num(0)),
    ("funding_paid", "funding", 9, lambda v: f"{v:+,.0f}"),
    ("mean_max_drawdown_pct", "avg maxDD%", 10, _num(1)),
    ("maker_share", "maker%", 7, lambda v: f"{100 * v:.0f}"),
    ("volume", "volume $", 14, _num(0)),
    ("liquidations", "liqs", 5, _int),
]


def _table(rows: Sequence[dict[str, Any]], columns: list[tuple[str, str, int, Fmt]]) -> list[str]:
    """Fixed-width text table; the first column is left-aligned, the rest right-aligned."""
    header = "  ".join(h.rjust(w) if i else h.ljust(w) for i, (_, h, w, _) in enumerate(columns))
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for i, (key, _, width, fmt) in enumerate(columns):
            value = row.get(key)
            text = "—" if value is None else fmt(value)
            cells.append(text.rjust(width) if i else text.ljust(width))
        lines.append("  ".join(cells))
    return lines


def format_report(summary: dict[str, Any], groups: Sequence[dict[str, Any]], agents: Optional[Sequence[dict[str, Any]]] = None) -> str:
    """Render the summary sections, the per-type table and optionally the per-agent table as text."""
    lines = [
        f"Scenario {summary.get('scenario')}  seed {summary.get('seed')}  transparency {summary.get('transparency')}  "
        f"duration {summary.get('duration_seconds', 0) / 3600:.2f}h",
        "",
    ]
    for title, fields in SUMMARY_SECTIONS:
        lines.append(title)
        for key, label, fmt in fields:
            lines.append(f"  {label:30s} {fmt(summary.get(key))}")
        lines.append("")
    lines.append("By participant type")
    lines.extend("  " + line for line in _table(groups, GROUP_COLUMNS))
    if agents is not None:
        lines.append("")
        lines.append(f"Every agent ({len(agents)}), sorted by total PnL")
        lines.extend("  " + line for line in _table(agents, AGENT_COLUMNS))
    return "\n".join(lines)
