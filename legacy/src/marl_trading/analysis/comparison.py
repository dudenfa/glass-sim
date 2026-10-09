from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping
import json


SUMMARY_METRIC_SPECS: tuple[tuple[str, str, int], ...] = (
    ("event_count", "Events", 0),
    ("step_count", "Steps", 0),
    ("trade_count", "Trades", 0),
    ("news_count", "News", 0),
    ("snapshot_count", "Snapshots", 0),
    ("unique_agent_count", "Agents", 0),
    ("snapshot_coverage_ratio", "Coverage", 3),
    ("spread_availability_ratio", "Spread Availability", 3),
    ("mean_spread", "Mean Spread", 4),
    ("midpoint_return_volatility_bps", "Midpoint Return Volatility Bps", 2),
    ("top_of_book_occupancy_ratio", "Top Of Book Occupancy", 3),
    ("mean_top_of_book_liquidity", "Mean Top Of Book Liquidity", 2),
    ("active_agent_mean", "Active Agent Mean", 2),
    ("mean_total_equity", "Mean Total Equity", 2),
    ("final_total_equity", "Final Total Equity", 2),
    ("final_midpoint", "Final Midpoint", 4),
    ("final_fundamental", "Final Fundamental", 4),
)

AGENT_METRIC_SPECS: tuple[tuple[str, str, int, tuple[str, ...]], ...] = (
    ("equity", "Equity", 2, ("ending_equity", "equity")),
    ("free_equity", "Free Equity", 2, ("ending_free_equity", "free_equity")),
    ("pnl", "PnL", 2, ("total_pnl", "pnl", "equity_delta")),
    ("realized_pnl", "Realized", 2, ("realized_pnl", "realized")),
    ("unrealized_pnl", "Unrealized", 2, ("unrealized_pnl", "unrealized")),
    ("peak_equity", "Peak Equity", 2, ("peak_equity",)),
    ("max_equity_drawdown", "Max Drawdown", 2, ("max_equity_drawdown",)),
    ("max_equity_drawdown_pct", "Max Drawdown %", 3, ("max_equity_drawdown_pct",)),
    (
        "max_equity_drawdown_from_start_replay",
        "Max Equity Drawdown From Start Replay",
        2,
        ("max_equity_drawdown_from_start_replay",),
    ),
    ("min_equity_delta", "Min Equity Delta", 2, ("min_equity_delta",)),
    ("peak_total_pnl", "Peak PnL", 2, ("peak_total_pnl",)),
    ("max_pnl_drawdown", "Max PnL Drawdown", 2, ("max_pnl_drawdown",)),
    ("max_pnl_drawdown_from_start", "Max PnL Drawdown From Start", 2, ("max_pnl_drawdown_from_start",)),
    ("cash", "Cash", 2, ("ending_cash", "cash")),
    ("cash_delta", "Cash Delta", 2, ("cash_delta",)),
    ("available_cash", "Available Cash", 2, ("available_cash",)),
    ("inventory", "Inventory", 0, ("ending_inventory", "inventory")),
    ("inventory_delta", "Inventory Delta", 0, ("inventory_delta",)),
    ("max_inventory", "Max Inventory", 0, ("max_inventory",)),
    ("min_inventory", "Min Inventory", 0, ("min_inventory",)),
    ("max_abs_inventory", "Max |Inventory|", 0, ("max_abs_inventory",)),
    ("available_inventory", "Available Inventory", 0, ("available_inventory",)),
    ("open_orders", "Open Orders", 0, ("open_orders",)),
)

_SNAPSHOT_IGNORE_KEYS = {
    "preset",
    "description",
    "seed",
    "horizon",
    "report",
    "summary",
    "agents",
    "portfolios",
    "participant_cards",
    "metadata",
    "label",
    "name",
}


@dataclass(frozen=True)
class RunSnapshot:
    label: str
    preset: str | None
    seed: int | None
    horizon: int | None
    summary: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricComparison:
    key: str
    label: str
    digits: int
    left: float | None
    right: float | None
    delta: float | None
    pct_change: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentComparison:
    agent_id: str
    left: dict[str, Any] | None
    right: dict[str, Any] | None
    metrics: tuple[MetricComparison, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def left_present(self) -> bool:
        return self.left is not None

    @property
    def right_present(self) -> bool:
        return self.right is not None


@dataclass(frozen=True)
class RunComparison:
    left: RunSnapshot
    right: RunSnapshot
    summary_metrics: tuple[MetricComparison, ...]
    agent_comparisons: tuple[AgentComparison, ...]
    left_only_agents: tuple[str, ...]
    right_only_agents: tuple[str, ...]
    shared_agents: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_mapping(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Market report JSON must decode to an object.")
    return data


def _extract_numeric(source: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)

    if hasattr(summary, "to_dict"):
        maybe_summary = summary.to_dict()  # type: ignore[call-arg]
        if isinstance(maybe_summary, Mapping):
            return dict(maybe_summary)

    if is_dataclass(summary):
        return asdict(summary)

    if isinstance(summary, (str, bytes)):
        return {}

    derived: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _SNAPSHOT_IGNORE_KEYS:
            continue
        if isinstance(value, (int, float)):
            derived[key] = value
    return derived


def _extract_agent_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    for key in ("agents", "portfolio_breakdown", "portfolios", "participant_cards"):
        raw_agents = payload.get(key)
        if raw_agents is None:
            continue

        if isinstance(raw_agents, Mapping):
            extracted = {
                str(agent_id): dict(agent_payload)
                for agent_id, agent_payload in raw_agents.items()
                if isinstance(agent_payload, Mapping)
            }
            if extracted:
                return extracted
            continue

        if isinstance(raw_agents, list):
            agents: dict[str, dict[str, Any]] = {}
            for index, item in enumerate(raw_agents):
                if isinstance(item, Mapping):
                    item_mapping = dict(item)
                elif hasattr(item, "to_dict"):
                    maybe_item = item.to_dict()  # type: ignore[call-arg]
                    if not isinstance(maybe_item, Mapping):
                        continue
                    item_mapping = dict(maybe_item)
                elif is_dataclass(item):
                    item_mapping = asdict(item)
                else:
                    continue
                agent_id = str(item_mapping.get("agent_id") or item_mapping.get("id") or item_mapping.get("name") or f"agent_{index}")
                agents[agent_id] = item_mapping
            if agents:
                return agents

    return {}


def load_market_run(source: str | Path | Mapping[str, Any]) -> RunSnapshot:
    payload = _coerce_mapping(source)
    preset = payload.get("preset")
    seed = payload.get("seed")
    horizon = payload.get("horizon")
    label = str(payload.get("label") or preset or payload.get("name") or "run")

    return RunSnapshot(
        label=label,
        preset=str(preset) if preset is not None else None,
        seed=int(seed) if seed is not None else None,
        horizon=int(horizon) if horizon is not None else None,
        summary=_extract_summary(payload),
        agents=_extract_agent_map(payload),
        raw=payload,
    )


def _metric_delta(left: float | None, right: float | None) -> tuple[float | None, float | None]:
    if left is None or right is None:
        return None, None
    delta = right - left
    pct_change = None if left == 0 else (delta / abs(left)) * 100.0
    return delta, pct_change


def _build_metric_comparison(
    key: str,
    label: str,
    digits: int,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    candidates: tuple[str, ...] | None = None,
) -> MetricComparison:
    keys = candidates or (key,)
    left_value = _extract_numeric(left, keys)
    right_value = _extract_numeric(right, keys)
    delta, pct_change = _metric_delta(left_value, right_value)
    return MetricComparison(
        key=key,
        label=label,
        digits=digits,
        left=left_value,
        right=right_value,
        delta=delta,
        pct_change=pct_change,
    )


def _agent_metrics(agent: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, _, _, candidates in AGENT_METRIC_SPECS:
        value = _extract_numeric(agent, candidates)
        if value is not None:
            metrics[key] = value
    return metrics


def compare_market_runs(left: str | Path | Mapping[str, Any], right: str | Path | Mapping[str, Any]) -> RunComparison:
    left_snapshot = load_market_run(left)
    right_snapshot = load_market_run(right)

    summary_metrics = tuple(
        _build_metric_comparison(key, label, digits, left_snapshot.summary, right_snapshot.summary)
        for key, label, digits in SUMMARY_METRIC_SPECS
    )

    left_agents = left_snapshot.agents
    right_agents = right_snapshot.agents
    agent_ids = sorted(set(left_agents) | set(right_agents))
    shared_agents = tuple(agent_id for agent_id in agent_ids if agent_id in left_agents and agent_id in right_agents)
    left_only_agents = tuple(agent_id for agent_id in agent_ids if agent_id in left_agents and agent_id not in right_agents)
    right_only_agents = tuple(agent_id for agent_id in agent_ids if agent_id in right_agents and agent_id not in left_agents)

    agent_comparisons: list[AgentComparison] = []
    for agent_id in agent_ids:
        left_agent = left_agents.get(agent_id)
        right_agent = right_agents.get(agent_id)
        metric_rows: list[MetricComparison] = []
        if left_agent is not None or right_agent is not None:
            for key, label, digits, candidates in AGENT_METRIC_SPECS:
                metric_rows.append(
                    _build_metric_comparison(
                        key,
                        label,
                        digits,
                        left_agent or {},
                        right_agent or {},
                        candidates=candidates,
                    )
                )
        agent_comparisons.append(
            AgentComparison(
                agent_id=agent_id,
                left=dict(left_agent) if left_agent is not None else None,
                right=dict(right_agent) if right_agent is not None else None,
                metrics=tuple(metric_rows),
            )
        )

    return RunComparison(
        left=left_snapshot,
        right=right_snapshot,
        summary_metrics=summary_metrics,
        agent_comparisons=tuple(agent_comparisons),
        left_only_agents=left_only_agents,
        right_only_agents=right_only_agents,
        shared_agents=shared_agents,
    )


def _format_number(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a"
    if digits <= 0:
        return f"{int(round(value)):,}"
    return f"{value:,.{digits}f}"


def _format_delta(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    if digits <= 0:
        return f"{sign}{int(round(value)):,}"
    return f"{sign}{value:,.{digits}f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _run_header(snapshot: RunSnapshot) -> str:
    parts = []
    if snapshot.preset:
        parts.append(f"preset={snapshot.preset}")
    if snapshot.seed is not None:
        parts.append(f"seed={snapshot.seed}")
    if snapshot.horizon is not None:
        parts.append(f"horizon={snapshot.horizon}")
    if snapshot.label and snapshot.label not in parts:
        parts.append(f"label={snapshot.label}")
    return " ".join(parts) if parts else "run"


def format_market_run_comparison(comparison: RunComparison) -> str:
    lines = [
        f"left: {_run_header(comparison.left)}",
        f"right: {_run_header(comparison.right)}",
        "delta = right - left",
        "",
        "Summary metrics",
        "| Metric | Left | Right | Delta | % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for metric in comparison.summary_metrics:
        lines.append(
            "| {label} | {left} | {right} | {delta} | {pct} |".format(
                label=metric.label,
                left=_format_number(metric.left, metric.digits),
                right=_format_number(metric.right, metric.digits),
                delta=_format_delta(metric.delta, metric.digits),
                pct=_format_percent(metric.pct_change),
            )
        )

    lines.extend(
        [
            "",
            "Agent coverage",
            f"- shared: {len(comparison.shared_agents)}",
            f"- left only: {len(comparison.left_only_agents)}",
            f"- right only: {len(comparison.right_only_agents)}",
        ]
    )
    if comparison.left_only_agents:
        lines.append(f"- left-only ids: {', '.join(comparison.left_only_agents)}")
    if comparison.right_only_agents:
        lines.append(f"- right-only ids: {', '.join(comparison.right_only_agents)}")

    if comparison.agent_comparisons:
        lines.extend(["", "Per-agent comparisons"])
        for agent in comparison.agent_comparisons:
            left_type = str(agent.left.get("agent_type") or agent.left.get("type") or "") if agent.left else ""
            right_type = str(agent.right.get("agent_type") or agent.right.get("type") or "") if agent.right else ""
            type_bits = [bit for bit in {left_type, right_type} if bit]
            status_bits: list[str] = []
            if agent.left_present and agent.right_present:
                status_bits.append("shared")
            elif agent.left_present:
                status_bits.append("left-only")
            else:
                status_bits.append("right-only")
            if type_bits:
                status_bits.append("type=" + "/".join(sorted(type_bits)))
            lines.append("")
            lines.append(f"### {agent.agent_id} ({', '.join(status_bits)})")
            lines.append("| Metric | Left | Right | Delta | % |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for metric in agent.metrics:
                lines.append(
                    "| {label} | {left} | {right} | {delta} | {pct} |".format(
                        label=metric.label,
                        left=_format_number(metric.left, metric.digits),
                        right=_format_number(metric.right, metric.digits),
                        delta=_format_delta(metric.delta, metric.digits),
                        pct=_format_percent(metric.pct_change),
                    )
                )

    return "\n".join(lines)


__all__ = [
    "AGENT_METRIC_SPECS",
    "MetricComparison",
    "RunComparison",
    "RunSnapshot",
    "SUMMARY_METRIC_SPECS",
    "compare_market_runs",
    "format_market_run_comparison",
    "load_market_run",
]
