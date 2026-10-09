"""The outcome of a simulation run."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Optional, Union

from .account import AccountSnapshot
from .analysis import market_summary, stylized_facts
from .config import MarketConfig
from .recorder import Recorder
from .report import format_report

if TYPE_CHECKING:  # pragma: no cover
    from .market import PerpetualMarket


@dataclass
class SimulationResult:
    """Everything a run produced, with derived tables and exports."""

    config: MarketConfig
    seed: int
    end_time: float
    recorder: Recorder
    accounts: list[AccountSnapshot]
    insurance_fund: float
    fees_collected: float
    conservation_error: float
    events_processed: int
    order_event_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_market(cls, market: PerpetualMarket) -> SimulationResult:
        """Snapshot a market (finished or not)."""
        return cls(
            config=market.config,
            seed=market.seed,
            end_time=market.time,
            recorder=market.recorder,
            accounts=market.snapshots(),
            insurance_fund=market.insurance_fund,
            fees_collected=market.fees_collected,
            conservation_error=market.conservation_error(),
            events_processed=market.events_processed,
            order_event_counts=dict(market.recorder.order_event_counts),
        )

    # ------------------------------------------------------------ metrics

    @property
    def trades(self):
        """Every trade of the run."""
        return self.recorder.trades

    @property
    def liquidations(self):
        """Every liquidation event of the run."""
        return self.recorder.liquidations

    @property
    def bars(self) -> dict[str, list[Optional[float]]]:
        """Per-interval time series, by field."""
        return self.recorder.bars

    def summary(self) -> dict[str, Any]:
        """Market-quality metrics plus run-level counts; every key is documented in ``docs/metrics.md``."""
        market = market_summary(self.recorder.bars, self.config.record_interval)
        liquidations = self.recorder.liquidations
        funding = self.recorder.funding
        market.update(
            {
                "scenario": self.config.name,
                "seed": self.seed,
                "transparency": self.config.transparency.value,
                "duration_seconds": self.end_time,
                "agents": len(self.accounts),
                "active_agents_final": sum(1 for a in self.accounts if a.active),
                "liquidations": len(liquidations),
                "liquidated_contracts": sum(liq.quantity for liq in liquidations),
                "adl_contracts": sum(liq.adl_quantity for liq in liquidations),
                "bad_debt": sum(liq.bad_debt for liq in liquidations),
                "funding_settlements": len(funding),
                "mean_funding_rate": (sum(f.rate for f in funding) / len(funding)) if funding else None,
                "insurance_fund_final": self.insurance_fund,
                "fees_collected": self.fees_collected,
                "orders_accepted": self.order_event_counts.get("accepted", 0),
                "orders_rejected": self.order_event_counts.get("rejected", 0),
                "orders_canceled": self.order_event_counts.get("canceled", 0),
                "events_processed": self.events_processed,
                "conservation_error": self.conservation_error,
            }
        )
        return market

    def stylized_facts(self, horizons_seconds=(10, 60, 300)) -> dict[str, dict[str, Optional[float]]]:
        """Return statistics of the trade price at several horizons."""
        return stylized_facts(self.recorder.bars["close"], self.config.record_interval, horizons_seconds)

    def agent_table(self) -> list[dict[str, Any]]:
        """One row per account, best PnL first."""
        rows = [asdict(snapshot) for snapshot in self.accounts]
        rows.sort(key=lambda row: row["total_pnl"], reverse=True)
        return rows

    def group_table(self) -> list[dict[str, Any]]:
        """PnL, risk and activity aggregated by agent type."""
        return group_table([asdict(snapshot) for snapshot in self.accounts])

    # ------------------------------------------------------------- export

    def to_frames(self):
        """Pandas DataFrames of every recorded series (requires pandas)."""
        return self.recorder.to_frames()

    def report(self, agents: bool = False) -> str:
        """Text report: market summary, participant types and optionally every agent."""
        return format_report(self.summary(), self.group_table(), self.agent_table() if agents else None)

    def print_report(self, agents: bool = False) -> None:
        """Print ``report()``."""
        print(self.report(agents=agents))

    def save(self, directory: Union[str, Path]) -> Path:
        """Write report.txt, summary/agents/groups/stylized-facts JSON and CSV time series (if pandas is available)."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.txt").write_text(self.report(agents=True) + "\n")
        (path / "summary.json").write_text(json.dumps(self.summary(), indent=2, default=str))
        (path / "stylized_facts.json").write_text(json.dumps(self.stylized_facts(), indent=2))
        (path / "agents.json").write_text(json.dumps(self.agent_table(), indent=2, default=str))
        (path / "groups.json").write_text(json.dumps(self.group_table(), indent=2, default=str))
        try:
            frames = self.to_frames()
        except ImportError:
            return path
        for name, frame in frames.items():
            if len(frame):
                frame.to_csv(path / f"{name}.csv")
        return path


def group_table(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-agent rows (as produced by ``agent_table``) by agent type."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agents:
        by_type[row["agent_type"]].append(row)
    rows = []
    for agent_type, members in by_type.items():
        initial = sum(m["initial_collateral"] for m in members)
        pnl = sum(m["total_pnl"] for m in members)
        maker = sum(m.get("maker_quantity", 0) for m in members)
        taker = sum(m.get("taker_quantity", 0) for m in members)
        rows.append(
            {
                "agent_type": agent_type,
                "agents": len(members),
                "inactive": sum(0 if m["active"] else 1 for m in members),
                "initial_collateral": initial,
                "total_pnl": pnl,
                "return_pct": 100.0 * pnl / initial if initial else 0.0,
                "median_return_pct": float(median(m["return_pct"] for m in members)),
                "realized_pnl": sum(m["realized_pnl"] for m in members),
                "fees_paid": sum(m["fees_paid"] for m in members),
                "funding_paid": sum(m["funding_paid"] for m in members),
                "volume": sum(m["volume"] for m in members),
                "liquidations": sum(m["liquidations"] for m in members),
                "mean_max_drawdown_pct": sum(m.get("max_drawdown_pct", 0.0) for m in members) / len(members),
                "maker_share": maker / (maker + taker) if (maker + taker) else 0.0,
                "net_position": sum(m["position"] for m in members),
            }
        )
    rows.sort(key=lambda row: row["return_pct"], reverse=True)
    return rows


@dataclass
class SavedRun:
    """A run loaded back from a directory written by ``SimulationResult.save``."""

    summary: dict[str, Any]
    agents: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    source: str

    @classmethod
    def load(cls, directory: Union[str, Path]) -> SavedRun:
        """Read ``summary.json`` and ``agents.json``; groups are rebuilt so the aggregation code stays in one place."""
        path = Path(directory)
        agents = json.loads((path / "agents.json").read_text())
        return cls(
            summary=json.loads((path / "summary.json").read_text()),
            agents=agents,
            groups=group_table(agents),
            source=str(path),
        )
