from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from marl_trading.analysis import (
    EventLog,
    EventType,
    MarketEvent,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderSide,
    build_agent_health_metrics,
    build_portfolio_health_rows,
    format_portfolio_health_breakdown,
    summarize_market_health,
)


def build_sample_log() -> EventLog:
    return EventLog(
        events=[
            MarketEvent(
                sequence=1,
                timestamp=1.0,
                event_type=EventType.SNAPSHOT,
                order_book=OrderBookSnapshot(
                    timestamp=1.0,
                    bids=(OrderBookLevel(99.0, 10.0), OrderBookLevel(98.5, 12.0)),
                    asks=(OrderBookLevel(100.0, 8.0), OrderBookLevel(100.5, 9.0)),
                ),
                payload={"active_agents": 4, "total_equity": 100_000.0, "latent_fundamental": 99.8},
            ),
            MarketEvent(
                sequence=2,
                timestamp=2.0,
                event_type=EventType.LIMIT_ORDER,
                agent_id="maker_1",
                side=OrderSide.BUY,
                order_type="limit",
                price=99.0,
                quantity=5.0,
            ),
            MarketEvent(
                sequence=3,
                timestamp=3.0,
                event_type=EventType.TRADE,
                agent_id="taker_1",
                side=OrderSide.BUY,
                price=100.0,
                quantity=2.0,
                payload={"buy_agent_id": "taker_1", "sell_agent_id": "maker_1"},
            ),
            MarketEvent(
                sequence=4,
                timestamp=4.0,
                event_type=EventType.SNAPSHOT,
                order_book=OrderBookSnapshot(
                    timestamp=4.0,
                    bids=(OrderBookLevel(100.0, 9.0), OrderBookLevel(99.5, 11.0)),
                    asks=(OrderBookLevel(101.0, 7.0), OrderBookLevel(101.5, 10.0)),
                ),
                payload={"active_agents": 3, "total_equity": 100_500.0, "latent_fundamental": 100.2},
            ),
            MarketEvent(
                sequence=5,
                timestamp=5.0,
                event_type=EventType.NEWS,
                payload={"headline": "news flash", "severity": 0.7, "impact": 0.3},
            ),
            MarketEvent(
                sequence=6,
                timestamp=6.0,
                event_type=EventType.SNAPSHOT,
                order_book=OrderBookSnapshot(
                    timestamp=6.0,
                    bids=(OrderBookLevel(99.5, 11.0), OrderBookLevel(99.0, 13.0)),
                    asks=(OrderBookLevel(100.5, 9.0), OrderBookLevel(101.0, 8.0)),
                ),
                payload={"active_agents": 3, "total_equity": 100_250.0, "latent_fundamental": 100.1},
            ),
        ]
    )


def test_summarize_market_health_from_event_log() -> None:
    summary = summarize_market_health(build_sample_log())

    assert summary.event_count == 6
    assert summary.step_count is None
    assert summary.trade_count == 1
    assert summary.news_count == 1
    assert summary.snapshot_count == 3
    assert summary.unique_agent_count == 2
    assert summary.snapshot_coverage_ratio == pytest.approx(0.5)
    assert summary.spread_availability_ratio == pytest.approx(1.0)
    assert summary.mean_spread == pytest.approx(1.0)
    assert summary.midpoint_return_volatility_bps is not None
    assert summary.midpoint_return_volatility_bps > 0.0
    assert summary.top_of_book_occupancy_ratio == pytest.approx(1.0)
    assert summary.mean_top_of_book_liquidity == pytest.approx(18.0)
    assert summary.active_agent_mean == pytest.approx(10.0 / 3.0)
    assert summary.mean_total_equity == pytest.approx(100_250.0)
    assert summary.final_total_equity == pytest.approx(100_250.0)
    assert summary.final_midpoint == pytest.approx(100.0)
    assert summary.final_fundamental == pytest.approx(100.1)


@dataclass
class _StepRecord:
    active_agents: int
    total_equity: float


@dataclass
class _RunResult:
    event_log: EventLog
    step_records: list[_StepRecord]
    final_fundamental: float


def test_summarize_market_health_from_run_result_uses_step_count() -> None:
    log = build_sample_log()
    result = _RunResult(
        event_log=log,
        step_records=[
            _StepRecord(active_agents=4, total_equity=100_000.0),
            _StepRecord(active_agents=3, total_equity=100_500.0),
            _StepRecord(active_agents=3, total_equity=100_250.0),
        ],
        final_fundamental=100.1,
    )

    summary = summarize_market_health(result)

    assert summary.step_count == 3
    assert summary.active_agent_mean == pytest.approx(10.0 / 3.0)
    assert summary.final_fundamental == pytest.approx(100.1)
    assert summary.final_total_equity == pytest.approx(100_250.0)
    assert summary.to_dict()["trade_count"] == 1


def test_build_portfolio_health_rows_uses_starting_inventory_by_agent_type() -> None:
    final_portfolios = {
        "maker_01": {
            "cash": 10_144.67,
            "inventory": 39.0,
            "equity": 14_571.41,
            "free_equity": 13_891.19,
            "status": "active",
            "deactivated_reason": None,
            "deactivated_at_ns": None,
        }
    }
    agent_configs = [
        SimpleNamespace(
            agent_id="maker_01",
            agent_type="market_maker",
            starting_cash=10_000.0,
            ruin_threshold=4_000.0,
        )
    ]

    rows = build_portfolio_health_rows(final_portfolios, agent_configs, starting_midpoint=100.0)

    assert len(rows) == 1
    row = rows[0]
    assert row.agent_id == "maker_01"
    assert row.agent_type == "market_maker"
    assert row.starting_cash == pytest.approx(10_000.0)
    assert row.starting_inventory == pytest.approx(40.0)
    assert row.starting_equity == pytest.approx(14_000.0)
    assert row.starting_free_equity == pytest.approx(14_000.0)
    assert row.ending_cash == pytest.approx(10_144.67)
    assert row.ending_inventory == pytest.approx(39.0)
    assert row.ending_equity == pytest.approx(14_571.41)
    assert row.total_pnl == pytest.approx(571.41)
    assert row.active is True


def test_build_agent_health_metrics_tracks_realized_and_unrealized_pnl() -> None:
    log = build_sample_log()
    agent_configs = [
        SimpleNamespace(agent_id="maker_1", agent_type="market_maker", starting_cash=10_000.0, ruin_threshold=4_000.0),
        SimpleNamespace(agent_id="taker_1", agent_type="noise_trader", starting_cash=10_000.0, ruin_threshold=4_000.0),
    ]

    metrics = build_agent_health_metrics(
        list(log.events),
        agent_configs,
        starting_midpoint=100.0,
        final_mark_price=100.0,
        open_orders_by_agent={"maker_1": 2},
    )

    assert metrics["maker_1"]["realized_pnl"] == pytest.approx(0.0)
    assert metrics["maker_1"]["unrealized_pnl"] == pytest.approx(0.0)
    assert metrics["maker_1"]["open_orders"] == 2
    assert metrics["taker_1"]["realized_pnl"] == pytest.approx(0.0)
    assert metrics["taker_1"]["unrealized_pnl"] == pytest.approx(0.0)
    assert metrics["taker_1"]["open_orders"] == 0


def test_build_portfolio_health_rows_respects_runtime_starting_inventory_override() -> None:
    final_portfolios = {
        "rl_01": {
            "starting_cash": 10_000.0,
            "starting_inventory": 0.0,
            "cash": 10_100.0,
            "inventory": 1.0,
            "equity": 10_200.0,
            "free_equity": 10_200.0,
            "status": "active",
        }
    }
    rows = build_portfolio_health_rows(
        final_portfolios,
        [SimpleNamespace(agent_id="rl_01", agent_type="trend_follower", starting_cash=10_000.0, ruin_threshold=4_000.0)],
        starting_midpoint=100.0,
        starting_inventory_overrides={"rl_01": 0.0},
    )

    assert rows[0].starting_inventory == pytest.approx(0.0)
    assert rows[0].starting_equity == pytest.approx(10_000.0)
    assert rows[0].total_pnl == pytest.approx(200.0)


def test_format_portfolio_health_breakdown_is_readable() -> None:
    rows = [
        build_portfolio_health_rows(
            {
                "retail_01": {
                    "cash": 10_968.88,
                    "inventory": 10.0,
                    "equity": 12_103.94,
                    "free_equity": 11_991.22,
                    "status": "active",
                }
            },
            [SimpleNamespace(agent_id="retail_01", agent_type="noise_trader", starting_cash=10_000.0, ruin_threshold=4_000.0)],
            starting_midpoint=100.0,
        )[0]
    ]

    report = format_portfolio_health_breakdown(rows)

    assert report.startswith("portfolio_breakdown:")
    assert "retail_01" in report
    assert "cash 10000.00 -> 10968.88" in report
    assert "inventory 20.00 -> 10.00" in report
    assert "open orders n/a" in report
