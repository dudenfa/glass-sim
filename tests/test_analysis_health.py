from __future__ import annotations

from dataclasses import dataclass

import pytest

from marl_trading.analysis import EventLog, EventType, MarketEvent, OrderBookLevel, OrderBookSnapshot, OrderSide, summarize_market_health


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
