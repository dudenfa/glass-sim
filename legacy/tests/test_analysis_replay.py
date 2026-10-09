from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.analysis import (
    EventLog,
    EventType,
    MarketEvent,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderSide,
    build_replay_series,
    plot_market_replay,
    summarize_event_log,
)


def build_sample_log() -> EventLog:
    return EventLog(
        events=[
            MarketEvent(
                sequence=1,
                timestamp=0.0,
                event_type=EventType.SNAPSHOT,
                order_book=OrderBookSnapshot(
                    timestamp=0.0,
                    bids=(OrderBookLevel(99.0, 10.0), OrderBookLevel(98.5, 12.0)),
                    asks=(OrderBookLevel(100.0, 8.0), OrderBookLevel(100.5, 9.0)),
                ),
                payload={"latent_fundamental": 99.75},
            ),
            MarketEvent(
                sequence=2,
                timestamp=1.0,
                event_type=EventType.TRADE,
                agent_id="agent_1",
                side=OrderSide.SELL,
                price=99.8,
                quantity=3.0,
                payload={"agent_state": "momentum", "agent_annotation": "inventory trimmed"},
            ),
            MarketEvent(
                sequence=3,
                timestamp=2.0,
                event_type=EventType.NEWS,
                payload={"message": "volatility spike", "severity": 0.9},
            ),
        ]
    )


def test_build_replay_series_extracts_prices_and_trades() -> None:
    log = build_sample_log()
    series = build_replay_series(log)

    assert series.midpoint[0] == 99.5
    assert series.spread[0] == 1.0
    assert series.fundamental_values == [99.75]
    assert series.trade_prices == [99.8]
    assert series.trade_sides == [OrderSide.SELL.value]
    assert series.news_labels == ["volatility spike"]
    assert series.news_severities == [0.9]
    assert [annotation.label for annotation in series.annotations] == ["inventory trimmed"]


def test_summarize_event_log_reports_key_counts() -> None:
    log = build_sample_log()
    summary = summarize_event_log(log)

    assert summary["event_count"] == 3
    assert summary["trade_count"] == 1
    assert summary["news_count"] == 1
    assert summary["snapshot_count"] == 1
    assert summary["fundamental_point_count"] == 1
    assert summary["annotation_count"] == 1
    assert summary["unique_agent_count"] == 1
    assert summary["final_midpoint"] == 99.5
    assert summary["news_severity_max"] == 0.9


def test_plot_market_replay_writes_main_and_depth_figures(tmp_path: Path) -> None:
    log = build_sample_log()
    output_path = tmp_path / "replay.png"
    depth_path = tmp_path / "depth.png"

    try:
        result = plot_market_replay(log, output_path=output_path, depth_output_path=depth_path)
    except RuntimeError as exc:
        assert "matplotlib" in str(exc).lower()
        return

    assert output_path.exists()
    assert depth_path.exists()
    assert result == depth_path
