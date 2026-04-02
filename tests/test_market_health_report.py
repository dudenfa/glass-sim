from __future__ import annotations

from marl_trading.analysis import MarketHealthSummary, format_market_health_summary
from scripts.run_market_health import build_market_health_report


def test_format_market_health_summary_is_compact() -> None:
    summary = MarketHealthSummary(
        event_count=120,
        step_count=60,
        trade_count=18,
        news_count=4,
        snapshot_count=48,
        unique_agent_count=5,
        snapshot_coverage_ratio=0.4,
        spread_availability_ratio=0.8,
        mean_spread=0.07,
        midpoint_return_volatility_bps=12.5,
        top_of_book_occupancy_ratio=0.9,
        mean_top_of_book_liquidity=37.2,
        active_agent_mean=3.75,
        mean_total_equity=101_250.5,
        final_total_equity=101_800.25,
        final_midpoint=100.125,
        final_fundamental=100.5,
    )

    report = format_market_health_summary(summary, preset_name="baseline", seed=7, horizon=240)

    lines = report.splitlines()
    assert lines[0] == "preset=baseline seed=7 horizon=240"
    assert "events=120 steps=60 trades=18 news=4 snapshots=48 agents=5" in lines[1]
    assert "coverage=0.400" in lines[2]
    assert "spread_availability=0.800" in lines[2]
    assert "mean_spread=0.0700" in lines[2]
    assert "final_total_equity=101800.25" in lines[2]
    assert "final_midpoint=100.1250" in lines[2]
    assert "final_fundamental=100.5000" in lines[2]


def test_build_market_health_report_uses_named_preset() -> None:
    payload = build_market_health_report("high_news", seed=11, horizon=24)

    assert payload["preset"] == "high_news"
    assert payload["seed"] == 11
    assert payload["horizon"] == 24
    assert payload["description"]
    assert payload["summary"].event_count > 0
    assert "preset=high_news" in payload["report"]
    assert "events=" in payload["report"]
    assert "final_total_equity=" in payload["report"]
