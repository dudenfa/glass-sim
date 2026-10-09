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


def test_build_market_health_report_can_include_portfolio_breakdown() -> None:
    payload = build_market_health_report("baseline", seed=7, horizon=24, portfolio_breakdown=True)

    assert payload["portfolio_breakdown"]
    first_row = payload["portfolio_breakdown"][0]
    assert first_row.agent_id
    assert first_row.starting_cash >= 0
    assert first_row.ending_cash >= 0
    assert first_row.starting_free_equity is not None
    assert first_row.peak_equity is not None
    assert first_row.max_equity_drawdown is not None
    assert first_row.max_equity_drawdown_from_start_replay is not None
    assert first_row.min_equity_delta is not None
    assert first_row.max_pnl_drawdown_from_start is not None
    assert first_row.max_abs_inventory is not None
    assert "portfolio_breakdown:" in payload["report"]
    assert "open orders" in payload["report"]
    assert "peak equity" in payload["report"]


def test_build_market_health_report_can_include_portfolio_breakdown() -> None:
    payload = build_market_health_report("baseline", seed=7, horizon=24, portfolio_breakdown=True)

    assert payload["portfolio_breakdown"] is not None
    assert len(payload["portfolio_breakdown"]) > 0
    assert "portfolio_breakdown:" in payload["report"]
    first_row = payload["portfolio_breakdown"][0].to_dict()
    assert "agent_id" in first_row
    assert "starting_cash" in first_row
    assert "ending_equity" in first_row
    assert "peak_equity" in first_row
    assert "max_equity_drawdown" in first_row
    assert "max_equity_drawdown_from_start_replay" in first_row
    assert "min_equity_delta" in first_row
    assert "max_pnl_drawdown_from_start" in first_row
    assert "max_abs_inventory" in first_row
