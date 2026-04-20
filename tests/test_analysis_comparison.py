from __future__ import annotations

import json

from marl_trading.analysis import PortfolioHealthRow, compare_market_runs, format_market_run_comparison, load_market_run


def _report(
    *,
    preset: str,
    trade_count: int,
    final_total_equity: float,
    agents: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "preset": preset,
        "seed": 7,
        "horizon": 1_000,
        "summary": {
            "event_count": 20,
            "step_count": 10,
            "trade_count": trade_count,
            "news_count": 3,
            "snapshot_count": 9,
            "unique_agent_count": len(agents),
            "snapshot_coverage_ratio": 0.45,
            "spread_availability_ratio": 0.5,
            "mean_spread": 0.25,
            "midpoint_return_volatility_bps": 12.0,
            "top_of_book_occupancy_ratio": 0.8,
            "mean_top_of_book_liquidity": 20.0,
            "active_agent_mean": 4.0,
            "mean_total_equity": 100.0,
            "final_total_equity": final_total_equity,
            "final_midpoint": 101.0,
            "final_fundamental": 102.0,
        },
        "agents": agents,
    }


def test_compare_market_runs_from_payloads_includes_summary_and_agents() -> None:
    left = _report(
        preset="baseline",
        trade_count=100,
        final_total_equity=1_000.0,
        agents={
            "maker_01": {
                "agent_type": "market_maker",
                "active": True,
                "equity": 500.0,
                "cash": 250.0,
                "inventory": 2,
                "realized_pnl": 20.0,
                "unrealized_pnl": 5.0,
                "open_orders": 2,
            },
            "trend_01": {
                "agent_type": "trend_follower",
                "active": True,
                "equity": 500.0,
                "cash": 300.0,
                "inventory": 1,
                "realized_pnl": 10.0,
                "unrealized_pnl": 2.0,
                "open_orders": 1,
            },
        },
    )
    right = _report(
        preset="high_news",
        trade_count=120,
        final_total_equity=1_050.0,
        agents={
            "maker_01": {
                "agent_type": "market_maker",
                "active": True,
                "equity": 530.0,
                "cash": 260.0,
                "inventory": 3,
                "realized_pnl": 25.0,
                "unrealized_pnl": 7.0,
                "open_orders": 1,
            },
            "rl_01": {
                "agent_type": "rl_agent",
                "active": True,
                "equity": 20.0,
                "cash": 20.0,
                "inventory": 0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_orders": 0,
            },
        },
    )

    comparison = compare_market_runs(left, right)

    assert comparison.left.preset == "baseline"
    assert comparison.right.preset == "high_news"
    assert comparison.summary_metrics[2].key == "trade_count"
    assert comparison.summary_metrics[2].delta == 20
    assert comparison.summary_metrics[-1].key == "final_fundamental"
    assert comparison.summary_metrics[-1].delta == 0.0
    assert comparison.left_only_agents == ("trend_01",)
    assert comparison.right_only_agents == ("rl_01",)
    assert comparison.shared_agents == ("maker_01",)

    shared = next(agent for agent in comparison.agent_comparisons if agent.agent_id == "maker_01")
    assert shared.left_present is True
    assert shared.right_present is True
    equity_row = next(metric for metric in shared.metrics if metric.key == "equity")
    assert equity_row.delta == 30.0

    formatted = format_market_run_comparison(comparison)
    assert "Summary metrics" in formatted
    assert "Agent coverage" in formatted
    assert "left-only ids: trend_01" in formatted
    assert "right-only ids: rl_01" in formatted
    assert "### maker_01 (shared, type=market_maker)" in formatted
    assert "### rl_01 (right-only, type=rl_agent)" in formatted


def test_load_market_run_from_json_file(tmp_path) -> None:
    report = _report(
        preset="baseline",
        trade_count=42,
        final_total_equity=999.5,
        agents={},
    )
    report["portfolio_breakdown"] = [
        {
            "agent_id": "maker_01",
            "agent_type": "market_maker",
            "equity": 500.0,
            "cash": 250.0,
            "inventory": 2,
            "realized_pnl": 20.0,
            "unrealized_pnl": 5.0,
            "open_orders": 2,
        }
    ]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    snapshot = load_market_run(path)

    assert snapshot.preset == "baseline"
    assert snapshot.seed == 7
    assert snapshot.horizon == 1_000
    assert snapshot.summary["trade_count"] == 42
    assert snapshot.agents["maker_01"]["open_orders"] == 2
    assert snapshot.raw["preset"] == "baseline"


def test_load_market_run_prefers_explicit_label_over_preset() -> None:
    snapshot = load_market_run(
        {
            "preset": "baseline",
            "label": "baseline_rl",
            "seed": 7,
            "horizon": 64,
            "summary": {"trade_count": 10},
        }
    )

    assert snapshot.preset == "baseline"
    assert snapshot.label == "baseline_rl"


def test_compare_market_runs_accepts_direct_portfolio_breakdown_rows() -> None:
    left = {
        "preset": "baseline",
        "seed": 7,
        "horizon": 60,
        "summary": {"trade_count": 10, "final_total_equity": 1000.0},
        "portfolio_breakdown": [
            PortfolioHealthRow(
                agent_id="maker_01",
                agent_type="market_maker",
                status="active",
                active=True,
                starting_cash=100.0,
                ending_cash=110.0,
                starting_inventory=4.0,
                ending_inventory=3.0,
                starting_equity=500.0,
                ending_equity=520.0,
                starting_free_equity=500.0,
                ending_free_equity=510.0,
                cash_delta=10.0,
                inventory_delta=-1.0,
                equity_delta=20.0,
                total_pnl=20.0,
                realized_pnl=15.0,
                unrealized_pnl=5.0,
                open_orders=1,
            )
        ],
    }
    right = {
        "preset": "high_news",
        "seed": 7,
        "horizon": 60,
        "summary": {"trade_count": 12, "final_total_equity": 1010.0},
        "portfolio_breakdown": [
            PortfolioHealthRow(
                agent_id="maker_01",
                agent_type="market_maker",
                status="active",
                active=True,
                starting_cash=100.0,
                ending_cash=112.0,
                starting_inventory=4.0,
                ending_inventory=2.0,
                starting_equity=500.0,
                ending_equity=530.0,
                starting_free_equity=500.0,
                ending_free_equity=520.0,
                cash_delta=12.0,
                inventory_delta=-2.0,
                equity_delta=30.0,
                total_pnl=30.0,
                realized_pnl=22.0,
                unrealized_pnl=8.0,
                open_orders=2,
            )
        ],
    }

    comparison = compare_market_runs(left, right)

    assert comparison.shared_agents == ("maker_01",)
    shared = comparison.agent_comparisons[0]
    equity_row = next(metric for metric in shared.metrics if metric.key == "equity")
    inventory_row = next(metric for metric in shared.metrics if metric.key == "inventory")
    assert equity_row.delta == 10.0
    assert inventory_row.delta == -1.0
