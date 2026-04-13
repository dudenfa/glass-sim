from __future__ import annotations

import json

import pytest

import scripts.compare_market_runs as compare_script


def _report(preset: str, trade_count: int) -> dict[str, object]:
    return {
        "preset": preset,
        "seed": 7,
        "horizon": 240,
        "summary": {
            "event_count": 10,
            "step_count": 5,
            "trade_count": trade_count,
            "news_count": 2,
            "snapshot_count": 4,
            "unique_agent_count": 2,
            "snapshot_coverage_ratio": 0.4,
            "spread_availability_ratio": 0.5,
            "mean_spread": 0.2,
            "midpoint_return_volatility_bps": 10.0,
            "top_of_book_occupancy_ratio": 0.8,
            "mean_top_of_book_liquidity": 20.0,
            "active_agent_mean": 4.0,
            "mean_total_equity": 100.0,
            "final_total_equity": 101.0,
            "final_midpoint": 100.0,
            "final_fundamental": 99.0,
        },
        "portfolio_breakdown": [
            {
                "agent_id": "maker_01",
                "agent_type": "market_maker",
                "active": True,
                "equity": 50.0,
                "cash": 40.0,
                "inventory": 1,
                "realized_pnl": 5.0,
                "unrealized_pnl": 1.0,
                "open_orders": 2,
            }
        ],
        "agents": {
            "maker_01": {
                "agent_type": "market_maker",
                "active": True,
                "equity": 50.0,
                "cash": 40.0,
                "inventory": 1,
                "realized_pnl": 5.0,
                "unrealized_pnl": 1.0,
                "open_orders": 2,
            }
        },
    }


def test_parse_run_spec_supports_bare_and_key_value_specs() -> None:
    assert compare_script.parse_run_spec("baseline") == {"preset": "baseline"}
    assert compare_script.parse_run_spec("preset=high_news seed=11 horizon=320") == {
        "preset": "high_news",
        "seed": "11",
        "horizon": "320",
    }


def test_cli_compares_json_reports(tmp_path, capsys) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(_report("baseline", 10)), encoding="utf-8")
    right.write_text(json.dumps(_report("high_news", 15)), encoding="utf-8")

    compare_script.main([str(left), str(right)])
    output = capsys.readouterr().out

    assert "Summary metrics" in output
    assert "Trades" in output
    assert "Agent coverage" in output
    assert "maker_01" in output


def test_cli_compares_run_specs_via_health_builder(monkeypatch, capsys) -> None:
    calls: list[tuple[str, int | None, int | None]] = []

    def fake_builder(
        preset: str,
        *,
        seed: int | None = None,
        horizon: int | None = None,
        portfolio_breakdown: bool = False,
    ):
        calls.append((preset, seed, horizon))
        return _report(preset, 10 if preset == "baseline" else 15)

    monkeypatch.setattr(compare_script, "build_market_health_report", fake_builder)

    compare_script.main(["preset=baseline seed=7 horizon=1000", "preset=high_news seed=7 horizon=1000", "--json"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["left"]["preset"] == "baseline"
    assert payload["right"]["preset"] == "high_news"
    assert payload["summary_metrics"][2]["key"] == "trade_count"
    assert calls == [("baseline", 7, 1000), ("high_news", 7, 1000)]


def test_cli_rejects_invalid_run_spec() -> None:
    with pytest.raises(ValueError):
        compare_script.parse_run_spec("seed=7 horizon=1000")
