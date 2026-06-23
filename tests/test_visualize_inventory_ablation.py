from __future__ import annotations

import json

import scripts.visualize_inventory_ablation as visualization


def _report(*, trade_count: int, spread: float, pnl_offset: float) -> dict[str, object]:
    return {
        "summary": {
            "trade_count": trade_count,
            "spread_availability_ratio": 0.4,
            "mean_spread": spread,
            "midpoint_return_volatility_bps": 12.0,
            "mean_top_of_book_liquidity": 3.0,
            "final_midpoint": 100.0 + pnl_offset,
        },
        "portfolio_breakdown": [
            {"agent_id": "rl_01", "total_pnl": 10.0 + pnl_offset},
            {"agent_id": "rl_02", "total_pnl": 20.0 + pnl_offset},
            {"agent_id": "rl_03", "total_pnl": 30.0 + pnl_offset},
        ],
    }


def test_dashboard_payload_and_html_cover_paired_reports(tmp_path) -> None:
    cash_path = tmp_path / "seed_21_all_cashonly.json"
    endowed_path = tmp_path / "seed_21_all_endowed_16.json"
    cash_path.write_text(json.dumps(_report(trade_count=100, spread=0.2, pnl_offset=0.0)), encoding="utf-8")
    endowed_path.write_text(json.dumps(_report(trade_count=125, spread=0.1, pnl_offset=5.0)), encoding="utf-8")

    payload = visualization.build_dashboard_payload(tmp_path)

    assert payload["seedCount"] == 1
    assert payload["runs"][0]["seed"] == 21
    assert payload["runs"][0]["endowed"]["final_midpoint"] == 105.0
    assert payload["metrics"][0]["deltaMean"] == 25.0
    assert payload["agents"][2]["deltaMean"] == 5.0
    assert "Inventory Endowment Ablation" in visualization._dashboard_html(payload)


def test_cli_writes_self_contained_dashboard(tmp_path) -> None:
    (tmp_path / "seed_21_all_cashonly.json").write_text(
        json.dumps(_report(trade_count=100, spread=0.2, pnl_offset=0.0)), encoding="utf-8"
    )
    (tmp_path / "seed_21_all_endowed_16.json").write_text(
        json.dumps(_report(trade_count=125, spread=0.1, pnl_offset=5.0)), encoding="utf-8"
    )
    output = tmp_path / "dashboard.html"

    visualization.main([str(tmp_path), "--output", str(output)])

    html = output.read_text(encoding="utf-8")
    assert "Market Quality By Seed" in html
    assert "Final price" in html
    assert "rl_03" in html
    assert "https://" not in html
