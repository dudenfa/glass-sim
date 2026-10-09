from __future__ import annotations

from scripts import compare_agent_ranking_panels as comparison


def _payload(*, total_pnl: float, trade_count: float) -> dict[str, object]:
    leaderboard_row = {
        "agentId": "rl_01",
        "agentType": "rl_agent",
        "medianRealizedPnl": total_pnl - 1.0,
        "medianUnrealizedPnl": 1.0,
        "medianTotalPnl": total_pnl,
        "medianPnlPct": total_pnl / 100.0,
        "meanDrawdown": 3.0,
        "meanMaxAbsInventory": 4.0,
    }
    return {
        "manifest": {"seeds": [31], "excluded_agent_ids": ["informed_01"]},
        "seedCount": 1,
        "structureOrder": ["s1_rl01"],
        "labels": {"s1_rl01": "S1: scripted + rl_01"},
        "structureLeaderboards": {"s1_rl01": [leaderboard_row]},
        "rows": [
            {
                "structureId": "s1_rl01",
                "seed": 31,
                "agentId": "rl_01",
                "realizedPnl": total_pnl - 1.0,
                "unrealizedPnl": 1.0,
                "totalPnl": total_pnl,
                "pnlPct": total_pnl / 100.0,
                "maxDrawdown": 3.0,
                "maxAbsInventory": 4.0,
            }
        ],
        "marketRows": [
            {
                "structureId": "s1_rl01",
                "seed": 31,
                "trade_count": trade_count,
                "spread_availability_ratio": 0.4,
                "mean_spread": 0.1,
                "midpoint_return_volatility_bps": 10.0,
                "mean_top_of_book_liquidity": 3.0,
                "final_midpoint": 100.0,
            }
        ],
        "marketSummary": [
            {
                "structureId": "s1_rl01",
                "structureLabel": "S1: scripted + rl_01",
                "trade_count": trade_count,
                "spread_availability_ratio": 0.4,
                "mean_spread": 0.1,
                "midpoint_return_volatility_bps": 10.0,
                "mean_top_of_book_liquidity": 3.0,
                "final_midpoint": 100.0,
            }
        ],
    }


def test_comparison_payload_reports_treatment_minus_baseline(monkeypatch, tmp_path) -> None:
    payloads = [_payload(total_pnl=100.0, trade_count=10.0), _payload(total_pnl=140.0, trade_count=15.0)]
    monkeypatch.setattr(comparison, "build_ranking_payload", lambda _path: payloads.pop(0))

    result = comparison.build_comparison_payload(tmp_path / "baseline", tmp_path / "treatment")

    assert result["agentRows"][0]["delta"]["medianTotalPnl"] == 40.0
    assert result["marketRows"][0]["delta"]["trade_count"] == 5.0
    assert result["pairedAgentRows"][0]["delta"]["medianTotalPnl"] == 40.0
    assert result["pairedMarketRows"][0]["delta"]["trade_count"] == 5.0
    result.update(
        {
            "title": "Test Comparison",
            "baselineLabel": "baseline",
            "treatmentLabel": "treatment",
        }
    )
    assert "# Test Comparison" in comparison._markdown(result)
