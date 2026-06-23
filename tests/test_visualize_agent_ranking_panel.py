from __future__ import annotations

import json

from scripts import visualize_agent_ranking_panel as visualization


def _report(seed: int, structure_id: str, label: str, pnl_offset: float) -> dict[str, object]:
    return {
        "seed": seed,
        "summary": {"trade_count": 100, "spread_availability_ratio": 0.4, "mean_spread": 0.1},
        "metadata": {"ranking_structure_id": structure_id, "ranking_structure_label": label},
        "portfolio_breakdown": [
            {"agent_id": "maker_01", "agent_type": "market_maker", "starting_equity": 100.0, "total_pnl": 4.0, "realized_pnl": 3.0, "unrealized_pnl": 1.0},
            {"agent_id": "rl_01", "agent_type": "rl_agent", "starting_equity": 100.0, "total_pnl": 6.0 + pnl_offset, "realized_pnl": 4.0, "unrealized_pnl": 2.0 + pnl_offset},
        ],
    }


def test_build_ranking_payload_ranks_by_starting_equity_normalized_pnl(tmp_path) -> None:
    (tmp_path / "s1_rl01_seed_31.json").write_text(
        json.dumps(_report(31, "s1_rl01", "S1: scripted + rl_01", 0.0)), encoding="utf-8"
    )
    (tmp_path / "s1_rl01_seed_32.json").write_text(
        json.dumps(_report(32, "s1_rl01", "S1: scripted + rl_01", 2.0)), encoding="utf-8"
    )

    payload = visualization.build_ranking_payload(tmp_path)

    leaderboard = payload["structureLeaderboards"]["s1_rl01"]
    assert leaderboard[0]["agentId"] == "rl_01"
    assert leaderboard[0]["wins"] == 2
    assert "Fixed-16 Agent Ranking Panel" in visualization._dashboard_html(payload)
    assert "## S1: scripted + rl_01" in visualization._markdown(payload)
