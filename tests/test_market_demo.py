from __future__ import annotations

from pathlib import Path

from marl_trading.market import plot_market_world
from scripts.run_market_demo import _write_outputs, run_demo


def test_run_demo_generates_activity_and_summary() -> None:
    result = run_demo(seed=7, horizon=80)

    assert result.summary["trade_count"] > 0
    assert result.summary["news_count"] > 0
    assert result.summary["snapshot_count"] > 0
    assert len(result.step_records) == 80
    assert result.final_portfolios


def test_demo_outputs_include_world_plot_when_available(tmp_path: Path) -> None:
    result = run_demo(seed=7, horizon=80)
    paths = _write_outputs(result, tmp_path, summary_only=False)

    assert Path(paths["event_log"]).exists()
    assert Path(paths["summary"]).exists()

    if "world_plot" in paths:
        assert Path(paths["world_plot"]).exists()
    else:
        assert "world_plot_error" in paths


def test_world_plot_can_be_written_directly(tmp_path: Path) -> None:
    result = run_demo(seed=7, horizon=80)
    output_path = tmp_path / "market_world.png"

    try:
        plotted = plot_market_world(result, output_path)
    except RuntimeError as exc:
        assert "matplotlib" in str(exc).lower()
        return

    assert plotted == output_path
    assert output_path.exists()
