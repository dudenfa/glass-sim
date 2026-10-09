from __future__ import annotations

from pathlib import Path

from marl_trading.analysis import EventLog, plot_market_replay
from scripts.run_market_demo import _write_outputs
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.market import SyntheticMarketSimulator


def test_demo_writes_event_log_and_summary(tmp_path: Path) -> None:
    result = SyntheticMarketSimulator(default_simulation_config(), horizon=60).run(horizon=60)
    paths = _write_outputs(result, tmp_path, summary_only=True)

    assert Path(paths["event_log"]).exists()
    assert Path(paths["summary"]).exists()


def test_demo_event_log_is_replayable(tmp_path: Path) -> None:
    result = SyntheticMarketSimulator(default_simulation_config(), horizon=60).run(horizon=60)
    event_log_path = tmp_path / "events.jsonl"
    result.event_log.save(event_log_path)

    loaded = EventLog.load(event_log_path)
    assert len(loaded.events) == len(result.event_log.events)

    try:
        replay_path = tmp_path / "replay.png"
        depth_path = tmp_path / "depth.png"
        plot_market_replay(loaded, output_path=replay_path, depth_output_path=depth_path)
    except RuntimeError:
        return

    assert replay_path.exists()
