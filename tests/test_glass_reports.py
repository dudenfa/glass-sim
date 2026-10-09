import pytest

from glass_sim import PerpetualMarket, scenarios
from glass_sim.account import MarginAccount
from glass_sim.compare import compare_panels, compare_runs, config_from_spec, parse_seeds, parse_spec
from glass_sim.results import SavedRun
from glass_sim.types import Side, Transparency


def test_path_metrics_track_peak_drawdown_and_exposure():
    acc = MarginAccount("a", 1_000.0)
    acc.apply_fill(Side.BUY, 10, 100.0, fee=0.0)
    for mark in (100.0, 110.0, 95.0, 105.0):
        acc.update_path(mark)
    assert acc.peak_equity == pytest.approx(1_100.0)
    assert acc.min_equity == pytest.approx(950.0)
    assert acc.max_drawdown == pytest.approx(150.0)
    assert acc.max_drawdown_pct == pytest.approx(100 * 150.0 / 1_100.0)
    assert acc.max_abs_position == 10
    assert acc.max_leverage == pytest.approx(950.0 / 950.0)


def test_report_contains_every_agent_and_activity_counts():
    result = PerpetualMarket(scenarios.small(seed=1, duration=600.0)).run()
    text = result.report(agents=True)
    for account in result.accounts:
        assert account.agent_id in text
    assert "By participant type" in text and "maxDD%" in text
    mm = next(a for a in result.accounts if a.agent_type == "market_maker")
    assert mm.maker_share > 0.9 and mm.orders_accepted > 0


def test_saved_run_round_trip_and_single_comparison(tmp_path):
    left = PerpetualMarket(scenarios.small(seed=2, duration=600.0)).run()
    right = PerpetualMarket(scenarios.small(seed=2, duration=600.0, transparency=Transparency.ANONYMOUS)).run()
    left.save(tmp_path / "left")
    loaded = SavedRun.load(tmp_path / "left")
    assert loaded.summary["trades"] == left.summary()["trades"]
    assert (tmp_path / "left" / "report.txt").read_text().startswith("Scenario small")
    text = compare_runs(loaded, right, show_agents=True)
    assert "left:  small seed=2 full" in text and "right: small seed=2 anonymous" in text
    for account in left.accounts:
        assert account.agent_id in text


def test_specs_and_panel_comparison():
    assert parse_seeds("1-3,7") == [1, 2, 3, 7]
    assert parse_spec("scenario=small seed=4") == {"scenario": "small", "seed": "4"}
    config = config_from_spec("scenario=small transparency=anonymous duration=5m", seed=9)
    assert config.seed == 9 and config.transparency is Transparency.ANONYMOUS and config.duration == 300.0
    with pytest.raises(ValueError):
        parse_spec("scenario=small colour=red")
    text = compare_panels("scenario=small duration=5m", "scenario=small duration=5m transparency=anonymous", [1, 2])
    assert "seeds: 2" in text and "return % retail" in text
