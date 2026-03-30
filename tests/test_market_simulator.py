from __future__ import annotations

import pytest

from marl_trading.agents.base import OrderIntent
from marl_trading.analysis import EventLog, EventType, summarize_event_log
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.exchange.models import OrderType, Side
from marl_trading.market import SyntheticMarketSimulator


def test_synthetic_market_demo_is_deterministic_and_generates_activity() -> None:
    config = default_simulation_config()
    sim1 = SyntheticMarketSimulator(config, horizon=120)
    sim2 = SyntheticMarketSimulator(config, horizon=120)

    result1 = sim1.run(horizon=120)
    result2 = sim2.run(horizon=120)

    assert result1.event_log.to_dict() == result2.event_log.to_dict()
    assert result1.summary["trade_count"] > 0
    assert result1.summary["news_count"] > 0
    assert result1.summary["snapshot_count"] > 0
    assert len(result1.step_records) == 120

    summary = summarize_event_log(result1.event_log)
    assert summary["has_order_book_snapshots"] is True
    assert summary["news_count"] == result1.summary["news_count"]


def test_synthetic_market_event_log_uses_replay_contract() -> None:
    config = default_simulation_config()
    result = SyntheticMarketSimulator(config, horizon=60).run(horizon=60)

    log = EventLog.from_dict(result.event_log.to_dict())
    assert len(log.snapshots()) > 0
    assert len(log.news_events()) > 0
    assert any(event.event_type == EventType.TRADE for event in log.events)


def test_synthetic_market_long_horizon_keeps_news_visible() -> None:
    config = default_simulation_config()
    result = SyntheticMarketSimulator(config, horizon=1_200).run(horizon=1_200)

    assert result.summary["news_count"] >= 20
    assert result.summary["trade_count"] > 0


def test_synthetic_market_settles_trades_using_currency_prices() -> None:
    sim = SyntheticMarketSimulator(default_simulation_config(), horizon=12)
    sim.reset(seed=7, horizon=12)

    retail_start_cash = sim.portfolios.get("retail_01").cash
    maker_start_cash = sim.portfolios.get("maker_01").cash

    sim._submit_intent(
        agent_id="retail_01",
        intent=OrderIntent(
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
            annotation="regression_buy",
        ),
        timestamp_ns=1_000,
        step_index=1,
    )

    retail_after = sim.portfolios.get("retail_01")
    maker_after = sim.portfolios.get("maker_01")

    assert retail_after.cash == pytest.approx(retail_start_cash - 100.01)
    assert maker_after.cash == pytest.approx(maker_start_cash + 100.01)
    assert retail_after.cash > 9_000.0
    assert maker_after.cash > 10_000.0
    assert retail_after.active is True
    assert maker_after.active is True
