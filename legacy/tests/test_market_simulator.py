from __future__ import annotations

from dataclasses import replace

import pytest

from marl_trading.agents.base import MarketObservation, OrderIntent, ScriptedAgent
from marl_trading.analysis import EventLog, EventType, OrderBookLevel, OrderBookSnapshot, summarize_event_log
from marl_trading.core.config import AgentConfig
from marl_trading.core.domain import AgentId
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.exchange.models import OrderType, Side
from marl_trading.market import SyntheticMarketSimulator
from marl_trading.rl.scenario import exclude_agents_from_config


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


def test_exogenous_path_is_fixed_when_agent_configuration_changes() -> None:
    config = default_simulation_config()
    extra_noise_agent = AgentConfig(
        agent_id=AgentId("retail_02"),
        agent_type="noise_trader",
        starting_cash=10_000.0,
        ruin_threshold=4_000.0,
        max_resting_orders=2,
    )
    expanded_config = replace(config, agents=(*config.agents, extra_noise_agent))

    baseline = SyntheticMarketSimulator(config, horizon=120).run(horizon=120)
    expanded = SyntheticMarketSimulator(expanded_config, horizon=120).run(horizon=120)

    assert baseline.final_fundamental == pytest.approx(expanded.final_fundamental)
    assert [record.fundamental for record in baseline.step_records] == pytest.approx(
        [record.fundamental for record in expanded.step_records]
    )
    assert [event.payload for event in baseline.event_log.news_events()] == [
        event.payload for event in expanded.event_log.news_events()
    ]


def test_preserved_excluded_turn_slot_is_a_noop_without_redistributing_actions() -> None:
    config = exclude_agents_from_config(
        default_simulation_config(),
        excluded_agent_ids=("informed_01",),
        preserve_excluded_turn_slots=True,
    )
    sim = SyntheticMarketSimulator(config, horizon=8)
    sim.reset(seed=7, horizon=8)
    decision_counts = {agent_id: 0 for agent_id in sim.agents}

    class _CountingAgent(ScriptedAgent):
        def __init__(self, agent_id: str, agent_type: str) -> None:
            super().__init__(agent_id=agent_id, agent_type=agent_type)

        def decide(self, observation: MarketObservation, rng):  # noqa: ARG002
            decision_counts[self.agent_id] += 1
            return ()

    for agent_id, agent in list(sim.agents.items()):
        sim.agents[agent_id] = _CountingAgent(agent_id, agent.agent_type)

    for _ in range(8):
        sim.step()

    assert config.turn_schedule_agent_ids == ("maker_01", "retail_01", "informed_01", "trend_01")
    assert decision_counts == {"maker_01": 2, "retail_01": 2, "trend_01": 2}
    noop_steps = [
        event.timestamp
        for event in sim.event_log.snapshots()
        if event.payload.get("scheduled_noop") is True
    ]
    assert noop_steps == [3.0, 7.0]


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


def test_simulator_accepts_multiple_intents_from_single_decision_step() -> None:
    class _DualQuoteAgent(ScriptedAgent):
        def __init__(self) -> None:
            super().__init__(agent_id="maker_01", agent_type="market_maker", max_resting_orders=3)

        def decide(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:  # noqa: ARG002
            return (
                OrderIntent(
                    side=Side.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=1,
                    limit_price=99.95,
                    annotation="test_bid",
                ),
                OrderIntent(
                    side=Side.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=1,
                    limit_price=100.05,
                    annotation="test_ask",
                ),
            )

    sim = SyntheticMarketSimulator(default_simulation_config(), horizon=3)
    sim.reset(seed=7, horizon=3)
    sim.agents["maker_01"] = _DualQuoteAgent()
    sim.open_orders["maker_01"].clear()

    sim.step()

    order_events = [
        event
        for event in sim.event_log.events
        if event.timestamp == 1.0 and event.event_type in {EventType.LIMIT_ORDER, EventType.MARKET_ORDER}
    ]
    annotations = [event.payload.get("annotation") for event in order_events]

    assert annotations.count("test_bid") == 1
    assert annotations.count("test_ask") == 1


def test_mark_price_for_inventory_prefers_executable_prices_over_fundamental() -> None:
    sim = SyntheticMarketSimulator(default_simulation_config(), horizon=3)
    sim.reset(seed=7, horizon=3)
    sim.last_trade_price = 100.25
    snapshot = OrderBookSnapshot(
        timestamp=1.0,
        bids=(OrderBookLevel(99.75, 5.0),),
        asks=(OrderBookLevel(100.25, 5.0),),
    )

    assert sim.mark_price_for_inventory(3.0, snapshot=snapshot) == pytest.approx(99.75)
    assert sim.mark_price_for_inventory(-2.0, snapshot=snapshot) == pytest.approx(100.25)
    assert sim.mark_price_for_inventory(0.0, snapshot=snapshot) == pytest.approx(100.0)

    ask_only = OrderBookSnapshot(
        timestamp=2.0,
        bids=(),
        asks=(OrderBookLevel(101.0, 4.0),),
    )
    assert sim.mark_price_for_inventory(3.0, snapshot=ask_only) == pytest.approx(100.25)
