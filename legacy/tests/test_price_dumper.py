from __future__ import annotations

import numpy as np

from marl_trading.agents import PriceDumperAgent
from marl_trading.agents.base import MarketObservation
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.exchange.models import OrderType, Side
from marl_trading.market import SyntheticMarketSimulator
from marl_trading.rl.scenario import add_price_dumper_config


def _observation(*, inventory: float = 600.0) -> MarketObservation:
    return MarketObservation(
        timestamp_ns=1,
        symbol="SYNTH",
        tick_size=0.01,
        best_bid=99.99,
        best_ask=100.01,
        midpoint=100.0,
        spread=0.02,
        latent_fundamental=100.0,
        recent_midpoints=(100.0,),
        recent_returns_bps=(),
        news_headline=None,
        news_severity=None,
        agent_cash=10_000.0,
        agent_inventory=inventory,
        agent_equity=10_000.0 + inventory * 100.0,
        open_orders=0,
        active_agents=5,
        portfolio_active=True,
        agent_type="price_dumper",
    )


def test_price_dumper_only_submits_sell_orders_with_both_order_types() -> None:
    agent = PriceDumperAgent("dumper_01")
    rng = np.random.default_rng(7)

    intents = [agent.decide(_observation(), rng)[0] for _ in range(40)]

    assert {intent.side for intent in intents} == {Side.SELL}
    assert {intent.order_type for intent in intents} == {OrderType.MARKET, OrderType.LIMIT}
    assert all(3 <= intent.quantity <= 8 for intent in intents)


def test_price_dumper_runtime_config_is_opt_in_and_has_large_inventory() -> None:
    baseline = default_simulation_config()
    updated = add_price_dumper_config(baseline)

    assert [agent.agent_id.value for agent in baseline.agents] == ["maker_01", "retail_01", "informed_01", "trend_01"]
    assert updated.agents[-1].agent_id.value == "dumper_01"
    assert updated.agents[-1].agent_type == "price_dumper"

    simulator = SyntheticMarketSimulator(updated, horizon=20)
    assert simulator.portfolios.get("dumper_01").starting_inventory == 600.0
