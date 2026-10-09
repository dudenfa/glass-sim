from __future__ import annotations

from marl_trading.configs.defaults import default_agent_configs, default_market_config
from marl_trading.core.config import (
    AgentBehaviorConfig,
    AgentConfig,
    InformedTraderBehaviorConfig,
    MarketMakerBehaviorConfig,
    NoiseTraderBehaviorConfig,
    SimulationConfig,
    TrendFollowerBehaviorConfig,
)
from marl_trading.core.domain import AgentId, SimulationId
from marl_trading.market.simulator import SyntheticMarketSimulator


def test_default_agent_configs_keep_behavior_optional() -> None:
    configs = default_agent_configs()
    assert all(agent.behavior is None for agent in configs)


def test_simulator_applies_scripted_agent_behavior_overrides() -> None:
    config = SimulationConfig(
        simulation_id=SimulationId("sim_tuning_surface"),
        market=default_market_config(),
        agents=(
            AgentConfig(
                agent_id=AgentId("maker_01"),
                agent_type="market_maker",
                starting_cash=10_000.0,
                ruin_threshold=4_000.0,
                max_resting_orders=3,
                behavior=AgentBehaviorConfig(
                    market_maker=MarketMakerBehaviorConfig(
                        inventory_anchor=12.0,
                        quote_size=7,
                        quote_padding_ticks=4,
                        inventory_tolerance=2.5,
                        min_quote_size=2,
                        max_quote_size=9,
                        bid_padding_ticks=5,
                        ask_padding_ticks=3,
                        inventory_skew_strength=1.25,
                        inventory_size_decay=0.65,
                        empty_side_padding_ticks=1,
                    ),
                ),
            ),
            AgentConfig(
                agent_id=AgentId("noise_01"),
                agent_type="noise_trader",
                starting_cash=10_000.0,
                ruin_threshold=4_000.0,
                max_resting_orders=2,
                behavior=AgentBehaviorConfig(
                    noise_trader=NoiseTraderBehaviorConfig(
                        aggressiveness=0.95,
                        market_order_probability=0.12,
                        sell_bias=0.7,
                        inventory_recycling_bias=0.35,
                        overpricing_sell_bias=0.4,
                        profit_taking_bias=0.2,
                    ),
                ),
            ),
            AgentConfig(
                agent_id=AgentId("trend_01"),
                agent_type="trend_follower",
                starting_cash=10_000.0,
                ruin_threshold=4_000.0,
                max_resting_orders=2,
                behavior=AgentBehaviorConfig(
                    trend_follower=TrendFollowerBehaviorConfig(
                        threshold_bps=3.8,
                        market_order_probability=0.15,
                        exit_threshold_bps=0.7,
                        overpricing_exit_bias=1.2,
                        inventory_pressure=0.8,
                    ),
                ),
            ),
            AgentConfig(
                agent_id=AgentId("informed_01"),
                agent_type="informed_trader",
                starting_cash=10_000.0,
                ruin_threshold=4_000.0,
                max_resting_orders=2,
                private_signal_strength=0.0,
                behavior=AgentBehaviorConfig(
                    informed_trader=InformedTraderBehaviorConfig(
                        private_signal_strength=1.8,
                        signal_noise=0.02,
                        news_bias=2.75,
                        threshold_bps=0.45,
                        sell_bias=1.9,
                        negative_news_sell_bias=1.4,
                        inventory_pressure=0.85,
                    ),
                ),
            ),
        ),
        seed=5,
        enable_news=False,
        enable_private_signals=True,
        public_tape_enabled=True,
    )

    simulator = SyntheticMarketSimulator(config, horizon=12)

    maker = simulator.agents["maker_01"]
    assert maker.inventory_anchor == 12.0
    assert maker.quote_size == 7
    assert maker.quote_padding_ticks == 4
    assert maker.inventory_tolerance == 2.5
    assert maker.min_quote_size == 2
    assert maker.max_quote_size == 9
    assert maker.bid_padding_ticks == 5
    assert maker.ask_padding_ticks == 3
    assert maker.inventory_skew_strength == 1.25
    assert maker.inventory_size_decay == 0.65
    assert maker.empty_side_padding_ticks == 1

    noise = simulator.agents["noise_01"]
    assert noise.aggressiveness == 0.95
    assert noise.market_order_probability == 0.12
    assert noise.sell_bias == 0.7
    assert noise.inventory_recycling_bias == 0.35
    assert noise.overpricing_sell_bias == 0.4
    assert noise.profit_taking_bias == 0.2

    trend = simulator.agents["trend_01"]
    assert trend.threshold_bps == 3.8
    assert trend.market_order_probability == 0.15
    assert trend.exit_threshold_bps == 0.7
    assert trend.overpricing_exit_bias == 1.2
    assert trend.inventory_pressure == 0.8

    informed = simulator.agents["informed_01"]
    assert config.agents[-1].private_signal_strength == 0.0
    assert informed.signal_noise == 0.02
    assert informed.news_bias == 2.75
    assert informed.threshold_bps == 0.45
    assert informed.sell_bias == 1.9
    assert informed.negative_news_sell_bias == 1.4
    assert informed.inventory_pressure == 0.85
    # The nested behavior override is what should drive the signal strength here.
    informed_kwargs = simulator._informed_trader_kwargs(config.agents[-1])
    assert informed_kwargs["signal_noise"] == 0.02
    assert informed_kwargs["news_bias"] == 2.75
    assert informed_kwargs["threshold_bps"] == 0.45
    assert informed_kwargs["sell_bias"] == 1.9
    assert informed_kwargs["negative_news_sell_bias"] == 1.4
    assert informed_kwargs["inventory_pressure"] == 0.85


def test_simulator_uses_legacy_defaults_without_behavior_overrides() -> None:
    config = SimulationConfig(
        simulation_id=SimulationId("sim_default_legacy"),
        market=default_market_config(),
        agents=default_agent_configs(),
        seed=7,
        enable_news=False,
        enable_private_signals=True,
        public_tape_enabled=True,
    )

    simulator = SyntheticMarketSimulator(config, horizon=12)

    maker = simulator.agents["maker_01"]
    assert maker.inventory_anchor == 40.0
    assert maker.quote_size == 3
    assert maker.quote_padding_ticks == 1
    assert maker.min_quote_size == 1
    assert maker.max_quote_size == 3
    assert maker.bid_padding_ticks == 1
    assert maker.ask_padding_ticks == 1
    assert maker.empty_side_padding_ticks == 1

    noise = simulator.agents["retail_01"]
    assert noise.aggressiveness == 0.55
    assert noise.market_order_probability == 0.7
    assert noise.sell_bias == 0.5
    assert noise.inventory_recycling_bias == 0.2
    assert noise.overpricing_sell_bias == 0.15
    assert noise.profit_taking_bias == 0.1

    trend = simulator.agents["trend_01"]
    assert trend.threshold_bps == 1.5
    assert trend.market_order_probability == 0.5
    assert trend.exit_threshold_bps == 0.6
    assert trend.overpricing_exit_bias == 0.9
    assert trend.inventory_pressure == 0.5

    informed = simulator.agents["informed_01"]
    assert informed.signal_noise == 0.3
    assert informed.news_bias == 1.25
    assert informed.threshold_bps == 1.0
    assert informed.sell_bias == 1.35
    assert informed.negative_news_sell_bias == 0.9
    assert informed.inventory_pressure == 0.6


def test_market_maker_side_specific_padding_overrides_symmetric_padding() -> None:
    config = SimulationConfig(
        simulation_id=SimulationId("sim_maker_padding_precedence"),
        market=default_market_config(),
        agents=(
            AgentConfig(
                agent_id=AgentId("maker_01"),
                agent_type="market_maker",
                starting_cash=10_000.0,
                ruin_threshold=4_000.0,
                max_resting_orders=3,
                behavior=AgentBehaviorConfig(
                    market_maker=MarketMakerBehaviorConfig(
                        quote_padding_ticks=6,
                        bid_padding_ticks=2,
                        ask_padding_ticks=4,
                    ),
                ),
            ),
        ),
        seed=11,
        enable_news=False,
        enable_private_signals=True,
        public_tape_enabled=True,
    )

    simulator = SyntheticMarketSimulator(config, horizon=8)
    maker = simulator.agents["maker_01"]

    assert maker.quote_padding_ticks == 6
    assert maker.bid_padding_ticks == 2
    assert maker.ask_padding_ticks == 4
