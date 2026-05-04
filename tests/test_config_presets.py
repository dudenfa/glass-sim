from __future__ import annotations

from marl_trading.configs import (
    available_preset_names,
    baseline_preset,
    build_preset_config,
    fragile_liquidity_preset,
    high_information_asymmetry_preset,
    high_news_preset,
)
from marl_trading.core.config import (
    AgentBehaviorConfig,
    InformedTraderBehaviorConfig,
    MarketMakerBehaviorConfig,
    NoiseTraderBehaviorConfig,
    SimulationConfig,
    TrendFollowerBehaviorConfig,
)
from marl_trading.core.domain import AgentId


def test_named_presets_build_valid_simulation_configs() -> None:
    names = available_preset_names()
    assert names == (
        "baseline",
        "high_news",
        "fragile_liquidity",
        "high_information_asymmetry",
    )

    builders = {
        "baseline": baseline_preset,
        "high_news": high_news_preset,
        "fragile_liquidity": fragile_liquidity_preset,
        "high_information_asymmetry": high_information_asymmetry_preset,
    }

    for name, builder in builders.items():
        config = builder()
        assert isinstance(config, SimulationConfig)
        assert config.market.symbol.value == "SYNTH"
        assert config.market.tick_size > 0
        assert len(config.agents) >= 4
        assert all(isinstance(agent.agent_id, AgentId) for agent in config.agents)
        assert build_preset_config(name) == config


def test_preset_differences_are_intentional() -> None:
    baseline = build_preset_config("baseline")
    high_news = build_preset_config("high_news")
    fragile = build_preset_config("fragile_liquidity")
    high_info = build_preset_config("high_information_asymmetry")

    assert high_news.market.event_horizon == baseline.market.event_horizon
    assert high_news.enable_news is True
    assert high_news.market.news_impact_scale > baseline.market.news_impact_scale
    assert high_news.market.fundamental_news_sensitivity > baseline.market.fundamental_news_sensitivity
    high_news_informed = next(agent for agent in high_news.agents if agent.agent_id.value == "informed_01")
    assert isinstance(high_news_informed.behavior, AgentBehaviorConfig)
    assert isinstance(high_news_informed.behavior.informed_trader, InformedTraderBehaviorConfig)
    assert high_news_informed.behavior.informed_trader.news_bias == 2.1
    assert high_news_informed.behavior.informed_trader.signal_noise == 0.08
    assert high_news_informed.behavior.informed_trader.threshold_bps == 0.7

    assert fragile.market.event_horizon == baseline.market.event_horizon
    assert fragile.market.max_order_levels < baseline.market.max_order_levels
    assert fragile.market.initial_spread > baseline.market.initial_spread
    fragile_maker = next(agent for agent in fragile.agents if agent.agent_id.value == "maker_01")
    fragile_noise = next(agent for agent in fragile.agents if agent.agent_id.value == "retail_01")
    fragile_trend = next(agent for agent in fragile.agents if agent.agent_id.value == "trend_01")
    assert fragile_maker.max_resting_orders == 1
    assert isinstance(fragile_maker.behavior, AgentBehaviorConfig)
    assert isinstance(fragile_maker.behavior.market_maker, MarketMakerBehaviorConfig)
    assert fragile_maker.behavior.market_maker.quote_size == 1
    assert fragile_maker.behavior.market_maker.quote_padding_ticks == 3
    assert fragile_noise.max_resting_orders == 1
    assert isinstance(fragile_noise.behavior, AgentBehaviorConfig)
    assert isinstance(fragile_noise.behavior.noise_trader, NoiseTraderBehaviorConfig)
    assert fragile_noise.behavior.noise_trader.aggressiveness == 0.8
    assert fragile_noise.behavior.noise_trader.market_order_probability == 0.9
    assert fragile_trend.max_resting_orders == 1
    assert isinstance(fragile_trend.behavior, AgentBehaviorConfig)
    assert isinstance(fragile_trend.behavior.trend_follower, TrendFollowerBehaviorConfig)
    assert fragile_trend.behavior.trend_follower.threshold_bps == 0.8
    assert fragile_trend.behavior.trend_follower.market_order_probability == 0.85

    assert high_info.public_tape_enabled is False
    assert sum(agent.agent_type == "informed_trader" for agent in high_info.agents) >= 2
    assert any(agent.agent_id.value == "informed_02" for agent in high_info.agents)
    high_info_informed = [agent for agent in high_info.agents if agent.agent_type == "informed_trader"]
    assert any(agent.private_signal_strength > 1.0 for agent in high_info_informed)
    primary_informed = next(agent for agent in high_info_informed if agent.agent_id.value == "informed_01")
    assert primary_informed.private_signal_strength == 1.8
    assert isinstance(primary_informed.behavior, AgentBehaviorConfig)
    assert isinstance(primary_informed.behavior.informed_trader, InformedTraderBehaviorConfig)
    assert primary_informed.behavior.informed_trader.private_signal_strength == 2.4
    assert any(
        isinstance(agent.behavior, AgentBehaviorConfig)
        and isinstance(agent.behavior.informed_trader, InformedTraderBehaviorConfig)
        and agent.behavior.informed_trader.private_signal_strength is not None
        and agent.behavior.informed_trader.private_signal_strength >= 2.4
        for agent in high_info_informed
    )
    assert any(
        isinstance(agent.behavior, AgentBehaviorConfig)
        and isinstance(agent.behavior.informed_trader, InformedTraderBehaviorConfig)
        and agent.behavior.informed_trader.signal_noise is not None
        and agent.behavior.informed_trader.signal_noise <= 0.12
        for agent in high_info_informed
    )


def test_unknown_preset_raises_key_error() -> None:
    try:
        build_preset_config("not-a-preset")
    except KeyError:
        return
    raise AssertionError("Expected build_preset_config to raise KeyError for an unknown preset.")
