from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final

from marl_trading.configs.defaults import default_simulation_config
from marl_trading.core.config import (
    AgentBehaviorConfig,
    AgentConfig,
    InformedTraderBehaviorConfig,
    MarketMakerBehaviorConfig,
    NoiseTraderBehaviorConfig,
    SimulationConfig,
    TrendFollowerBehaviorConfig,
)
from marl_trading.core.domain import AgentId

PresetBuilder = Callable[[], SimulationConfig]


@dataclass(frozen=True)
class PresetDefinition:
    name: str
    description: str
    build: PresetBuilder

    def __call__(self) -> SimulationConfig:
        return self.build()


def _normalized_name(name: str) -> str:
    return name.strip().lower()


def _with_market(base: SimulationConfig, **changes: object) -> SimulationConfig:
    return replace(base, market=replace(base.market, **changes))


def _with_agent_updates(
    base: SimulationConfig,
    updates: Mapping[str, Mapping[str, object]],
) -> SimulationConfig:
    updated_agents: list[AgentConfig] = []
    for agent in base.agents:
        agent_updates = updates.get(agent.agent_id.value)
        if agent_updates:
            updated_agents.append(replace(agent, **agent_updates))
        else:
            updated_agents.append(agent)
    return replace(base, agents=tuple(updated_agents))


def baseline_preset() -> SimulationConfig:
    return default_simulation_config()


def high_news_preset() -> SimulationConfig:
    base = default_simulation_config()
    return _with_agent_updates(
        _with_market(
            replace(base, seed=11),
            event_horizon=320,
        ),
        {
            "informed_01": {
                "behavior": AgentBehaviorConfig(
                    informed_trader=InformedTraderBehaviorConfig(
                        news_bias=1.45,
                        threshold_bps=0.85,
                    )
                )
            }
        },
    )


def fragile_liquidity_preset() -> SimulationConfig:
    base = default_simulation_config()
    return _with_agent_updates(
        _with_market(
            replace(base, seed=13),
            initial_spread=0.06,
            max_order_levels=4,
            event_horizon=1_200,
        ),
        {
            "maker_01": {
                "max_resting_orders": 1,
                "behavior": AgentBehaviorConfig(
                    market_maker=MarketMakerBehaviorConfig(
                        inventory_anchor=24.0,
                        quote_size=1,
                        quote_padding_ticks=3,
                    )
                ),
            },
            "retail_01": {
                "max_resting_orders": 1,
                "behavior": AgentBehaviorConfig(
                    noise_trader=NoiseTraderBehaviorConfig(
                        aggressiveness=0.8,
                        market_order_probability=0.9,
                    )
                ),
            },
            "informed_01": {
                "max_resting_orders": 1,
                "behavior": AgentBehaviorConfig(
                    informed_trader=InformedTraderBehaviorConfig(
                        threshold_bps=0.85,
                        signal_noise=0.25,
                    )
                ),
            },
            "trend_01": {
                "max_resting_orders": 1,
                "behavior": AgentBehaviorConfig(
                    trend_follower=TrendFollowerBehaviorConfig(
                        threshold_bps=0.8,
                        market_order_probability=0.85,
                    )
                ),
            },
        },
    )


def high_information_asymmetry_preset() -> SimulationConfig:
    base = default_simulation_config()
    return _with_agent_updates(
        replace(
            base,
            seed=29,
            enable_private_signals=True,
            public_tape_enabled=False,
        ),
        {
            "retail_01": {
                "starting_cash": 8_500.0,
                "ruin_threshold": 3_400.0,
                "max_resting_orders": 1,
                "behavior": AgentBehaviorConfig(
                    noise_trader=NoiseTraderBehaviorConfig(
                        aggressiveness=0.45,
                        market_order_probability=0.55,
                    )
                ),
            },
            "informed_01": {
                "starting_cash": 12_500.0,
                "ruin_threshold": 5_000.0,
                "private_signal_strength": 1.8,
                "behavior": AgentBehaviorConfig(
                    informed_trader=InformedTraderBehaviorConfig(
                        private_signal_strength=2.4,
                        signal_noise=0.08,
                        news_bias=1.8,
                        threshold_bps=0.55,
                    )
                ),
            },
            "trend_01": {
                "agent_id": AgentId("informed_02"),
                "agent_type": "informed_trader",
                "starting_cash": 9_000.0,
                "ruin_threshold": 3_600.0,
                "max_resting_orders": 1,
                "private_signal_strength": 0.9,
                "behavior": AgentBehaviorConfig(
                    informed_trader=InformedTraderBehaviorConfig(
                        private_signal_strength=1.2,
                        signal_noise=0.12,
                        news_bias=1.35,
                        threshold_bps=0.75,
                    )
                ),
            },
        },
    )


PRESETS: Final[dict[str, PresetDefinition]] = {
    "baseline": PresetDefinition(
        name="baseline",
        description="Default balanced market with the current simulator settings.",
        build=baseline_preset,
    ),
    "high_news": PresetDefinition(
        name="high_news",
        description="Shorter horizon with a denser news cadence.",
        build=high_news_preset,
    ),
    "fragile_liquidity": PresetDefinition(
        name="fragile_liquidity",
        description="Thin book depth and lower resting-order capacity.",
        build=fragile_liquidity_preset,
    ),
    "high_information_asymmetry": PresetDefinition(
        name="high_information_asymmetry",
        description="Two informed agents, weaker public tape, and stronger private signals.",
        build=high_information_asymmetry_preset,
    ),
}


def available_preset_names() -> tuple[str, ...]:
    return tuple(PRESETS)


def get_preset(name: str) -> PresetDefinition:
    return PRESETS[_normalized_name(name)]


def build_preset_config(name: str) -> SimulationConfig:
    return get_preset(name).build()


__all__ = [
    "PresetDefinition",
    "PRESETS",
    "available_preset_names",
    "baseline_preset",
    "build_preset_config",
    "fragile_liquidity_preset",
    "get_preset",
    "high_information_asymmetry_preset",
    "high_news_preset",
]
