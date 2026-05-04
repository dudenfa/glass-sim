"""Named configuration presets for the synthetic market simulator."""

from marl_trading.configs.defaults import (
    default_agent_configs,
    default_market_config,
    default_simulation_config,
)
from marl_trading.configs.presets import (
    PRESETS,
    PresetDefinition,
    available_preset_names,
    baseline_preset,
    baseline_trend_duo_preset,
    build_preset_config,
    fragile_liquidity_preset,
    get_preset,
    high_information_asymmetry_preset,
    high_news_preset,
)

__all__ = [
    "PRESETS",
    "PresetDefinition",
    "available_preset_names",
    "baseline_preset",
    "baseline_trend_duo_preset",
    "build_preset_config",
    "default_agent_configs",
    "default_market_config",
    "default_simulation_config",
    "fragile_liquidity_preset",
    "get_preset",
    "high_information_asymmetry_preset",
    "high_news_preset",
]
