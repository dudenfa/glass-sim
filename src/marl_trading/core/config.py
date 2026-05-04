from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from marl_trading.core.domain import AgentId, AssetSymbol, SimulationId


@dataclass(frozen=True)
class MarketConfig:
    symbol: AssetSymbol
    starting_mid_price: float = 100.0
    tick_size: float = 0.01
    initial_spread: float = 0.02
    max_order_levels: int = 10
    event_horizon: int = 10_000
    news_impact_scale: float = 0.6
    fundamental_news_sensitivity: float = 0.9


@dataclass(frozen=True)
class MarketMakerBehaviorConfig:
    inventory_anchor: float | None = None
    quote_size: int | None = None
    quote_padding_ticks: int | None = None
    inventory_tolerance: float | None = None
    min_quote_size: int | None = None
    max_quote_size: int | None = None
    bid_padding_ticks: int | None = None
    ask_padding_ticks: int | None = None
    inventory_skew_strength: float | None = None
    inventory_size_decay: float | None = None
    empty_side_padding_ticks: int | None = None


@dataclass(frozen=True)
class NoiseTraderBehaviorConfig:
    aggressiveness: float | None = None
    market_order_probability: float | None = None
    sell_bias: float | None = None
    inventory_recycling_bias: float | None = None
    overpricing_sell_bias: float | None = None
    profit_taking_bias: float | None = None


@dataclass(frozen=True)
class TrendFollowerBehaviorConfig:
    threshold_bps: float | None = None
    market_order_probability: float | None = None
    exit_threshold_bps: float | None = None
    overpricing_exit_bias: float | None = None
    inventory_pressure: float | None = None


@dataclass(frozen=True)
class InformedTraderBehaviorConfig:
    private_signal_strength: float | None = None
    signal_noise: float | None = None
    news_bias: float | None = None
    threshold_bps: float | None = None
    sell_bias: float | None = None
    negative_news_sell_bias: float | None = None
    inventory_pressure: float | None = None


@dataclass(frozen=True)
class AgentBehaviorConfig:
    market_maker: MarketMakerBehaviorConfig | None = None
    noise_trader: NoiseTraderBehaviorConfig | None = None
    trend_follower: TrendFollowerBehaviorConfig | None = None
    informed_trader: InformedTraderBehaviorConfig | None = None


@dataclass(frozen=True)
class AgentConfig:
    agent_id: AgentId
    agent_type: str
    starting_cash: float
    ruin_threshold: float
    max_resting_orders: int = 3
    private_signal_strength: float = 0.0
    behavior: AgentBehaviorConfig | None = None


@dataclass(frozen=True)
class SimulationConfig:
    simulation_id: SimulationId
    market: MarketConfig
    agents: Tuple[AgentConfig, ...] = field(default_factory=tuple)
    seed: int = 0
    enable_news: bool = True
    enable_private_signals: bool = True
    public_tape_enabled: bool = True
