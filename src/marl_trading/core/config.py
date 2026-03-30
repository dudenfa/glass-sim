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


@dataclass(frozen=True)
class AgentConfig:
    agent_id: AgentId
    agent_type: str
    starting_cash: float
    ruin_threshold: float
    max_resting_orders: int = 3
    private_signal_strength: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    simulation_id: SimulationId
    market: MarketConfig
    agents: Tuple[AgentConfig, ...] = field(default_factory=tuple)
    seed: int = 0
    enable_news: bool = True
    enable_private_signals: bool = True
    public_tape_enabled: bool = True
