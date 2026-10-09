from __future__ import annotations

from marl_trading.core.config import AgentConfig, MarketConfig, SimulationConfig
from marl_trading.core.domain import AgentId, AssetSymbol, SimulationId


def default_market_config() -> MarketConfig:
    return MarketConfig(
        symbol=AssetSymbol("SYNTH"),
        starting_mid_price=100.0,
        tick_size=0.01,
        initial_spread=0.02,
        max_order_levels=10,
        event_horizon=10_000,
    )


def default_agent_configs() -> tuple[AgentConfig, ...]:
    return (
        AgentConfig(
            agent_id=AgentId("maker_01"),
            agent_type="market_maker",
            starting_cash=10_000.0,
            ruin_threshold=4_000.0,
            max_resting_orders=3,
            private_signal_strength=0.0,
        ),
        AgentConfig(
            agent_id=AgentId("retail_01"),
            agent_type="noise_trader",
            starting_cash=10_000.0,
            ruin_threshold=4_000.0,
            max_resting_orders=2,
            private_signal_strength=0.0,
        ),
        AgentConfig(
            agent_id=AgentId("informed_01"),
            agent_type="informed_trader",
            starting_cash=10_000.0,
            ruin_threshold=4_000.0,
            max_resting_orders=2,
            private_signal_strength=1.0,
        ),
        AgentConfig(
            agent_id=AgentId("trend_01"),
            agent_type="trend_follower",
            starting_cash=10_000.0,
            ruin_threshold=4_000.0,
            max_resting_orders=2,
            private_signal_strength=0.0,
        ),
    )


def default_simulation_config() -> SimulationConfig:
    return SimulationConfig(
        simulation_id=SimulationId("sim_0001"),
        market=default_market_config(),
        agents=default_agent_configs(),
        seed=7,
        enable_news=True,
        enable_private_signals=True,
        public_tape_enabled=True,
    )
