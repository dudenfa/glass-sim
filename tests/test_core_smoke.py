from marl_trading import AgentId, AssetSymbol, MarketConfig, SimulationConfig
from marl_trading.configs import default_simulation_config


def test_package_exports_and_defaults() -> None:
    market = MarketConfig(symbol=AssetSymbol("SYNTH"))
    sim = SimulationConfig(
        simulation_id=default_simulation_config().simulation_id,
        market=market,
    )

    assert AgentId("maker_01").value == "maker_01"
    assert sim.market.symbol.value == "SYNTH"


def test_default_simulation_config_is_well_formed() -> None:
    config = default_simulation_config()

    assert config.market.symbol.value == "SYNTH"
    assert len(config.agents) >= 4
    assert config.public_tape_enabled is True
