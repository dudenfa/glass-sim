"""Ready-made market configurations.

``default()`` is the reference ecology. The other scenarios change one thing
at a time relative to it, so they can be compared directly:

- ``anonymous()`` / ``attributed()``: same ecology, lower transparency;
- ``calm()`` / ``volatile()``: different index volatility and news intensity;
- ``retail_frenzy()``: more, higher-leverage retail traders;
- ``no_arbitrage()``: removes the basis arbitrageurs that anchor perp to index.

All scenario functions accept keyword overrides for any ``MarketConfig`` field,
e.g. ``scenarios.default(seed=3, duration=parse_duration("1h"))``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Tuple

from .agents import (
    BasisArbitrageur,
    CopyTrader,
    FlowFollower,
    InformedTrader,
    LiquidationHunter,
    MarketMaker,
    MomentumTrader,
    NoiseTrader,
    Whale,
)
from .config import AgentSpec, HOUR, IndexConfig, MarketConfig, NewsConfig
from .types import Transparency


def default_population(retail: int = 60, retail_leverage: Tuple[float, float] = (3.0, 40.0)) -> Tuple[AgentSpec, ...]:
    return (
        AgentSpec(
            MarketMaker,
            count=3,
            collateral=(300_000.0, 600_000.0),
            params={"wake_interval": (1.0, 2.0), "level_fraction": (0.01, 0.02), "min_half_spread_bps": (1.5, 3.0), "skew_bps": (8.0, 16.0)},
            randomize=("wake_interval", "level_fraction", "min_half_spread_bps", "skew_bps"),
        ),
        AgentSpec(
            NoiseTrader,
            count=retail,
            collateral=(1_000.0, 20_000.0),
            params={
                "wake_interval": (45.0, 180.0),
                "leverage": retail_leverage,
                "mean_holding": (600.0, 3600.0),
                "trade_probability": (0.2, 0.5),
                "take_profit": (0.1, 0.6),
                "chase_bias": (0.0, 0.4),
            },
            randomize=("wake_interval", "leverage", "mean_holding", "trade_probability", "take_profit", "chase_bias"),
        ),
        AgentSpec(
            MomentumTrader,
            count=8,
            collateral=(20_000.0, 80_000.0),
            params={"wake_interval": (20.0, 60.0), "fast_seconds": (120.0, 300.0), "slow_seconds": (900.0, 1800.0), "leverage": (1.0, 3.0)},
            randomize=("wake_interval", "fast_seconds", "slow_seconds", "leverage"),
        ),
        AgentSpec(
            BasisArbitrageur,
            count=4,
            collateral=(100_000.0, 250_000.0),
            params={"wake_interval": (3.0, 8.0), "entry_bps": (8.0, 20.0), "leverage": (2.0, 4.0)},
            randomize=("wake_interval", "entry_bps", "leverage"),
        ),
        AgentSpec(
            InformedTrader,
            count=2,
            collateral=(50_000.0, 150_000.0),
            params={"leverage": (2.0, 5.0)},
            randomize=("leverage",),
            private_signals=True,
        ),
        AgentSpec(Whale, count=1, collateral=(1_000_000.0, 2_000_000.0)),
        AgentSpec(
            FlowFollower,
            count=3,
            collateral=(20_000.0, 60_000.0),
            params={"wake_interval": (10.0, 30.0), "leverage": (2.0, 5.0), "min_flow": (800, 2_000)},
            randomize=("wake_interval", "leverage", "min_flow"),
        ),
        AgentSpec(
            CopyTrader,
            count=6,
            collateral=(5_000.0, 30_000.0),
            params={"wake_interval": (20.0, 60.0), "leverage": (2.0, 8.0)},
            randomize=("wake_interval", "leverage"),
        ),
        AgentSpec(LiquidationHunter, count=1, collateral=(200_000.0, 400_000.0)),
    )


def default(**overrides) -> MarketConfig:
    """Reference ecology: 88 agents of nine types, FULL transparency, 4 hours."""
    config = MarketConfig(agents=default_population(), transparency=Transparency.FULL, duration=4 * HOUR, name="default")
    return replace(config, **overrides)


def anonymous(**overrides) -> MarketConfig:
    """Reference ecology with an anonymous public feed (classic exchange)."""
    return default(transparency=Transparency.ANONYMOUS, name="anonymous", **overrides)


def attributed(**overrides) -> MarketConfig:
    """Reference ecology where orders and trades show their owner, positions stay private."""
    return default(transparency=Transparency.ATTRIBUTED, name="attributed", **overrides)


def calm(**overrides) -> MarketConfig:
    """Half the index volatility and a quarter of the news intensity."""
    return default(index=IndexConfig(daily_volatility=0.015, vol_of_vol=0.01), news=NewsConfig(rate_per_hour=0.5, jump_std=0.003), name="calm", **overrides)


def volatile(**overrides) -> MarketConfig:
    """Double the index volatility and news intensity, larger news jumps."""
    return default(index=IndexConfig(daily_volatility=0.06, vol_of_vol=0.04), news=NewsConfig(rate_per_hour=4.0, jump_std=0.012), name="volatile", **overrides)


def retail_frenzy(**overrides) -> MarketConfig:
    """Twice as many retail traders, all at 20-40x leverage."""
    return default(agents=default_population(retail=120, retail_leverage=(20.0, 40.0)), name="retail_frenzy", **overrides)


def no_arbitrage(**overrides) -> MarketConfig:
    """Reference ecology without the basis arbitrageurs that anchor perp to index."""
    agents = tuple(spec for spec in default_population() if spec.agent is not BasisArbitrageur)
    return default(agents=agents, name="no_arbitrage", **overrides)


def small(**overrides) -> MarketConfig:
    """A fast, small market for tests and quick experiments (~20 agents, 30 minutes)."""
    agents = (
        AgentSpec(MarketMaker, count=2, collateral=300_000.0),
        AgentSpec(NoiseTrader, count=12, collateral=(2_000.0, 10_000.0), params={"wake_interval": 60.0, "leverage": (2.0, 15.0)}, randomize=("leverage",)),
        AgentSpec(MomentumTrader, count=2, collateral=40_000.0),
        AgentSpec(BasisArbitrageur, count=2, collateral=150_000.0),
        AgentSpec(InformedTrader, count=1, collateral=80_000.0, private_signals=True),
        AgentSpec(CopyTrader, count=2, collateral=10_000.0),
    )
    config = MarketConfig(agents=agents, duration=0.5 * HOUR, name="small")
    return replace(config, **overrides)


SCENARIOS: Dict[str, Callable[..., MarketConfig]] = {
    "default": default,
    "anonymous": anonymous,
    "attributed": attributed,
    "calm": calm,
    "volatile": volatile,
    "retail_frenzy": retail_frenzy,
    "no_arbitrage": no_arbitrage,
    "small": small,
}


def get(name: str, **overrides) -> MarketConfig:
    try:
        factory = SCENARIOS[name]
    except KeyError as exc:
        raise KeyError(f"unknown scenario {name!r}; available: {', '.join(sorted(SCENARIOS))}") from exc
    return factory(**overrides)


def names() -> List[str]:
    return sorted(SCENARIOS)
