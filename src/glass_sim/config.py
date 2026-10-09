"""Configuration objects for a perpetual-futures market simulation.

Everything a researcher may want to vary is a plain dataclass field, so a
scenario is just a ``MarketConfig`` value and experiments can be expressed as
``dataclasses.replace(config, transparency=Transparency.ANONYMOUS)``.

Time is measured in simulated seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .types import Transparency

HOUR = 3600.0
DAY = 24 * HOUR


@dataclass(frozen=True)
class ContractSpec:
    """The traded instrument: a linear perpetual future settled in quote currency."""

    symbol: str = "SYNTH-PERP"
    tick_size: float = 0.01
    contract_size: float = 1.0
    initial_price: float = 100.0
    max_leverage: float = 40.0
    maintenance_margin_ratio: Optional[float] = None
    maker_fee: float = 0.00015
    taker_fee: float = 0.00045

    @property
    def initial_margin_ratio(self) -> float:
        return 1.0 / self.max_leverage

    @property
    def mmr(self) -> float:
        # Hyperliquid convention: maintenance margin is half the initial margin at max leverage.
        if self.maintenance_margin_ratio is not None:
            return self.maintenance_margin_ratio
        return 0.5 / self.max_leverage


@dataclass(frozen=True)
class IndexConfig:
    """The index (oracle) price: the spot reference the perpetual is anchored to.

    Log-price follows a diffusion whose volatility is itself stochastic
    (log-volatility mean-reverts), which produces volatility clustering in the
    reference price. News jumps are added on top by ``NewsConfig``.
    """

    daily_volatility: float = 0.03
    daily_drift: float = 0.0
    update_interval: float = 1.0
    vol_mean_reversion: float = 1.0 / 1800.0
    vol_of_vol: float = 0.02
    enabled: bool = True


@dataclass(frozen=True)
class NewsConfig:
    """Public news shocks and the private signals informed agents receive early.

    News arrives as a Poisson process. Each item moves the index by a log jump.
    Agents subscribed to private signals learn about an item ``lead_time``
    seconds before it becomes public, with a noisy estimate of its size.
    """

    rate_per_hour: float = 2.0
    jump_std: float = 0.006
    lead_time: Tuple[float, float] = (20.0, 90.0)
    signal_noise: float = 0.35
    enabled: bool = True


@dataclass(frozen=True)
class FundingConfig:
    """Perpetual funding: longs pay shorts when the perp trades above the index.

    Every ``sample_interval`` the premium ``(mark - index) / index`` is sampled.
    Every ``interval`` the funding rate ``clip(mean_premium + interest_rate, ±cap)``
    is applied: each account pays ``position * contract_size * index * rate``.
    """

    interval: float = HOUR
    sample_interval: float = 5.0
    interest_rate: float = 0.0000125
    cap: float = 0.005
    enabled: bool = True


@dataclass(frozen=True)
class LiquidationConfig:
    """What happens when equity drops below the maintenance requirement.

    The account's orders are canceled and its position is closed with a market
    order. A penalty (fraction of closed notional, capped at the remaining
    equity) goes to the insurance fund. If the book cannot absorb the whole
    position, the remainder is auto-deleveraged against the most profitable
    opposite positions at the mark price. Negative equity (bad debt) is covered
    by the insurance fund, which may go negative.
    """

    penalty_rate: float = 0.01
    insurance_fund: float = 250_000.0
    deactivate_below_fraction: float = 0.05
    enabled: bool = True


@dataclass(frozen=True)
class LatencyConfig:
    """Delay between an agent deciding and its message reaching the exchange."""

    base: float = 0.05
    jitter: float = 0.05


CollateralSpec = Union[float, Tuple[float, float]]


@dataclass(frozen=True)
class AgentSpec:
    """A group of identical-type agents.

    ``agent`` is an ``Agent`` subclass (or any callable returning an Agent).
    ``collateral`` is either a fixed amount or a ``(low, high)`` range sampled
    log-uniformly per agent. ``params`` are passed as keyword arguments to the
    agent constructor; a value that is a ``(low, high)`` tuple under a key listed
    in ``randomize`` is sampled uniformly per agent, giving heterogeneous agents.
    """

    agent: Callable[..., Any]
    count: int = 1
    collateral: CollateralSpec = 10_000.0
    params: Dict[str, Any] = field(default_factory=dict)
    randomize: Tuple[str, ...] = ()
    prefix: Optional[str] = None
    private_signals: bool = False
    external: bool = False


@dataclass(frozen=True)
class MarketConfig:
    contract: ContractSpec = field(default_factory=ContractSpec)
    index: IndexConfig = field(default_factory=IndexConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    liquidation: LiquidationConfig = field(default_factory=LiquidationConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    transparency: Transparency = Transparency.FULL
    agents: Tuple[AgentSpec, ...] = ()
    duration: float = 4 * HOUR
    seed: int = 0
    record_interval: float = 1.0
    pseudonymous_ids: bool = False
    recent_trades_kept: int = 2_000
    name: str = "custom"

    def with_agents(self, *specs: AgentSpec) -> "MarketConfig":
        from dataclasses import replace

        return replace(self, agents=tuple(specs))


def parse_duration(value: Union[str, float, int]) -> float:
    """Parse ``"90s"``, ``"30m"``, ``"4h"``, ``"1d"`` or a number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    units = {"s": 1.0, "m": 60.0, "h": HOUR, "d": DAY}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def format_time(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


__all__: List[str] = [
    "AgentSpec",
    "ContractSpec",
    "DAY",
    "FundingConfig",
    "HOUR",
    "IndexConfig",
    "LatencyConfig",
    "LiquidationConfig",
    "MarketConfig",
    "NewsConfig",
    "format_time",
    "parse_duration",
]
