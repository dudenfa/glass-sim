"""Exogenous processes: the index (oracle) price, public news and private signals.

These processes use their own random stream, separate from every agent's
stream, so two experiments with the same seed face exactly the same index
path and news regardless of how agents behave. This makes paired
counterfactual comparisons (e.g. transparency on/off) valid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import DAY, IndexConfig, NewsConfig

HEADLINES = (
    "Macro data surprise",
    "Large fund rebalancing reported",
    "Regulatory headline",
    "Exchange listing rumor",
    "Risk-on flows across markets",
    "Risk-off flows across markets",
    "Protocol upgrade announced",
    "Security incident rumor",
    "Central bank remarks",
    "Liquidity provider pulls back",
)


class IndexProcess:
    """Stochastic-volatility diffusion in log price, plus externally applied jumps."""

    def __init__(self, config: IndexConfig, initial_price: float, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.log_price = math.log(initial_price)
        self.base_sigma = config.daily_volatility / math.sqrt(DAY)
        self.log_vol_multiplier = 0.0

    @property
    def price(self) -> float:
        """Current index price."""
        return math.exp(self.log_price)

    @property
    def current_sigma_per_second(self) -> float:
        """Instantaneous volatility per second."""
        return self.base_sigma * math.exp(self.log_vol_multiplier)

    def advance(self, dt: float) -> float:
        """Advance the diffusion by ``dt`` seconds and return the new price."""
        if not self.config.enabled or dt <= 0:
            return self.price
        kappa = self.config.vol_mean_reversion
        xi = self.config.vol_of_vol
        # Log-volatility multiplier: Ornstein-Uhlenbeck around 0 (i.e. around base sigma).
        self.log_vol_multiplier += -kappa * self.log_vol_multiplier * dt + xi * math.sqrt(dt) * float(self.rng.normal())
        self.log_vol_multiplier = max(-2.0, min(2.0, self.log_vol_multiplier))
        sigma = self.current_sigma_per_second
        drift = self.config.daily_drift / DAY
        self.log_price += (drift - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * float(self.rng.normal())
        return self.price

    def jump(self, log_jump: float) -> float:
        """Apply a news jump; volatility rises with it so the index is choppier after news, as in real markets."""
        self.log_price += log_jump
        # A shock also raises volatility for a while, like real markets after news.
        self.log_vol_multiplier = min(2.0, self.log_vol_multiplier + 4.0 * abs(log_jump) / max(self.config.daily_volatility, 1e-9))
        return self.price


@dataclass(frozen=True)
class NewsItem:
    """A public news event: a log jump applied to the index at ``time``."""

    news_id: int
    time: float
    headline: str
    log_jump: float


@dataclass(frozen=True)
class PrivateSignal:
    """What an informed agent learns ahead of a public news item."""

    news_id: int
    arrival_time: float
    public_time: float
    expected_log_jump: float


def generate_news(config: NewsConfig, duration: float, rng: np.random.Generator) -> list[NewsItem]:
    """Poisson news arrivals for the whole session, drawn up front from the news stream so agents cannot perturb them."""
    if not config.enabled or config.rate_per_hour <= 0:
        return []
    items: list[NewsItem] = []
    t = 0.0
    rate = config.rate_per_hour / 3600.0
    news_id = 0
    while True:
        t += float(rng.exponential(1.0 / rate))
        if t >= duration:
            break
        news_id += 1
        headline = HEADLINES[int(rng.integers(len(HEADLINES)))]
        items.append(NewsItem(news_id=news_id, time=t, headline=headline, log_jump=float(rng.normal(0.0, config.jump_std))))
    return items


def generate_signals(config: NewsConfig, news: list[NewsItem], rng: np.random.Generator) -> list[PrivateSignal]:
    """One private signal per news item, shared by all subscribed agents."""
    low, high = config.lead_time
    signals: list[PrivateSignal] = []
    for item in news:
        lead = float(rng.uniform(low, high))
        noise = float(rng.normal(0.0, config.signal_noise * config.jump_std))
        signals.append(
            PrivateSignal(
                news_id=item.news_id,
                arrival_time=max(0.0, item.time - lead),
                public_time=item.time,
                expected_log_jump=item.log_jump + noise,
            )
        )
    return signals


def sample_collateral(spec, rng: np.random.Generator) -> float:
    """A fixed amount, or log-uniform in a ``(low, high)`` range so small and large accounts are equally represented."""
    if isinstance(spec, (tuple, list)):
        low, high = float(spec[0]), float(spec[1])
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    return float(spec)


def optional_float(value) -> Optional[float]:
    """``float(value)`` or ``None``."""
    return None if value is None else float(value)
