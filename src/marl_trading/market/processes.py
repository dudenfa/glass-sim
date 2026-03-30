from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NewsEvent:
    headline: str
    severity: float
    impact: float


@dataclass
class FundamentalProcess:
    current_value: float
    drift_per_step: float = 0.002
    volatility_per_step: float = 0.08
    news_sensitivity: float = 0.9
    floor_value: float = 1.0

    def advance(self, rng: np.random.Generator, news_impact: float = 0.0) -> float:
        shock = float(rng.normal(0.0, self.volatility_per_step))
        self.current_value = max(
            self.floor_value,
            float(self.current_value + self.drift_per_step + shock + self.news_sensitivity * news_impact),
        )
        return float(self.current_value)


@dataclass
class PublicNewsProcess:
    interval_steps: int = 45
    impact_scale: float = 0.6
    headlines: tuple[str, ...] = (
        "Macro liquidity surprise",
        "Exchange flow imbalance",
        "Risk appetite shift",
        "Informed desks reprice risk",
        "Volatility pocket appears",
        "Fresh demand wave",
    )

    def maybe_emit(self, step_index: int, rng: np.random.Generator) -> NewsEvent | None:
        if step_index <= 0 or self.interval_steps <= 0:
            return None
        if step_index % self.interval_steps != 0:
            return None

        headline = self.headlines[(step_index // self.interval_steps) % len(self.headlines)]
        direction = 1.0 if rng.random() < 0.5 else -1.0
        severity = direction * float(0.5 + rng.random())
        impact = severity * self.impact_scale
        return NewsEvent(headline=headline, severity=severity, impact=impact)
