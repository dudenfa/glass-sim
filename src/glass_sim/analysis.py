"""Market-quality metrics and stylized facts computed from recorded bars.

Stylized facts are the statistical regularities shared by real financial
markets (Cont, 2001). The functions here let you check a simulation against
them:

- heavy tails: excess kurtosis of returns is positive and decreases as the
  sampling horizon grows (aggregational Gaussianity);
- no linear predictability: autocorrelation of returns is close to zero;
- volatility clustering: autocorrelation of absolute returns is positive and
  decays slowly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def log_returns(prices: Sequence[float], step: int = 1) -> np.ndarray:
    values = np.asarray([p for p in prices if p is not None and p > 0], dtype=float)
    if len(values) <= step:
        return np.zeros(0)
    sampled = values[::step]
    return np.diff(np.log(sampled))


def autocorrelation(series: np.ndarray, lag: int) -> Optional[float]:
    if len(series) <= lag + 2:
        return None
    x = series[:-lag] - series[:-lag].mean()
    y = series[lag:] - series[lag:].mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denominator == 0:
        return None
    return float((x * y).sum() / denominator)


def excess_kurtosis(series: np.ndarray) -> Optional[float]:
    if len(series) < 8:
        return None
    centered = series - series.mean()
    variance = float((centered**2).mean())
    if variance == 0:
        return None
    return float((centered**4).mean() / variance**2 - 3.0)


def stylized_facts(prices: Sequence[float], record_interval: float = 1.0, horizons_seconds: Sequence[int] = (10, 60, 300)) -> Dict[str, Dict[str, Optional[float]]]:
    """Kurtosis and autocorrelations of returns at several sampling horizons."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for horizon in horizons_seconds:
        step = max(1, int(round(horizon / record_interval)))
        returns = log_returns(prices, step)
        nonzero = returns[returns != 0]
        out[f"{horizon}s"] = {
            "n_returns": float(len(returns)),
            "zero_return_fraction": float(1 - len(nonzero) / len(returns)) if len(returns) else None,
            "std_bps": float(returns.std() * 1e4) if len(returns) > 1 else None,
            "excess_kurtosis": excess_kurtosis(returns),
            "acf_return_lag1": autocorrelation(returns, 1),
            "acf_abs_return_lag1": autocorrelation(np.abs(returns), 1),
            "acf_abs_return_lag5": autocorrelation(np.abs(returns), 5),
            "acf_abs_return_lag20": autocorrelation(np.abs(returns), 20),
        }
    return out


def _clean(values: Sequence[Optional[float]]) -> np.ndarray:
    return np.asarray([v for v in values if v is not None], dtype=float)


def market_summary(bars: Dict[str, List[Optional[float]]], record_interval: float = 1.0) -> Dict[str, Optional[float]]:
    """Liquidity, price-discovery and activity metrics from the recorded bars."""
    mids = bars["mid"]
    spreads = _clean(bars["spread"])
    mid_values = _clean(mids)
    n = len(bars["time"])
    spread_bps = spreads / mid_values.mean() * 1e4 if len(spreads) and len(mid_values) else np.zeros(0)
    mark = np.asarray(bars["mark"], dtype=float)
    index = np.asarray(bars["index"], dtype=float)
    close = np.asarray(bars["close"], dtype=float)
    dislocation_bps = (mark - index) / index * 1e4 if len(index) else np.zeros(0)
    last_vs_index_bps = (close - index) / index * 1e4 if len(index) else np.zeros(0)
    returns = log_returns(bars["close"], max(1, int(60 / record_interval)))
    minutes = n * record_interval / 60.0
    return {
        "bars": float(n),
        "trades": float(sum(bars["trades"])),
        "volume_contracts": float(sum(bars["volume"])),
        "trades_per_minute": float(sum(bars["trades"]) / minutes) if minutes else None,
        "two_sided_book_fraction": float(len(spreads) / n) if n else None,
        "mean_spread": float(spreads.mean()) if len(spreads) else None,
        "median_spread_bps": float(np.median(spread_bps)) if len(spread_bps) else None,
        "mean_bid_depth": float(np.mean(bars["bid_depth"])) if n else None,
        "mean_ask_depth": float(np.mean(bars["ask_depth"])) if n else None,
        "mark_index_mean_abs_bps": float(np.mean(np.abs(dislocation_bps))) if len(dislocation_bps) else None,
        "last_index_mean_abs_bps": float(np.mean(np.abs(last_vs_index_bps))) if len(last_vs_index_bps) else None,
        "last_index_max_abs_bps": float(np.max(np.abs(last_vs_index_bps))) if len(last_vs_index_bps) else None,
        "return_vol_1m_bps": float(returns.std() * 1e4) if len(returns) > 1 else None,
        "final_last": float(close[-1]) if len(close) else None,
        "final_mark": float(mark[-1]) if len(mark) else None,
        "final_index": float(index[-1]) if len(index) else None,
        "mean_open_interest": float(np.mean(bars["open_interest"])) if n else None,
        "max_open_interest": float(np.max(bars["open_interest"])) if n else None,
    }
