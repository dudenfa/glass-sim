"""Small building blocks shared by the scripted agents."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..types import Action, MarketOrder, Side
from ..views import MarketView


def ema(values: Sequence[float], span: float) -> Optional[float]:
    """Exponential moving average with ``span`` samples (most recent last)."""
    if not values:
        return None
    alpha = 2.0 / (span + 1.0)
    value = float(values[0])
    for v in values[1:]:
        value += alpha * (float(v) - value)
    return value


def realized_vol_per_second(prices: Sequence[float]) -> float:
    """Standard deviation of per-sample log returns (samples are one record interval apart)."""
    clean = [p for p in prices if p and p > 0]
    if len(clean) < 3:
        return 0.0
    returns = np.diff(np.log(np.asarray(clean, dtype=float)))
    return float(returns.std())


def move_toward(view: MarketView, target: int, max_clip: int, tag: str = "") -> list[Action]:
    """Market orders that move the position toward ``target`` by at most ``max_clip`` contracts.

    Reducing trades are sent ``reduce_only`` so they never fail on margin.
    """
    position = view.position
    delta = int(target) - position
    if delta == 0 or max_clip <= 0:
        return []
    side = Side.BUY if delta > 0 else Side.SELL
    quantity = min(abs(delta), max_clip)
    if side is Side.BUY and view.best_ask is None:
        return []
    if side is Side.SELL and view.best_bid is None:
        return []
    reducing = position != 0 and (position > 0) != (side is Side.BUY)
    if reducing:
        reduce_qty = min(quantity, abs(position))
        actions: list[Action] = [MarketOrder(side, reduce_qty, reduce_only=True, tag=tag)]
        if quantity > reduce_qty:
            actions.append(MarketOrder(side, quantity - reduce_qty, tag=tag))
        return actions
    return [MarketOrder(side, quantity, tag=tag)]


def capacity(view: MarketView, leverage: float) -> int:
    """Contracts the account's equity supports at ``leverage`` (at the mark price)."""
    return view.max_position(leverage=leverage)


def clamp(value: float, low: float, high: float) -> float:
    """Clip ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def sign(value: float) -> int:
    """-1, 0 or 1."""
    return 1 if value > 0 else (-1 if value < 0 else 0)


def safe_log_ratio(a: Optional[float], b: Optional[float]) -> float:
    """``log(a / b)``, or 0 when either side is missing or non-positive."""
    if not a or not b or a <= 0 or b <= 0:
        return 0.0
    return math.log(a / b)
