"""Agents for the perpetual-futures market: the base interface and scripted traders."""

from .base import Agent, ExternalAgent
from .scripted import (
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

__all__ = [
    "Agent",
    "BasisArbitrageur",
    "CopyTrader",
    "ExternalAgent",
    "FlowFollower",
    "InformedTrader",
    "LiquidationHunter",
    "MarketMaker",
    "MomentumTrader",
    "NoiseTrader",
    "Whale",
]
