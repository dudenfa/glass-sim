from __future__ import annotations

from typing import TYPE_CHECKING

from .processes import FundamentalProcess, NewsEvent, PublicNewsProcess
from .simulator import MarketRunResult, MarketStepRecord, SyntheticMarketSimulator, run_market_demo
from .visualization import plot_market_world


if TYPE_CHECKING:
    from marl_trading.live import LiveMarketSession, LiveServerConfig, MarketViewServer


def serve_market_view(config: "LiveServerConfig | None" = None) -> "MarketViewServer":
    from marl_trading.live import LiveServerConfig
    from marl_trading.live.server import serve_market_view as _serve_market_view

    return _serve_market_view(LiveServerConfig() if config is None else config)


def __getattr__(name: str):
    if name == "LiveMarketSession":
        from marl_trading.live import LiveMarketSession

        globals()[name] = LiveMarketSession
        return LiveMarketSession
    if name == "LiveServerConfig":
        from marl_trading.live import LiveServerConfig

        globals()[name] = LiveServerConfig
        return LiveServerConfig
    if name == "MarketViewServer":
        from marl_trading.live import MarketViewServer

        globals()[name] = MarketViewServer
        return MarketViewServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FundamentalProcess",
    "LiveMarketSession",
    "LiveServerConfig",
    "MarketRunResult",
    "MarketStepRecord",
    "MarketViewServer",
    "NewsEvent",
    "PublicNewsProcess",
    "SyntheticMarketSimulator",
    "plot_market_world",
    "run_market_demo",
    "serve_market_view",
]
