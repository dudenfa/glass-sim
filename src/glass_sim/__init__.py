"""A transparent perpetual-futures market simulator for AI trading agents.

Quick start::

    from glass_sim import PerpetualMarket, scenarios

    results = PerpetualMarket(scenarios.default(seed=1)).run()
    results.print_report()

Write your own agent by subclassing ``Agent`` and returning actions
(``LimitOrder``, ``MarketOrder``, ``Cancel``, ``CancelAll``) from ``on_wakeup``;
add it to a scenario with an ``AgentSpec``. See ``docs/user-guide.md``.
"""

from . import scenarios
from .account import AccountSnapshot, MarginAccount
from .agents import (
    Agent,
    BasisArbitrageur,
    CopyTrader,
    ExternalAgent,
    FlowFollower,
    InformedTrader,
    LiquidationHunter,
    MarketMaker,
    MomentumTrader,
    NoiseTrader,
    Whale,
)
from .config import (
    HOUR,
    AgentSpec,
    ContractSpec,
    FundingConfig,
    IndexConfig,
    LatencyConfig,
    LiquidationConfig,
    MarketConfig,
    NewsConfig,
    format_time,
    parse_duration,
)
from .market import PerpetualMarket
from .processes import NewsItem, PrivateSignal
from .replay import Replay, ReplayUnavailable
from .results import SavedRun, SimulationResult
from .types import (
    Action,
    Cancel,
    CancelAll,
    Fill,
    LimitOrder,
    MarketOrder,
    Side,
    TimeInForce,
    TradeRecord,
    Transparency,
)
from .views import MarketView, TransparencyError

__all__ = [
    "AccountSnapshot",
    "Action",
    "Agent",
    "AgentSpec",
    "BasisArbitrageur",
    "Cancel",
    "CancelAll",
    "ContractSpec",
    "CopyTrader",
    "ExternalAgent",
    "Fill",
    "FlowFollower",
    "FundingConfig",
    "HOUR",
    "IndexConfig",
    "InformedTrader",
    "LatencyConfig",
    "LimitOrder",
    "LiquidationConfig",
    "LiquidationHunter",
    "MarginAccount",
    "MarketConfig",
    "MarketMaker",
    "MarketOrder",
    "MarketView",
    "MomentumTrader",
    "NewsConfig",
    "NewsItem",
    "NoiseTrader",
    "PerpetualMarket",
    "PrivateSignal",
    "Replay",
    "ReplayUnavailable",
    "SavedRun",
    "Side",
    "SimulationResult",
    "TimeInForce",
    "TradeRecord",
    "Transparency",
    "TransparencyError",
    "Whale",
    "format_time",
    "parse_duration",
    "scenarios",
]
