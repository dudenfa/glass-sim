"""The agent interface.

To create an agent, subclass ``Agent`` and implement ``on_wakeup``:

.. code-block:: python

    from glass_sim import Agent, MarketOrder, Side

    class CoinFlipper(Agent):
        agent_type = "coin_flipper"

        def on_wakeup(self, view, rng):
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            return [MarketOrder(side, 1)]

The market calls ``on_wakeup`` at random times (a Poisson process with mean
``wake_interval`` seconds, override ``next_wakeup_delay`` for other clocks).
The agent inspects the ``MarketView`` and returns a list of actions. Actions
reach the exchange after a small latency, so the market may have moved.

Each agent has its own random generator ``rng``: use it (never the global
``random``/``numpy.random``) so that simulations are reproducible and adding
one agent never changes another agent's random choices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import numpy as np

from ..types import Action, Fill

if TYPE_CHECKING:  # pragma: no cover
    from ..processes import NewsItem, PrivateSignal
    from ..views import MarketView


class Agent:
    """Base class for every market participant."""

    agent_type: str = "agent"
    default_prefix: str = "agent"

    def __init__(self, wake_interval: float = 10.0) -> None:
        if wake_interval <= 0:
            raise ValueError("wake_interval must be positive")
        self.wake_interval = float(wake_interval)
        # Set by the market when the agent joins.
        self.agent_id: str = ""
        self.public_id: str = ""

    # --------------------------------------------------------------- clock

    def next_wakeup_delay(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.wake_interval))

    # ------------------------------------------------------------- hooks

    def on_start(self, view: "MarketView", rng: np.random.Generator) -> Sequence[Action]:
        """Called once at time 0, before any wakeup. Return initial actions."""
        return ()

    def on_wakeup(self, view: "MarketView", rng: np.random.Generator) -> Sequence[Action]:
        """Main decision point. Return the actions to send to the exchange."""
        return ()

    def on_fill(self, fill: Fill, view: "MarketView") -> None:
        """Called when one of the agent's orders trades. No actions allowed here."""

    def on_signal(self, signal: "PrivateSignal", view: "MarketView", rng: np.random.Generator) -> Sequence[Action]:
        """Called when a private signal arrives (only for subscribed agents)."""
        return ()

    def on_news(self, news: "NewsItem", view: "MarketView", rng: np.random.Generator) -> Sequence[Action]:
        """Called when public news is released."""
        return ()

    def on_liquidation(self, view: "MarketView") -> None:
        """Called after the agent's position was liquidated."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.agent_id or '?'})"


class ExternalAgent(Agent):
    """An agent whose decisions come from outside the simulator (e.g. an RL policy).

    If ``policy`` is given it is called as ``policy(view) -> actions`` at every
    wakeup, which is convenient for evaluating a trained model. Without a
    policy the market pauses at each wakeup: ``PerpetualMarket.run_until_decision``
    returns the agent id, and the caller answers with ``PerpetualMarket.act``.
    """

    agent_type = "external"
    default_prefix = "ext"

    def __init__(self, wake_interval: float = 5.0, policy: Optional[Callable[["MarketView"], Sequence[Action]]] = None) -> None:
        super().__init__(wake_interval=wake_interval)
        self.policy = policy

    def on_wakeup(self, view: "MarketView", rng: np.random.Generator) -> Sequence[Action]:
        if self.policy is None:
            raise RuntimeError("ExternalAgent without policy is driven by PerpetualMarket.act")
        return list(self.policy(view))


def actions_list(actions: Optional[Sequence[Action]]) -> List[Action]:
    if not actions:
        return []
    return list(actions)
