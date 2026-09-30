"""Write your own trading agent and add it to the market.

    PYTHONPATH=src python examples/futures/02_custom_agent.py

An agent is a class with an ``on_wakeup(view, rng)`` method that returns a
list of actions. ``view`` is what the agent can see (depending on the market's
transparency level); ``rng`` is the agent's own random generator.
"""

from dataclasses import replace

from marl_trading.futures import (
    Agent,
    AgentSpec,
    LimitOrder,
    MarketOrder,
    PerpetualMarket,
    Side,
    TimeInForce,
    Transparency,
    parse_duration,
    scenarios,
)


class FadeTheCrowd(Agent):
    """Takes the opposite side of the aggregate retail position (needs FULL transparency).

    Under FULL transparency every account's position is public, so the agent
    can add up what retail traders hold and bet against it.
    """

    agent_type = "crowd_fader"
    default_prefix = "fader"

    def __init__(self, wake_interval: float = 20.0, leverage: float = 3.0) -> None:
        super().__init__(wake_interval=wake_interval)
        self.leverage = leverage

    def on_wakeup(self, view, rng):
        if not view.can_see(Transparency.FULL):
            return []
        retail_net = sum(a.position for a in view.accounts() if a.public_id.startswith("retail"))
        capacity = view.max_position(leverage=self.leverage)
        target = -capacity if retail_net > 500 else capacity if retail_net < -500 else 0
        delta = target - view.position
        if delta == 0:
            return []
        side = Side.BUY if delta > 0 else Side.SELL
        # Cross the spread with an immediate-or-cancel limit order, never more than 5 bps away.
        touch = view.best_ask if side is Side.BUY else view.best_bid
        if touch is None:
            return []
        limit = touch * (1.0005 if side is Side.BUY else 0.9995)
        return [LimitOrder(side, min(abs(delta), 200), view.round_price(limit), tif=TimeInForce.IOC)]


base = scenarios.default(seed=3, duration=parse_duration("2h"))
config = replace(base, agents=base.agents + (AgentSpec(FadeTheCrowd, count=2, collateral=100_000.0),))
result = PerpetualMarket(config).run()
result.print_report()
