"""Drive an agent from outside the simulator: the hook for reinforcement learning.

    PYTHONPATH=src python examples/04_external_agent_loop.py

An ``ExternalAgent`` without a policy pauses the market every time it wakes
up. ``run_until_decision()`` returns its id; you inspect ``market.view(id)``
(an observation), send actions with ``act()`` (an action), and read the
account's equity change (a reward). A Gymnasium / PettingZoo wrapper is a thin
layer on top of this loop.
"""

from dataclasses import replace

from glass_sim import AgentSpec, ExternalAgent, MarketOrder, PerpetualMarket, Side, parse_duration, scenarios

base = scenarios.default(seed=7, duration=parse_duration("30m"))
learner = AgentSpec(ExternalAgent, count=1, collateral=50_000.0, params={"wake_interval": 5.0}, external=True, prefix="rl")
market = PerpetualMarket(replace(base, agents=base.agents + (learner,)))

steps = 0
last_equity = None
while (agent_id := market.run_until_decision()) is not None:
    view = market.view(agent_id)
    equity = view.me.equity
    reward = 0.0 if last_equity is None else equity - last_equity
    last_equity = equity

    # A trivial "policy": mean-revert toward the index.
    gap = (view.mid or view.last_price) - view.index_price
    if gap > 0.03 and view.position > -50:
        actions = [MarketOrder(Side.SELL, 10)]
    elif gap < -0.03 and view.position < 50:
        actions = [MarketOrder(Side.BUY, 10)]
    else:
        actions = []
    market.act(agent_id, actions)
    steps += 1

result = market.results()
me = next(a for a in result.accounts if a.agent_id == "rl_01")
print(f"{steps} decisions, final position {me.position}, total PnL {me.total_pnl:+.2f}")
