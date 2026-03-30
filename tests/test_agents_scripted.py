from __future__ import annotations

from marl_trading.agents import InformedTraderAgent, MarketMakerAgent, MarketObservation, NoiseTraderAgent, TrendFollowerAgent
from marl_trading.exchange.models import OrderType, Side


def _observation(**overrides):
    base = dict(
        timestamp_ns=1,
        symbol="SYNTH",
        tick_size=0.01,
        best_bid=99.99,
        best_ask=100.01,
        midpoint=100.0,
        spread=0.02,
        latent_fundamental=101.0,
        recent_midpoints=(99.8, 100.0),
        recent_returns_bps=(8.0, 12.0),
        news_headline=None,
        news_severity=None,
        agent_cash=10_000.0,
        agent_inventory=20.0,
        agent_equity=12_000.0,
        open_orders=0,
        active_agents=4,
        portfolio_active=True,
        agent_type="test",
        public_note="demo",
    )
    base.update(overrides)
    return MarketObservation(**base)


def test_market_maker_bootstrap_returns_bid_and_ask() -> None:
    agent = MarketMakerAgent("maker_01")
    intents = agent.bootstrap(_observation(agent_type="market_maker"), rng=None)

    assert len(intents) == 2
    assert intents[0].side is Side.BUY
    assert intents[1].side is Side.SELL
    assert intents[0].order_type is OrderType.LIMIT
    assert intents[1].order_type is OrderType.LIMIT


def test_noise_trader_can_emit_market_order() -> None:
    class DummyRng:
        def __init__(self) -> None:
            self.values = iter([0.1, 0.2, 0.1, 0.0])

        def random(self) -> float:
            return next(self.values)

    agent = NoiseTraderAgent("noise_01", aggressiveness=1.0, market_order_probability=1.0)
    intent = agent.decide(_observation(agent_type="noise_trader"), rng=DummyRng())

    assert intent is not None
    assert intent.order_type is OrderType.MARKET


def test_trend_follower_responds_to_positive_returns() -> None:
    agent = TrendFollowerAgent("trend_01", threshold_bps=1.0, market_order_probability=0.0)
    intent = agent.decide(_observation(agent_type="trend_follower"), rng=__import__("numpy").random.default_rng(7))

    assert intent is not None
    assert intent.side is Side.BUY
    assert intent.order_type is OrderType.LIMIT


def test_informed_trader_responds_to_fundamental_gap() -> None:
    agent = InformedTraderAgent("inf_01", signal_noise=0.0, threshold_bps=0.5)
    intent = agent.decide(_observation(agent_type="informed_trader", latent_fundamental=101.5), rng=__import__("numpy").random.default_rng(7))

    assert intent is not None
    assert intent.side is Side.BUY
