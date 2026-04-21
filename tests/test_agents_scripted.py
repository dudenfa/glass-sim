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
    intents = agent.decide(_observation(agent_type="noise_trader"), rng=DummyRng())

    assert len(intents) == 1
    intent = intents[0]
    assert intent.order_type is OrderType.MARKET


def test_trend_follower_responds_to_positive_returns() -> None:
    agent = TrendFollowerAgent("trend_01", threshold_bps=1.0, market_order_probability=0.0)
    intents = agent.decide(_observation(agent_type="trend_follower"), rng=__import__("numpy").random.default_rng(7))

    assert len(intents) == 1
    intent = intents[0]
    assert intent.side is Side.BUY
    assert intent.order_type is OrderType.LIMIT


def test_informed_trader_responds_to_fundamental_gap() -> None:
    agent = InformedTraderAgent("inf_01", signal_noise=0.0, threshold_bps=0.5)
    intents = agent.decide(_observation(agent_type="informed_trader", latent_fundamental=101.5), rng=__import__("numpy").random.default_rng(7))

    assert len(intents) == 1
    intent = intents[0]
    assert intent.side is Side.BUY


def test_market_maker_returns_two_sided_quotes_in_normal_mode() -> None:
    agent = MarketMakerAgent("maker_01", quote_size=3, max_resting_orders=3)

    intents = agent.decide(_observation(agent_type="market_maker", agent_inventory=40.0, open_orders=0), rng=None)

    assert [intent.side for intent in intents] == [Side.BUY, Side.SELL]
    assert all(intent.order_type is OrderType.LIMIT for intent in intents)


def test_market_maker_skews_quotes_without_dropping_a_side() -> None:
    agent = MarketMakerAgent(
        "maker_01",
        inventory_anchor=40.0,
        inventory_tolerance=4.0,
        quote_size=3,
        max_quote_size=5,
        bid_padding_ticks=2,
        ask_padding_ticks=2,
        inventory_skew_strength=1.0,
        inventory_size_decay=0.5,
    )
    intents = agent.decide(_observation(agent_type="market_maker", agent_inventory=52.0, open_orders=0), rng=None)

    assert [intent.side for intent in intents] == [Side.BUY, Side.SELL]
    bid_intent, ask_intent = intents
    assert ask_intent.quantity >= bid_intent.quantity
    assert ask_intent.limit_price <= bid_intent.limit_price + 0.06
    assert bid_intent.limit_price <= 99.98


def test_market_maker_restores_empty_ask_side_when_inventory_available() -> None:
    agent = MarketMakerAgent("maker_01", empty_side_padding_ticks=1)

    intents = agent.decide(
        _observation(
            agent_type="market_maker",
            best_ask=None,
            midpoint=None,
            spread=None,
            agent_inventory=40.0,
            open_orders=2,
        ),
        rng=None,
    )

    assert len(intents) == 1
    assert intents[0].side is Side.SELL
    assert intents[0].annotation == "restore_empty_side"


def test_market_maker_restores_empty_bid_side_when_cash_available() -> None:
    agent = MarketMakerAgent("maker_01", empty_side_padding_ticks=1)

    intents = agent.decide(
        _observation(
            agent_type="market_maker",
            best_bid=None,
            midpoint=None,
            spread=None,
            agent_cash=10_000.0,
            open_orders=2,
        ),
        rng=None,
    )

    assert len(intents) == 1
    assert intents[0].side is Side.BUY
    assert intents[0].annotation == "restore_empty_side"


def test_market_maker_degrades_to_one_sided_when_inventory_constrained() -> None:
    agent = MarketMakerAgent("maker_01", min_quote_size=1, max_quote_size=3)

    intents = agent.decide(
        _observation(agent_type="market_maker", agent_inventory=0.0, open_orders=0),
        rng=None,
    )

    assert len(intents) == 1
    assert intents[0].side is Side.BUY
