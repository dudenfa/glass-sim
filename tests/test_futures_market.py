import math
from dataclasses import replace

import pytest

from marl_trading.futures import (
    Agent,
    AgentSpec,
    Cancel,
    ExternalAgent,
    FundingConfig,
    IndexConfig,
    LatencyConfig,
    LimitOrder,
    MarketConfig,
    MarketOrder,
    NewsConfig,
    PerpetualMarket,
    Side,
    TimeInForce,
    Transparency,
    TransparencyError,
    scenarios,
)


class Idle(Agent):
    def __init__(self):
        super().__init__(wake_interval=1e12)


def bare_market(accounts, funding=None, **overrides):
    config = MarketConfig(
        index=IndexConfig(enabled=False),
        news=NewsConfig(enabled=False),
        funding=funding or FundingConfig(enabled=False),
        latency=LatencyConfig(base=0.0, jitter=0.0),
        duration=3600.0,
        **overrides,
    )
    agents = []
    for agent_id, collateral in accounts.items():
        agent = Idle()
        agent.agent_id = agent_id
        agents.append((agent, collateral))
    return PerpetualMarket(config, agents=agents)


def last_rejection(market):
    events = [e for e in market.recorder.recent_orders if e.kind == "rejected"]
    return events[-1].reason if events else None


# ---------------------------------------------------------------- order entry


def test_trade_updates_both_accounts_fees_and_conservation():
    m = bare_market({"mm": 100_000.0, "t": 10_000.0})
    m.execute("mm", [LimitOrder(Side.SELL, 10, 100.5)])
    m.execute("t", [MarketOrder(Side.BUY, 4)])
    mm, t = m.accounts["mm"], m.accounts["t"]
    assert t.position == 4 and mm.position == -4
    assert t.entry_price == pytest.approx(100.5)
    assert t.fees_paid == pytest.approx(4 * 100.5 * m.contract.taker_fee)
    assert mm.fees_paid == pytest.approx(4 * 100.5 * m.contract.maker_fee)
    assert m.last_price == pytest.approx(100.5)
    assert mm.open_sell_qty == 6
    assert m.conservation_error() == pytest.approx(0.0, abs=1e-9)


def test_insufficient_margin_is_rejected_but_reducing_is_allowed():
    m = bare_market({"mm": 1_000_000.0, "t": 100.0})
    m.execute("mm", [LimitOrder(Side.SELL, 5_000, 100.1), LimitOrder(Side.BUY, 5_000, 99.9)])
    m.execute("t", [MarketOrder(Side.BUY, 1_000)])
    assert m.accounts["t"].position == 0
    assert last_rejection(m) == "insufficient_margin"
    m.execute("t", [MarketOrder(Side.BUY, 30)])  # 3,000 notional at 40x needs 75 of margin
    assert m.accounts["t"].position == 30
    m.execute("t", [MarketOrder(Side.SELL, 30, reduce_only=True)])
    assert m.accounts["t"].position == 0


def test_post_only_ioc_and_reduce_only_rules():
    m = bare_market({"mm": 100_000.0, "t": 10_000.0})
    m.execute("mm", [LimitOrder(Side.SELL, 5, 101.0)])
    m.execute("t", [LimitOrder(Side.BUY, 1, 101.0, tif=TimeInForce.POST_ONLY)])
    assert last_rejection(m) == "post_only_would_cross"
    m.execute("t", [LimitOrder(Side.BUY, 8, 101.0, tif=TimeInForce.IOC)])
    assert m.accounts["t"].position == 5
    assert len(m.book) == 0, "IOC remainder must not rest"
    m.execute("t", [MarketOrder(Side.BUY, 3, reduce_only=True)])
    assert last_rejection(m) == "reduce_only_no_position"


def test_market_order_without_liquidity_is_rejected():
    m = bare_market({"t": 10_000.0})
    m.execute("t", [MarketOrder(Side.BUY, 1)])
    assert last_rejection(m) == "no_liquidity"


def test_cancel_releases_reservation():
    m = bare_market({"mm": 100_000.0})
    m.execute("mm", [LimitOrder(Side.BUY, 10, 99.0)])
    order_id = m.book.orders_of("mm")[0].order_id
    assert m.accounts["mm"].open_buy_qty == 10
    m.execute("mm", [Cancel(order_id)])
    assert m.accounts["mm"].open_buy_qty == 0
    assert m.accounts["mm"].open_orders == 0


# ---------------------------------------------------------------- liquidation


def test_liquidation_closes_position_and_insurance_covers_bad_debt():
    m = bare_market({"mm": 1_000_000.0, "t": 1_000.0, "other": 100_000.0})
    m.execute("mm", [LimitOrder(Side.SELL, 350, 100.0)])
    m.execute("t", [MarketOrder(Side.BUY, 350)])
    liq_price = m.accounts["t"].liquidation_price(m.contract.mmr)
    assert 98.0 < liq_price < 99.0
    # The market moves down: the maker requotes lower and a trade prints there.
    m.execute("mm", [LimitOrder(Side.BUY, 1_000, 97.0), LimitOrder(Side.SELL, 10, 97.1)])
    fund_before = m.insurance_fund
    m.execute("other", [MarketOrder(Side.BUY, 1)])
    m.shock_index(math.log(97.0 / 100.0))
    t = m.accounts["t"]
    assert t.position == 0
    assert not t.active
    assert t.collateral == 0.0
    record = m.recorder.liquidations[-1]
    assert record.agent_id == "t" and record.quantity == 350 and record.bad_debt > 0
    assert m.insurance_fund == pytest.approx(fund_before - record.bad_debt)
    assert m.conservation_error() == pytest.approx(0.0, abs=1e-6)


def test_auto_deleveraging_when_book_cannot_absorb():
    m = bare_market({"t": 1_000.0, "short": 100_000.0})
    m.execute("short", [LimitOrder(Side.SELL, 350, 100.0)])
    m.execute("t", [MarketOrder(Side.BUY, 350)])
    assert m.accounts["short"].position == -350
    m.shock_index(math.log(97.0 / 100.0))  # no bids at all: the position must be ADL'd
    assert m.accounts["t"].position == 0
    assert m.accounts["short"].position == 0
    record = m.recorder.liquidations[-1]
    assert record.adl_quantity == 350
    assert any(trade.adl for trade in m.recorder.trades)
    assert m.conservation_error() == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------------- funding


def test_funding_longs_pay_shorts_when_perp_is_rich():
    funding = FundingConfig(interval=20.0, sample_interval=1.0, interest_rate=0.0, cap=0.01)
    m = bare_market({"mm": 1_000_000.0, "long": 50_000.0}, funding=funding)
    m.execute("mm", [LimitOrder(Side.SELL, 100, 101.0), LimitOrder(Side.BUY, 100, 100.8)])
    m.execute("long", [MarketOrder(Side.BUY, 50)])
    m.run_until(21.0)
    assert m.recorder.funding, "a funding settlement should have happened"
    record = m.recorder.funding[0]
    assert record.rate > 0
    assert m.accounts["long"].funding_paid > 0
    assert m.accounts["mm"].funding_paid == pytest.approx(-m.accounts["long"].funding_paid)
    assert m.conservation_error() == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------- reproducibility


def _fingerprint(result):
    return (len(result.trades), round(sum(t.price * t.quantity for t in result.trades), 6), tuple(round(a.equity, 6) for a in result.accounts))


def test_same_seed_same_result_and_different_seed_differs():
    a = PerpetualMarket(scenarios.small(seed=5, duration=600.0)).run()
    b = PerpetualMarket(scenarios.small(seed=5, duration=600.0)).run()
    c = PerpetualMarket(scenarios.small(seed=6, duration=600.0)).run()
    assert _fingerprint(a) == _fingerprint(b)
    assert _fingerprint(a) != _fingerprint(c)


def test_exogenous_path_is_shared_across_treatments():
    full = PerpetualMarket(scenarios.small(seed=3, duration=600.0, transparency=Transparency.FULL)).run()
    anon = PerpetualMarket(scenarios.small(seed=3, duration=600.0, transparency=Transparency.ANONYMOUS)).run()
    assert full.bars["index"] == anon.bars["index"]


def test_accounting_is_conserved_throughout_a_run():
    m = PerpetualMarket(scenarios.small(seed=2, duration=900.0))
    total = m.total_deposits + m.config.liquidation.insurance_fund
    while not m.finished:
        kind = m.step()
        if kind == "record":
            assert abs(m.conservation_error()) < 1e-6 * total
    assert abs(m.results().conservation_error) < 1e-6 * total


# --------------------------------------------------------------- transparency


def _transparency_market(level):
    m = bare_market({"a": 100_000.0, "b": 100_000.0}, transparency=level)
    m.execute("a", [LimitOrder(Side.SELL, 5, 100.0)])
    m.execute("b", [MarketOrder(Side.BUY, 2)])
    return m


def test_anonymous_market_hides_identities_and_positions():
    view = _transparency_market(Transparency.ANONYMOUS).view("b")
    assert view.recent_trades()[-1].buyer is None
    with pytest.raises(TransparencyError):
        view.book_orders(Side.SELL)
    with pytest.raises(TransparencyError):
        view.accounts()
    assert view.me.position == 2


def test_attributed_market_shows_owners_not_positions():
    view = _transparency_market(Transparency.ATTRIBUTED).view("b")
    trade = view.recent_trades()[-1]
    assert (trade.buyer, trade.seller) == ("b", "a")
    assert view.book_orders(Side.SELL)[0].owner == "a"
    with pytest.raises(TransparencyError):
        view.accounts()


def test_full_market_shows_every_position():
    view = _transparency_market(Transparency.FULL).view("b")
    accounts = {a.public_id: a for a in view.accounts()}
    assert accounts["a"].position == -2
    assert "b" not in accounts
    assert {a.public_id for a in view.accounts(include_self=True)} == {"a", "b"}


def test_pseudonymous_ids_hide_agent_names():
    m = PerpetualMarket(scenarios.small(seed=1, duration=60.0, pseudonymous_ids=True))
    public = {acc.public_id for acc in m.accounts.values()}
    assert all(p.startswith("0x") for p in public)
    assert len(public) == len(m.accounts)


# ------------------------------------------------------------------- external


def test_external_agent_decision_loop():
    base = scenarios.small(seed=4, duration=300.0)
    config = replace(base, agents=base.agents + (AgentSpec(ExternalAgent, count=1, collateral=50_000.0, params={"wake_interval": 10.0}, external=True),))
    m = PerpetualMarket(config)
    decisions = 0
    while True:
        agent_id = m.run_until_decision()
        if agent_id is None:
            break
        decisions += 1
        view = m.view(agent_id)
        side = Side.BUY if view.position <= 0 else Side.SELL
        m.act(agent_id, [MarketOrder(side, 1)])
    assert decisions > 10
    assert m.accounts["ext_01"].trade_count > 0
    assert m.finished


def test_external_agent_with_policy_runs_inline():
    base = scenarios.small(seed=4, duration=300.0)
    policy = lambda view: [MarketOrder(Side.BUY, 1)] if view.position < 5 else []
    spec = AgentSpec(ExternalAgent, count=1, collateral=50_000.0, params={"wake_interval": 10.0, "policy": policy})
    result = PerpetualMarket(replace(base, agents=base.agents + (spec,))).run()
    ext = next(a for a in result.accounts if a.agent_id == "ext_01")
    assert ext.position == 5


# ------------------------------------------------------------------ scenarios


@pytest.mark.parametrize("name", scenarios.names())
def test_every_scenario_runs_and_conserves(name):
    result = PerpetualMarket(scenarios.get(name, seed=1, duration=300.0)).run()
    summary = result.summary()
    assert summary["trades"] > 0
    assert abs(summary["conservation_error"]) < 1e-6 * sum(a.initial_collateral for a in result.accounts)
    assert result.end_time == pytest.approx(300.0)


@pytest.mark.parametrize(
    "level, active",
    [
        (Transparency.ANONYMOUS, set()),
        (Transparency.ATTRIBUTED, {"flow_follower"}),
        (Transparency.FULL, {"flow_follower", "copy_trader"}),
    ],
)
def test_transparency_aware_agents_only_act_when_they_can_see(level, active):
    result = PerpetualMarket(scenarios.default(seed=1, duration=1800.0, transparency=level)).run()
    traded = {a.agent_type for a in result.accounts if a.trade_count > 0}
    for agent_type in ("flow_follower", "copy_trader"):
        assert (agent_type in traded) == (agent_type in active), agent_type
