import numpy as np
import pytest

pytest.importorskip("gymnasium")

from glass_sim.rl import (
    ActionConfig,
    ActionSpace,
    EnvConfig,
    LadderActionSpace,
    ObservationEncoder,
    PerpTradingEnv,
    RewardTracker,
    make_action_space,
)
from glass_sim.types import Cancel, LimitOrder, MarketOrder, Side, TimeInForce

LADDER = ActionConfig(kind="ladder")


def small_env(action_config=ActionConfig(), **components):
    return PerpTradingEnv(EnvConfig(scenario="small", episode_duration=600.0, actions=action_config), **components)


def test_factory_picks_the_menu():
    assert type(make_action_space(ActionConfig())) is ActionSpace
    ladder = make_action_space(LADDER)
    assert isinstance(ladder, LadderActionSpace)
    assert ladder.n == len(ladder.names) == 3 + 2 * 2 + 2 * 4 * 2
    with pytest.raises(ValueError):
        make_action_space(ActionConfig(kind="nope"))


def test_ladder_env_passes_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = small_env(LADDER)
    assert env.action_space.n == 23
    check_env(env, skip_render_check=True)


def test_ladder_quotes_at_the_requested_offset():
    env = small_env(LADDER)
    env.reset()
    view = env.view
    space = env.actions
    index = space.names.index("bid_10bps_0.5x")
    orders = space.decode(index, view)
    assert len(orders) == 1 and isinstance(orders[0], LimitOrder)
    order = orders[0]
    assert order.side is Side.BUY and order.tif is TimeInForce.POST_ONLY
    assert order.price == pytest.approx(view.round_price(view.best_bid * (1 - 10 / 1e4)))
    assert order.price < view.best_bid
    ask = space.decode(space.names.index("ask_5bps_2x"), view)[0]
    assert ask.side is Side.SELL and ask.price > view.best_ask
    assert ask.quantity > order.quantity
    market = space.decode(space.names.index("market_sell_2x"), view)[0]
    assert isinstance(market, MarketOrder) and market.side is Side.SELL


def test_ladder_replaces_or_stacks_quotes():
    env = small_env(LADDER)
    env.reset()
    bid = env.actions.names.index("bid_5bps_0.5x")
    env.step(bid)
    assert env.view.my_orders(), "the quote rests"
    orders = env.actions.decode(bid, env.view)
    assert isinstance(orders[0], Cancel), "one order per side: the old bid is canceled"

    env = small_env(ActionConfig(kind="ladder", max_orders_per_side=3))
    env.reset()
    bid = env.actions.names.index("bid_10bps_0.5x")
    env.step(bid)
    assert env.view.my_orders(), "the quote rests"
    orders = env.actions.decode(bid, env.view)
    assert len(orders) == 1, "room left on the side: quotes stack"


def test_ladder_episode_keeps_accounting_consistent():
    env = small_env(LADDER)
    env.reset(options={"market_seed": 3})
    rng = np.random.default_rng(0)
    done = False
    while not done:
        action = rng.choice(np.flatnonzero(env.action_masks()))
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    assert info["action"] in env.actions.names
    assert abs(env.market.conservation_error()) < 1e-6


class CountingReward:
    """Custom reward: +1 per decision with a nonzero position."""

    def __init__(self, market, agent_id):
        self.market, self.agent_id, self.resets = market, agent_id, 0

    def reset(self):
        self.resets += 1

    def step(self):
        held = self.market.accounts[self.agent_id].position != 0
        return float(held), {"held": held}


class TinyEncoder:
    """Custom observation: spread and own position only."""

    feature_names = ["spread_bps", "position"]
    bounds = (-1e6, 1e6)
    size = 2

    def encode(self, view):
        spread = 0.0 if view.spread is None else view.spread / view.mark_price * 1e4
        return np.array([spread, view.position], dtype=np.float32)


class BuyOnly(ActionSpace):
    """Custom menu: hold or buy one contract."""

    names = ["hold", "buy_one"]

    @property
    def n(self):
        return 2

    def mask(self, view):
        return np.array([True, view.best_ask is not None])

    def decode(self, index, view):
        return [MarketOrder(Side.BUY, 1, tag="rl")] if index == 1 else []


def test_custom_components_are_used():
    from gymnasium.utils.env_checker import check_env

    env = small_env(encoder=TinyEncoder(), actions=BuyOnly(), reward=CountingReward)
    assert env.observation_space.shape == (2,) and env.action_space.n == 2
    obs, _ = env.reset()
    assert obs.shape == (2,) and env.tracker.resets == 1
    _, reward, _, _, info = env.step(1)
    assert info["action"] == "buy_one" and "held" in info
    assert env.market.accounts["rl_01"].position > 0
    assert reward == 1.0
    check_env(small_env(encoder=TinyEncoder(), actions=BuyOnly(), reward=CountingReward), skip_render_check=True)


def test_default_components_match_the_config():
    env = small_env()
    assert isinstance(env.encoder, ObservationEncoder)
    assert type(env.actions) is ActionSpace
    env.reset()
    assert isinstance(env.tracker, RewardTracker)
