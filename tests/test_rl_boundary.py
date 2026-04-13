from __future__ import annotations

from dataclasses import replace

import pytest

from marl_trading.agents.base import MarketObservation
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.exchange.models import OrderType, Side
from marl_trading.rl import (
    RLAction,
    RLActionType,
    SingleAgentEnvConfig,
    SingleAgentMarketEnv,
    action_to_order_intent,
    compute_reward,
    feature_vector,
    observation_to_feature_dict,
)


def _observation(**overrides):
    base = dict(
        timestamp_ns=12,
        symbol="SYNTH",
        tick_size=0.01,
        best_bid=100.0,
        best_ask=100.02,
        midpoint=100.01,
        spread=0.02,
        latent_fundamental=100.05,
        recent_midpoints=(99.95, 100.0, 100.01),
        recent_returns_bps=(2.0, -1.0, 0.5),
        news_headline="Macro liquidity surprise",
        news_severity=0.75,
        agent_cash=10_000.0,
        agent_inventory=4.0,
        agent_equity=10_400.0,
        open_orders=2,
        active_agents=4,
        portfolio_active=True,
        agent_type="market_maker",
        public_note="step=12",
    )
    base.update(overrides)
    return MarketObservation(**base)


def test_action_mapping_covers_small_rl_action_set() -> None:
    obs = _observation()

    assert action_to_order_intent(RLAction(RLActionType.HOLD), obs) is None
    assert action_to_order_intent(RLAction(RLActionType.CANCEL_OLDEST), obs) is None

    market_buy = action_to_order_intent(RLAction(RLActionType.MARKET_BUY, quantity=2), obs)
    assert market_buy is not None
    assert market_buy.side is Side.BUY
    assert market_buy.order_type is OrderType.MARKET
    assert market_buy.quantity == 2

    market_sell = action_to_order_intent(RLAction(RLActionType.MARKET_SELL, quantity=3), obs)
    assert market_sell is not None
    assert market_sell.side is Side.SELL
    assert market_sell.order_type is OrderType.MARKET
    assert market_sell.quantity == 3

    limit_buy = action_to_order_intent(RLAction(RLActionType.LIMIT_BUY, quantity=1, price_offset_ticks=2), obs)
    assert limit_buy is not None
    assert limit_buy.side is Side.BUY
    assert limit_buy.order_type is OrderType.LIMIT
    assert limit_buy.limit_price == pytest.approx(99.98)

    limit_sell = action_to_order_intent(RLAction(RLActionType.LIMIT_SELL, quantity=1, price_offset_ticks=2), obs)
    assert limit_sell is not None
    assert limit_sell.side is Side.SELL
    assert limit_sell.order_type is OrderType.LIMIT
    assert limit_sell.limit_price == pytest.approx(100.04)


def test_observation_features_are_compact_and_ordered() -> None:
    obs = _observation()
    features = observation_to_feature_dict(obs)

    assert features["best_bid"] == pytest.approx(100.0)
    assert features["best_ask"] == pytest.approx(100.02)
    assert features["fundamental_gap"] == pytest.approx(0.04)
    assert features["portfolio_active"] == pytest.approx(1.0)
    assert feature_vector(obs) == tuple(
        features[key]
        for key in (
            "best_bid",
            "best_ask",
            "midpoint",
            "spread",
            "fundamental",
            "fundamental_gap",
            "return_bps_1",
            "return_bps_2",
            "return_bps_3",
            "news_severity",
            "agent_cash",
            "agent_inventory",
            "agent_equity",
            "open_orders",
            "active_agents",
            "portfolio_active",
        )
    )


def test_reward_is_equity_delta_minus_inventory_penalty() -> None:
    reward = compute_reward(
        previous_equity=100.0,
        current_equity=104.5,
        current_inventory=3.0,
        inventory_penalty_coefficient=0.5,
    )

    assert reward.equity_delta == pytest.approx(4.5)
    assert reward.inventory_penalty == pytest.approx(1.5)
    assert reward.total_reward == pytest.approx(3.0)


def test_single_agent_env_is_deterministic_for_fixed_seed_and_actions() -> None:
    config = default_simulation_config()
    env_config = SingleAgentEnvConfig(learning_agent_id="maker_01", reward_inventory_penalty=0.1)
    env_a = SingleAgentMarketEnv(config=replace(config), env_config=env_config, horizon=48)
    env_b = SingleAgentMarketEnv(config=replace(config), env_config=env_config, horizon=48)

    reset_a = env_a.reset(seed=7, horizon=48)
    reset_b = env_b.reset(seed=7, horizon=48)
    assert reset_a == reset_b

    action = RLAction(RLActionType.LIMIT_BUY, quantity=1, price_offset_ticks=1)
    step_a = env_a.step(action)
    step_b = env_b.step(action)

    assert step_a[0] == step_b[0]
    assert step_a[1] == pytest.approx(step_b[1])
    assert step_a[2] == step_b[2]
    assert step_a[3]["step_index"] == step_b[3]["step_index"]
    assert step_a[3]["applied_action"] == "limit_buy"
    assert "reward_breakdown" in step_a[3]
