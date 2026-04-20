from __future__ import annotations

from dataclasses import replace

import pytest

from marl_trading.agents.base import MarketObservation
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.exchange.models import OrderType, Side
from marl_trading.rl import (
    GymSingleAgentMarketEnv,
    ModelPolicyAdapter,
    PPOPolicyAdapter,
    RLAction,
    RLActionType,
    RuntimePolicyControlledAgent,
    SingleAgentEnvConfig,
    SingleAgentMarketEnv,
    action_to_order_intent,
    compute_reward,
    decode_policy_action,
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


def test_decode_policy_action_supports_multidiscrete_shape() -> None:
    action = decode_policy_action([3, 1, 2])
    assert action.action_type is RLActionType.LIMIT_BUY
    assert action.quantity == 2
    assert action.price_offset_ticks == 3


def test_runtime_policy_agent_maps_policy_prediction_to_order_intent() -> None:
    obs = _observation()

    class _FakePredictor:
        def predict(self, observation, deterministic: bool = True):  # noqa: ARG002
            return [3, 0, 1], None

    agent = RuntimePolicyControlledAgent(
        "trend_01",
        policy=ModelPolicyAdapter(_FakePredictor(), deterministic=True),
        max_resting_orders=1,
    )
    intent = agent.decide(obs, rng=None)

    assert intent is not None
    assert intent.side is Side.BUY
    assert intent.order_type is OrderType.LIMIT
    assert intent.limit_price == pytest.approx(99.98)
    assert agent.last_decision is not None
    assert agent.last_decision.rl_action.action_type is RLActionType.LIMIT_BUY


def test_try_load_ppo_policy_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    checkpoint = tmp_path / "ppo_checkpoint.zip"
    checkpoint.write_text("placeholder", encoding="utf-8")
    import importlib

    real_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "stable_baselines3":
            raise ImportError("missing stable-baselines3")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    adapter, status = PPOPolicyAdapter.try_load(checkpoint)

    assert adapter is None
    assert status.available is False
    assert status.reason == "stable-baselines3 is not installed."
    assert status.checkpoint_path == checkpoint.resolve()


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


def test_single_agent_env_can_build_market_run_result_after_episode() -> None:
    config = default_simulation_config()
    env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(learning_agent_id="trend_01"),
        horizon=18,
    )
    env.reset(seed=11, horizon=18)

    done = False
    while not done:
        _, _, done, _ = env.step(RLAction(RLActionType.HOLD))

    result = env.build_run_result()
    assert result.summary["horizon"] == 18
    assert "trend_01" in result.final_portfolios
    assert result.final_fundamental > 0


def test_learning_slot_can_override_starting_inventory() -> None:
    config = default_simulation_config()
    env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(
            learning_agent_id="trend_01",
            learning_agent_starting_inventory=0.0,
        ),
        horizon=24,
    )

    env.reset(seed=7, horizon=24)
    portfolio = env.simulator.portfolios.get("trend_01")
    observation = env.get_observation()

    assert portfolio.starting_inventory == pytest.approx(0.0)
    assert portfolio.inventory == pytest.approx(0.0)
    assert observation.agent_inventory == pytest.approx(0.0)


def test_gym_wrapper_exposes_spaces_and_reset_semantics() -> None:
    config = default_simulation_config()
    core_env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(
            learning_agent_id="trend_01",
            auto_increment_seed_on_reset=True,
        ),
        horizon=24,
    )
    gym_env = GymSingleAgentMarketEnv(core_env, max_quantity=2, max_price_offset_ticks=4)

    assert gym_env.learning_agent_id == "trend_01"
    assert tuple(gym_env.observation_space.shape) == (16,)
    assert tuple(int(value) for value in gym_env.action_space.nvec.tolist()) == (6, 2, 4)

    obs_1, info_1 = gym_env.reset()
    obs_2, info_2 = gym_env.reset()

    assert tuple(obs_1.tolist()) != tuple(obs_2.tolist())
    assert info_1["learning_agent_id"] == "trend_01"
    assert info_2["seed"] == info_1["seed"] + 1


def test_gym_wrapper_step_is_deterministic_for_fixed_seed_and_action() -> None:
    config = default_simulation_config()
    env_config = SingleAgentEnvConfig(learning_agent_id="maker_01")
    gym_a = GymSingleAgentMarketEnv(SingleAgentMarketEnv(config=replace(config), env_config=env_config, horizon=48))
    gym_b = GymSingleAgentMarketEnv(SingleAgentMarketEnv(config=replace(config), env_config=env_config, horizon=48))

    reset_a, info_a = gym_a.reset(seed=7)
    reset_b, info_b = gym_b.reset(seed=7)
    assert tuple(reset_a.tolist()) == tuple(reset_b.tolist())
    assert info_a["seed"] == info_b["seed"] == 7

    action = [3, 0, 0]
    step_a = gym_a.step(action)
    step_b = gym_b.step(action)

    assert tuple(step_a[0].tolist()) == tuple(step_b[0].tolist())
    assert step_a[1] == pytest.approx(step_b[1])
    assert step_a[2] == step_b[2]
    assert step_a[3] == step_b[3]
    assert step_a[4]["rl_action"] == "limit_buy"
