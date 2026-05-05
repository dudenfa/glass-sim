from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
    mask_invalid_action,
    observation_to_feature_dict,
)
from marl_trading.rl.live import RuntimePolicyDecision, decode_policy_action
from marl_trading.rl.scenario import prepare_frozen_agent_config, prepare_learning_agent_config


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
        available_cash=10_000.0,
        available_inventory=4.0,
        open_orders=2,
        active_agents=4,
        portfolio_active=True,
        agent_type="market_maker",
        public_note="step=12",
    )
    base.update(overrides)
    if "available_cash" not in overrides:
        base["available_cash"] = float(base["agent_cash"])
    if "available_inventory" not in overrides:
        base["available_inventory"] = float(base["agent_inventory"])
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
    assert features["has_best_bid"] == pytest.approx(1.0)
    assert features["has_best_ask"] == pytest.approx(1.0)
    assert features["fundamental_gap"] == pytest.approx(0.04)
    assert features["portfolio_active"] == pytest.approx(1.0)
    assert feature_vector(obs) == tuple(
        features[key]
        for key in (
            "best_bid",
            "best_ask",
            "has_best_bid",
            "has_best_ask",
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


def test_reward_defaults_new_coefficients_to_zero() -> None:
    reward = compute_reward(
        previous_equity=100.0,
        current_equity=104.5,
        current_inventory=3.0,
        previous_realized_pnl=1.25,
        current_realized_pnl=4.25,
        absolute_inventory_penalty_coefficient=0.5,
        inventory_risk_penalty_coefficient=0.0,
    )

    assert reward.realized_pnl_delta == pytest.approx(3.0)
    assert reward.equity_delta == pytest.approx(4.5)
    assert reward.signal_reward == pytest.approx(4.5)
    assert reward.inactivity_penalty == pytest.approx(0.0)
    assert reward.inventory_penalty == pytest.approx(1.5)
    assert reward.inventory_risk_penalty == pytest.approx(0.0)
    assert reward.total_reward == pytest.approx(3.0)


def test_reward_can_use_realized_pnl_delta_as_primary_signal() -> None:
    reward = compute_reward(
        previous_equity=100.0,
        current_equity=104.5,
        current_inventory=3.0,
        previous_realized_pnl=1.25,
        current_realized_pnl=4.25,
        realized_pnl_delta_coefficient=1.0,
        equity_delta_coefficient=0.0,
        absolute_inventory_penalty_coefficient=0.5,
        inventory_risk_penalty_coefficient=0.0,
    )

    assert reward.realized_pnl_delta == pytest.approx(3.0)
    assert reward.equity_delta == pytest.approx(4.5)
    assert reward.signal_reward == pytest.approx(3.0)
    assert reward.inactivity_penalty == pytest.approx(0.0)
    assert reward.inventory_penalty == pytest.approx(1.5)
    assert reward.inventory_risk_penalty == pytest.approx(0.0)
    assert reward.total_reward == pytest.approx(1.5)


def test_reward_can_include_optional_equity_shaping_and_inactivity_penalties() -> None:
    reward = compute_reward(
        previous_equity=100.0,
        current_equity=104.5,
        current_inventory=3.0,
        previous_realized_pnl=1.25,
        current_realized_pnl=4.25,
        realized_pnl_delta_coefficient=1.0,
        equity_delta_coefficient=0.4,
        inactivity_penalty_applied=True,
        inactivity_penalty_coefficient=0.3,
        absolute_inventory_penalty_coefficient=0.5,
        inventory_risk_penalty_coefficient=0.2,
    )

    assert reward.realized_pnl_delta == pytest.approx(3.0)
    assert reward.equity_delta == pytest.approx(4.5)
    assert reward.signal_reward == pytest.approx(4.8)
    assert reward.inactivity_penalty == pytest.approx(0.3)
    assert reward.inventory_penalty == pytest.approx(1.5)
    assert reward.inventory_risk_penalty == pytest.approx(1.8)
    assert reward.total_reward == pytest.approx(1.2)


def test_reward_zeroes_out_equity_when_only_realized_signal_is_enabled() -> None:
    reward = compute_reward(
        previous_equity=100.0,
        current_equity=104.5,
        current_inventory=3.0,
        previous_realized_pnl=1.25,
        current_realized_pnl=4.25,
        realized_pnl_delta_coefficient=1.0,
        equity_delta_coefficient=0.0,
    )

    assert reward.realized_pnl_delta == pytest.approx(3.0)
    assert reward.equity_delta == pytest.approx(4.5)
    assert reward.signal_reward == pytest.approx(3.0)
    assert reward.inactivity_penalty == pytest.approx(0.0)
    assert reward.inventory_penalty == pytest.approx(0.0)
    assert reward.inventory_risk_penalty == pytest.approx(0.0)
    assert reward.total_reward == pytest.approx(3.0)


def test_decode_policy_action_supports_multidiscrete_shape() -> None:
    action = decode_policy_action([3, 1, 2])
    assert action.action_type is RLActionType.LIMIT_BUY
    assert action.quantity == 2
    assert action.price_offset_ticks == 3


def test_mask_invalid_action_blocks_sell_without_inventory() -> None:
    effective, reason = mask_invalid_action(
        RLAction(RLActionType.LIMIT_SELL, quantity=2, price_offset_ticks=1),
        _observation(agent_inventory=0.0, open_orders=0),
    )

    assert effective.action_type is RLActionType.HOLD
    assert reason == "insufficient_inventory_for_sell"


def test_mask_invalid_action_blocks_sell_when_inventory_is_fully_reserved() -> None:
    effective, reason = mask_invalid_action(
        RLAction(RLActionType.LIMIT_SELL, quantity=1, price_offset_ticks=1),
        _observation(agent_inventory=1.0, available_inventory=0.0, open_orders=1),
    )

    assert effective.action_type is RLActionType.HOLD
    assert reason == "insufficient_inventory_for_sell"


def test_mask_invalid_action_blocks_buy_when_cash_is_fully_reserved() -> None:
    effective, reason = mask_invalid_action(
        RLAction(RLActionType.LIMIT_BUY, quantity=1, price_offset_ticks=1),
        _observation(agent_cash=100.0, available_cash=0.0, open_orders=1),
    )

    assert effective.action_type is RLActionType.HOLD
    assert reason == "insufficient_cash_for_buy"


def test_mask_invalid_action_blocks_market_buy_without_ask_liquidity() -> None:
    effective, reason = mask_invalid_action(
        RLAction(RLActionType.MARKET_BUY, quantity=1, price_offset_ticks=1),
        _observation(best_ask=None, spread=None),
    )

    assert effective.action_type is RLActionType.HOLD
    assert reason == "no_ask_liquidity_for_market_buy"


def test_mask_invalid_action_blocks_market_sell_without_bid_liquidity() -> None:
    effective, reason = mask_invalid_action(
        RLAction(RLActionType.MARKET_SELL, quantity=1, price_offset_ticks=1),
        _observation(best_bid=None, spread=None, agent_inventory=2.0),
    )

    assert effective.action_type is RLActionType.HOLD
    assert reason == "no_bid_liquidity_for_market_sell"


def test_mask_invalid_action_blocks_cancel_without_open_orders() -> None:
    effective, reason = mask_invalid_action(
        RLAction(RLActionType.CANCEL_OLDEST),
        _observation(agent_inventory=2.0, open_orders=0),
    )

    assert effective.action_type is RLActionType.HOLD
    assert reason == "no_open_orders_to_cancel"


def test_phase_a_action_mask_blocks_invalid_sell_actions_with_zero_inventory() -> None:
    config = prepare_frozen_agent_config(
        prepare_learning_agent_config(
            default_simulation_config(),
            learning_agent_id="rl_02",
            add_learning_agent=True,
            learning_agent_template_id="trend_01",
        ),
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )
    core_env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(
            learning_agent_id="trend_01",
            learning_agent_starting_inventory=0.0,
            phase_a_action_space=True,
            include_cancel_action=False,
            fixed_order_quantity=1,
            fixed_price_offset_ticks=1,
        ),
        horizon=24,
    )
    gym_env = GymSingleAgentMarketEnv(core_env)

    _obs, info = gym_env.reset(seed=7)

    assert info["action_types"] == ["hold", "market_buy", "market_sell", "limit_buy", "limit_sell"]
    assert info["phase_a_action_space"] is True
    assert info["action_mask"] == [True, True, False, True, False]


def test_phase_a_action_mask_blocks_market_buy_when_ask_side_is_empty() -> None:
    mask = GymSingleAgentMarketEnv(
        SingleAgentMarketEnv(
            config=replace(default_simulation_config()),
            env_config=SingleAgentEnvConfig(
                learning_agent_id="trend_01",
                learning_agent_starting_inventory=0.0,
                phase_a_action_space=True,
                include_cancel_action=False,
                fixed_order_quantity=1,
                fixed_price_offset_ticks=1,
            ),
            horizon=24,
        )
    ).action_masks()

    empty_ask_mask = GymSingleAgentMarketEnv(
        SingleAgentMarketEnv(
            config=replace(default_simulation_config()),
            env_config=SingleAgentEnvConfig(
                learning_agent_id="trend_01",
                learning_agent_starting_inventory=0.0,
                phase_a_action_space=True,
                include_cancel_action=False,
                fixed_order_quantity=1,
                fixed_price_offset_ticks=1,
            ),
            horizon=24,
        )
    )
    empty_ask_mask.core_env._current_observation = _observation(  # noqa: SLF001
        best_ask=None,
        spread=None,
        agent_inventory=0.0,
        open_orders=0,
    )

    assert mask.astype(bool).tolist()[1] is True
    assert empty_ask_mask.action_masks().astype(bool).tolist() == [True, False, False, True, False]


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


def test_runtime_policy_agent_masks_invalid_sell_and_reports_reason() -> None:
    obs = _observation(agent_inventory=0.0, open_orders=0)

    class _FakePredictor:
        def predict(self, observation, deterministic: bool = True):  # noqa: ARG002
            return [4, 0, 1], None

    agent = RuntimePolicyControlledAgent(
        "trend_01",
        policy=ModelPolicyAdapter(_FakePredictor(), deterministic=True),
        max_resting_orders=1,
    )
    intent = agent.decide(obs, rng=None)

    assert intent is None
    diagnostics = agent.diagnostics()
    assert diagnostics["action_counts"]["hold"] >= 1
    assert diagnostics["requested_action_counts"]["limit_sell"] >= 1
    assert diagnostics["invalid_action_count"] >= 1
    assert diagnostics["last_failure_reason"] == "insufficient_inventory_for_sell"
    assert diagnostics["last_action_type"] == "hold"
    assert diagnostics["last_requested_action_type"] == "limit_sell"


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
    config = prepare_frozen_agent_config(
        prepare_learning_agent_config(
            default_simulation_config(),
            learning_agent_id="rl_02",
            add_learning_agent=True,
            learning_agent_template_id="trend_01",
        ),
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )
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
    assert step_a[3]["realized_pnl_delta"] == pytest.approx(step_b[3]["realized_pnl_delta"])
    assert "reward_breakdown" in step_a[3]


def test_single_agent_env_supports_realized_pnl_reward_with_inactivity_penalty() -> None:
    config = prepare_frozen_agent_config(
        prepare_learning_agent_config(
            default_simulation_config(),
            learning_agent_id="rl_02",
            add_learning_agent=True,
            learning_agent_template_id="trend_01",
        ),
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )
    env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(
            learning_agent_id="trend_01",
            learning_agent_starting_inventory=0.0,
            reward_realized_pnl_delta_coefficient=1.0,
            reward_equity_delta_coefficient=0.0,
            reward_inactivity_penalty=0.25,
        ),
        horizon=24,
    )

    env.reset(seed=7, horizon=24)
    _, reward, done, info = env.step(RLAction(RLActionType.HOLD))

    assert done is False
    assert info["learning_agent_trade_count"] == 0
    assert info["learning_agent_had_trade"] is False
    assert info["inactivity_penalty_applied"] is True
    assert info["realized_pnl_delta"] == pytest.approx(0.0)
    assert info["reward_breakdown"].signal_reward == pytest.approx(0.0)
    assert info["reward_breakdown"].inactivity_penalty == pytest.approx(0.25)
    assert reward == pytest.approx(-0.25)


def test_single_agent_env_step_uses_realized_pnl_signal_over_equity_delta() -> None:
    env = SingleAgentMarketEnv.__new__(SingleAgentMarketEnv)
    env.env_config = SingleAgentEnvConfig(
        learning_agent_id="trend_01",
        reward_realized_pnl_delta_coefficient=1.0,
        reward_equity_delta_coefficient=0.0,
    )
    env.horizon = 10
    env.simulator = SimpleNamespace(
        current_step_index=1,
        open_orders={"trend_01": []},
    )
    env.simulator.step = lambda: setattr(env.simulator, "current_step_index", env.simulator.current_step_index + 1)
    env._proxy = SimpleNamespace(set_action=lambda action: None)
    env._done = False
    env._last_reward = 0.0
    env._last_info = {}
    env._last_seed = 7

    previous_observation = _observation(agent_inventory=2.0, agent_equity=100.0)
    current_observation = _observation(agent_inventory=2.0, agent_equity=108.0)
    env.get_observation = lambda: previous_observation
    env._learning_agent_inactive = lambda: False
    env._is_learning_turn = lambda: True
    env._build_observation = lambda: current_observation
    env._done_reason = lambda done, observation: None
    env._current_realized_pnl = lambda: 0.0
    env._ingest_new_trade_events = lambda: 0

    _, reward, done, info = env.step(RLAction(RLActionType.HOLD))

    assert done is False
    assert reward == pytest.approx(0.0)
    assert info["realized_pnl_delta"] == pytest.approx(0.0)
    assert info["reward_breakdown"].equity_delta == pytest.approx(8.0)
    assert info["reward_breakdown"].signal_reward == pytest.approx(0.0)
    assert info["learning_agent_trade_count"] == 0


def test_single_agent_env_masks_invalid_sell_before_execution() -> None:
    env = SingleAgentMarketEnv.__new__(SingleAgentMarketEnv)
    env.env_config = SingleAgentEnvConfig(learning_agent_id="trend_01")
    env.horizon = 10
    env.simulator = SimpleNamespace(
        current_step_index=1,
        open_orders={"trend_01": []},
    )
    env.simulator.step = lambda: setattr(env.simulator, "current_step_index", env.simulator.current_step_index + 1)
    captured_actions = []
    env._proxy = SimpleNamespace(set_action=lambda action: captured_actions.append(action))
    env._done = False
    env._last_reward = 0.0
    env._last_info = {}
    env._last_seed = 7

    previous_observation = _observation(agent_inventory=0.0, open_orders=0, agent_equity=100.0)
    current_observation = _observation(agent_inventory=0.0, open_orders=0, agent_equity=100.0)
    env.get_observation = lambda: previous_observation
    env._learning_agent_inactive = lambda: False
    env._is_learning_turn = lambda: True
    env._build_observation = lambda: current_observation
    env._done_reason = lambda done, observation: None
    env._current_realized_pnl = lambda: 0.0
    env._ingest_new_trade_events = lambda: 0

    _, reward, done, info = env.step(RLAction(RLActionType.LIMIT_SELL, quantity=2, price_offset_ticks=1))

    assert done is False
    assert reward == pytest.approx(0.0)
    assert captured_actions[-1].action_type is RLActionType.HOLD
    assert info["requested_action"] == "limit_sell"
    assert info["applied_action"] == "hold"
    assert info["action_masked"] is True
    assert info["invalid_action_reason"] == "insufficient_inventory_for_sell"


def test_single_agent_env_can_build_market_run_result_after_episode() -> None:
    config = prepare_frozen_agent_config(
        prepare_learning_agent_config(
            default_simulation_config(),
            learning_agent_id="rl_02",
            add_learning_agent=True,
            learning_agent_template_id="trend_01",
        ),
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )
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
    config = prepare_frozen_agent_config(
        prepare_learning_agent_config(
            default_simulation_config(),
            learning_agent_id="rl_02",
            add_learning_agent=True,
            learning_agent_template_id="trend_01",
        ),
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )
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


def test_frozen_runtime_agent_can_be_attached_with_inventory_override(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFrozenPolicy:
        def action_for(self, observation):  # noqa: ARG002
            raw_action = (0, 0, 0)
            return RuntimePolicyDecision(
                features=tuple(),
                raw_action=raw_action,
                rl_action=decode_policy_action(raw_action),
            )

    monkeypatch.setattr(
        "marl_trading.rl.env.PPOPolicyAdapter.try_load",
        lambda checkpoint_path, device="cpu", deterministic=True: (_FakeFrozenPolicy(), SimpleNamespace(available=True, checkpoint_path=checkpoint_path, reason=None)),
    )

    config = prepare_frozen_agent_config(
        prepare_learning_agent_config(
            default_simulation_config(),
            learning_agent_id="rl_02",
            add_learning_agent=True,
            learning_agent_template_id="trend_01",
        ),
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )
    env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(
            learning_agent_id="rl_02",
            learning_agent_starting_inventory=0.0,
            frozen_agent_id="rl_01",
            frozen_agent_checkpoint_path="/tmp/frozen_rl_01.zip",
            frozen_agent_starting_inventory=0.0,
        ),
        horizon=24,
    )

    env.reset(seed=7, horizon=24)
    frozen_portfolio = env.simulator.portfolios.get("rl_01")
    learning_portfolio = env.simulator.portfolios.get("rl_02")

    assert env.simulator.agents["rl_01"].agent_type == "rl_agent"
    assert frozen_portfolio.starting_inventory == pytest.approx(0.0)
    assert learning_portfolio.starting_inventory == pytest.approx(0.0)


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
    assert tuple(gym_env.observation_space.shape) == (18,)
    assert gym_env.action_space.n == 5

    obs_1, info_1 = gym_env.reset()
    obs_2, info_2 = gym_env.reset()

    assert tuple(obs_1.tolist()) != tuple(obs_2.tolist())
    assert info_1["learning_agent_id"] == "trend_01"
    assert info_1["action_types"] == ["hold", "market_buy", "market_sell", "limit_buy", "limit_sell"]
    assert info_1["phase_a_action_space"] is True
    assert info_2["seed"] == info_1["seed"] + 1


def test_gym_wrapper_cycles_explicit_train_seed_schedule() -> None:
    config = default_simulation_config()
    core_env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(
            learning_agent_id="trend_01",
            train_seeds=(3, 5, 8),
            auto_increment_seed_on_reset=False,
        ),
        horizon=24,
    )
    gym_env = GymSingleAgentMarketEnv(core_env, max_quantity=2, max_price_offset_ticks=4)

    _obs_1, info_1 = gym_env.reset()
    _obs_2, info_2 = gym_env.reset()
    _obs_3, info_3 = gym_env.reset()
    _obs_4, info_4 = gym_env.reset()

    assert info_1["train_seeds"] == [3, 5, 8]
    assert info_1["seed"] == 3
    assert info_2["seed"] == 5
    assert info_3["seed"] == 8
    assert info_4["seed"] == 3


def test_gym_wrapper_uses_fixed_phase_a_quantity_and_offset() -> None:
    config = default_simulation_config()
    core_env = SingleAgentMarketEnv(
        config=replace(config),
        env_config=SingleAgentEnvConfig(
            learning_agent_id="trend_01",
            fixed_order_quantity=2,
            fixed_price_offset_ticks=3,
        ),
        horizon=24,
    )
    gym_env = GymSingleAgentMarketEnv(core_env)

    action = gym_env._coerce_action(3)

    assert action.action_type is RLActionType.LIMIT_BUY
    assert action.quantity == 2
    assert action.price_offset_ticks == 3


def test_gym_wrapper_step_is_deterministic_for_fixed_seed_and_action() -> None:
    config = default_simulation_config()
    env_config = SingleAgentEnvConfig(learning_agent_id="maker_01")
    gym_a = GymSingleAgentMarketEnv(SingleAgentMarketEnv(config=replace(config), env_config=env_config, horizon=48))
    gym_b = GymSingleAgentMarketEnv(SingleAgentMarketEnv(config=replace(config), env_config=env_config, horizon=48))

    reset_a, info_a = gym_a.reset(seed=7)
    reset_b, info_b = gym_b.reset(seed=7)
    assert tuple(reset_a.tolist()) == tuple(reset_b.tolist())
    assert info_a["seed"] == info_b["seed"] == 7

    action = 3
    step_a = gym_a.step(action)
    step_b = gym_b.step(action)

    assert tuple(step_a[0].tolist()) == tuple(step_b[0].tolist())
    assert step_a[1] == pytest.approx(step_b[1])
    assert step_a[2] == step_b[2]
    assert step_a[3] == step_b[3]
    assert step_a[4]["rl_action"] == "limit_buy"
    assert step_a[4]["action_types"] == ["hold", "market_buy", "market_sell", "limit_buy", "limit_sell"]
    assert isinstance(step_a[4]["action_mask"], list)
