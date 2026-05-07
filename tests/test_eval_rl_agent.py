from __future__ import annotations

from pathlib import Path

import pytest

from marl_trading.configs import build_preset_config
from marl_trading.market.simulator import SyntheticMarketSimulator
from marl_trading.rl.live import RuntimePolicyDecision
from marl_trading.rl.boundary import RLAction, RLActionType
from marl_trading.rl import SingleAgentEnvConfig, SingleAgentMarketEnv
from scripts import eval_rl_agent


def test_parse_args_requires_checkpoint() -> None:
    args = eval_rl_agent.parse_args(["--checkpoint", "model.zip"])

    assert args.checkpoint == Path("model.zip")
    assert args.algorithm == "auto"
    assert args.learning_agent_id == "trend_01"
    assert args.add_learning_agent is False
    assert args.learning_agent_template_id is None
    assert args.learning_agent_starting_inventory == 0.0
    assert args.frozen_agent_checkpoint is None
    assert args.frozen_agent_id is None
    assert args.add_frozen_agent is False
    assert args.frozen_agent_template_id is None
    assert args.frozen_agent_starting_inventory is None
    assert args.phase_a_action_space is True
    assert args.include_cancel_action is False
    assert args.fixed_order_quantity == 1
    assert args.fixed_price_offset_ticks == 1
    assert args.reward_equity_delta_coefficient == 0.0
    assert args.reward_inactivity_penalty == 0.0
    assert args.reward_inventory_penalty == 0.0
    assert args.reward_inventory_risk_penalty == 0.0
    assert args.preset == "baseline"


def test_build_parser_help_describes_reward_shaping() -> None:
    help_text = eval_rl_agent.build_parser().format_help()

    assert "Reward shaping" in help_text
    assert "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta" in help_text
    assert "--reward-equity-delta-coefficient" in help_text
    assert "--reward-inactivity-penalty" in help_text
    assert "--inv-penalty" in help_text
    assert "--reward-inventory-risk-penalty" in help_text


def test_parse_args_accepts_short_reward_aliases() -> None:
    args = eval_rl_agent.parse_args(
        [
            "--checkpoint",
            "model.zip",
            "--reward-inactivity-penalty",
            "0.2",
            "--reward-equity-delta-coefficient",
            "0.15",
            "--inv-penalty",
            "0.1",
            "--inv-risk-penalty",
            "0.05",
        ]
    )

    assert args.reward_equity_delta_coefficient == 0.15
    assert args.reward_inactivity_penalty == 0.2
    assert args.reward_inventory_penalty == 0.1
    assert args.reward_inventory_risk_penalty == 0.05


def test_parse_args_accepts_maskable_phase_a_flags() -> None:
    args = eval_rl_agent.parse_args(
        [
            "--checkpoint",
            "model.zip",
            "--algorithm",
            "maskable_ppo",
            "--include-cancel-action",
            "--fixed-order-quantity",
            "2",
            "--fixed-price-offset-ticks",
            "3",
        ]
    )

    assert args.algorithm == "maskable_ppo"
    assert args.phase_a_action_space is True
    assert args.include_cancel_action is True
    assert args.fixed_order_quantity == 2
    assert args.fixed_price_offset_ticks == 3


def test_parse_args_accepts_add_learning_agent_mode() -> None:
    args = eval_rl_agent.parse_args(
        [
            "--checkpoint",
            "model.zip",
            "--learning-agent-id",
            "rl_01",
            "--add-learning-agent",
            "--learning-agent-template-id",
            "trend_01",
        ]
    )

    assert args.learning_agent_id == "rl_01"
    assert args.add_learning_agent is True
    assert args.learning_agent_template_id == "trend_01"


def test_parse_args_accepts_frozen_agent_mode() -> None:
    args = eval_rl_agent.parse_args(
        [
            "--checkpoint",
            "model.zip",
            "--frozen-agent-checkpoint",
            "checkpoints/ppo_baseline_rl_01_v1.zip",
            "--frozen-agent-id",
            "rl_01",
            "--add-frozen-agent",
            "--frozen-agent-template-id",
            "trend_01",
            "--frozen-agent-starting-inventory",
            "0",
        ]
    )

    assert args.frozen_agent_checkpoint == Path("checkpoints/ppo_baseline_rl_01_v1.zip")
    assert args.frozen_agent_id == "rl_01"
    assert args.add_frozen_agent is True
    assert args.frozen_agent_template_id == "trend_01"
    assert args.frozen_agent_starting_inventory == 0.0


def test_normalize_checkpoint_load_path_strips_zip_suffix(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ppo_test.zip"

    normalized = eval_rl_agent._normalize_checkpoint_load_path(checkpoint)

    assert normalized.endswith("ppo_test")
    assert not normalized.endswith(".zip")


def test_resolve_algorithm_uses_checkpoint_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ppo_test.zip"
    checkpoint.write_text("", encoding="utf-8")
    checkpoint.with_suffix(".json").write_text('{"algorithm":"maskable_ppo"}', encoding="utf-8")

    resolved = eval_rl_agent.resolve_algorithm(checkpoint, "auto")

    assert resolved == "maskable_ppo"


def test_build_rl_evaluation_payload_matches_market_health_shape(tmp_path: Path) -> None:
    config = build_preset_config("baseline")
    simulator = SyntheticMarketSimulator(config, horizon=24)
    result = simulator.run(horizon=24)

    payload = eval_rl_agent.build_rl_evaluation_payload(
        checkpoint_path=tmp_path / "ppo.zip",
        algorithm="maskable_ppo",
        preset_name="baseline",
        learning_agent_id="trend_01",
        add_learning_agent=False,
        learning_agent_template_id=None,
        learning_agent_starting_inventory=0.0,
        frozen_agent_checkpoint=None,
        frozen_agent_id=None,
        add_frozen_agent=False,
        frozen_agent_template_id=None,
        frozen_agent_starting_inventory=None,
        phase_a_action_space=True,
        include_cancel_action=False,
        fixed_order_quantity=1,
        fixed_price_offset_ticks=1,
        reward_equity_delta_coefficient=0.15,
        reward_inactivity_penalty=0.2,
        reward_inventory_penalty=0.1,
        reward_inventory_risk_penalty=0.05,
        result=result,
        config=config,
        horizon=24,
        deterministic=True,
        open_orders_by_agent={"trend_01": 1},
    )

    assert payload["preset"] == "baseline"
    assert payload["label"] == "baseline_rl"
    assert payload["metadata"]["algorithm"] == "maskable_ppo"
    assert payload["metadata"]["learning_agent_id"] == "trend_01"
    assert payload["metadata"]["add_learning_agent"] is False
    assert payload["metadata"]["learning_agent_template_id"] is None
    assert payload["metadata"]["learning_agent_starting_inventory"] == 0.0
    assert payload["metadata"]["frozen_agent_checkpoint"] is None
    assert payload["metadata"]["frozen_agent_id"] is None
    assert payload["metadata"]["add_frozen_agent"] is False
    assert payload["metadata"]["frozen_agent_template_id"] is None
    assert payload["metadata"]["frozen_agent_starting_inventory"] is None
    assert payload["metadata"]["phase_a_action_space"] is True
    assert payload["metadata"]["include_cancel_action"] is False
    assert payload["metadata"]["fixed_order_quantity"] == 1
    assert payload["metadata"]["fixed_price_offset_ticks"] == 1
    assert payload["metadata"]["reward_equity_delta_coefficient"] == 0.15
    assert payload["metadata"]["reward_inactivity_penalty"] == 0.2
    assert payload["metadata"]["reward_inventory_penalty"] == 0.1
    assert payload["metadata"]["reward_inventory_risk_penalty"] == 0.05
    assert payload["metadata"]["reward_signal"] == "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta"
    assert payload["metadata"]["reward_base_term"] == "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta"
    assert payload["metadata"]["reward_formula"] == (
        "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta - inactivity_penalty(if no trade) - "
        "abs(inventory) * reward_inventory_penalty - inventory^2 * reward_inventory_risk_penalty"
    )
    assert payload["metadata"]["reward_summary"] == (
        "realized_pnl_delta + 0.15 * equity_delta - 0.2 * inactivity(if no trade) - 0.1 * abs(inventory) - 0.05 * inventory^2"
    )
    assert payload["metadata"]["reward_shaping"]["equity_delta"]["coefficient"] == 0.15
    assert payload["metadata"]["reward_shaping"]["inactivity_penalty"]["coefficient"] == 0.2
    assert payload["metadata"]["reward_shaping"]["linear_inventory_penalty"]["coefficient"] == 0.1
    assert payload["metadata"]["reward_shaping"]["quadratic_inventory_risk_penalty"]["coefficient"] == 0.05
    assert payload["metadata"]["runtime_learning_agent_mode"] == "replace"
    assert payload["metadata"]["runtime_frozen_agent_mode"] is None
    assert payload["summary"]["trade_count"] >= 0
    assert isinstance(payload["portfolio_breakdown"], list)
    assert payload["agents"] == payload["portfolio_breakdown"]
    first_agent = payload["portfolio_breakdown"][0]
    assert "peak_equity" in first_agent
    assert "max_equity_drawdown" in first_agent
    assert "max_equity_drawdown_from_start_replay" in first_agent
    assert "min_equity_delta" in first_agent
    assert "max_pnl_drawdown_from_start" in first_agent
    assert "max_abs_inventory" in first_agent


def test_build_rl_evaluation_payload_adjusts_learning_slot_starting_inventory(tmp_path: Path) -> None:
    config = build_preset_config("baseline")
    simulator = SyntheticMarketSimulator(config, horizon=24)
    result = simulator.run(horizon=24)

    payload = eval_rl_agent.build_rl_evaluation_payload(
        checkpoint_path=tmp_path / "ppo.zip",
        algorithm="ppo",
        preset_name="baseline",
        learning_agent_id="trend_01",
        add_learning_agent=False,
        learning_agent_template_id=None,
        learning_agent_starting_inventory=0.0,
        frozen_agent_checkpoint=None,
        frozen_agent_id=None,
        add_frozen_agent=False,
        frozen_agent_template_id=None,
        frozen_agent_starting_inventory=None,
        phase_a_action_space=True,
        include_cancel_action=False,
        fixed_order_quantity=1,
        fixed_price_offset_ticks=1,
        reward_equity_delta_coefficient=0.0,
        reward_inactivity_penalty=0.0,
        reward_inventory_penalty=0.0,
        reward_inventory_risk_penalty=0.0,
        result=result,
        config=config,
        horizon=24,
        deterministic=True,
        open_orders_by_agent={"trend_01": 0},
    )

    trend = next(agent for agent in payload["portfolio_breakdown"] if agent["agent_id"] == "trend_01")
    assert trend["starting_inventory"] == 0.0
    assert trend["starting_equity"] == 10000.0


def test_build_rl_evaluation_payload_adjusts_frozen_and_learning_runtime_starting_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubFrozenPolicy:
        def action_for(self, observation) -> RuntimePolicyDecision:  # noqa: ARG002
            return RuntimePolicyDecision(
                features=(),
                raw_action=(0,),
                rl_action=RLAction(RLActionType.HOLD),
            )

    config = eval_rl_agent.build_eval_config(
        "baseline",
        learning_agent_id="rl_02",
        add_learning_agent=True,
        learning_agent_template_id="trend_01",
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )[0]
    monkeypatch.setattr(SingleAgentMarketEnv, "_load_frozen_policy", lambda self: _StubFrozenPolicy())
    env = SingleAgentMarketEnv(
        config=config,
        env_config=SingleAgentEnvConfig(
            learning_agent_id="rl_02",
            learning_agent_starting_inventory=0.0,
            frozen_agent_id="rl_01",
            frozen_agent_checkpoint_path=str(tmp_path / "frozen.zip"),
            frozen_agent_starting_inventory=0.0,
        ),
        horizon=24,
    )
    env.reset(seed=7, horizon=24)
    result = env.build_run_result()

    payload = eval_rl_agent.build_rl_evaluation_payload(
        checkpoint_path=tmp_path / "ppo.zip",
        algorithm="ppo",
        preset_name="baseline",
        learning_agent_id="rl_02",
        add_learning_agent=True,
        learning_agent_template_id="trend_01",
        learning_agent_starting_inventory=0.0,
        frozen_agent_checkpoint=tmp_path / "frozen.zip",
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
        frozen_agent_starting_inventory=0.0,
        phase_a_action_space=True,
        include_cancel_action=False,
        fixed_order_quantity=1,
        fixed_price_offset_ticks=1,
        reward_equity_delta_coefficient=0.0,
        reward_inactivity_penalty=0.0,
        reward_inventory_penalty=0.0,
        reward_inventory_risk_penalty=0.0,
        result=result,
        config=config,
        horizon=24,
        deterministic=True,
        open_orders_by_agent={
            agent_id: len(queue)
            for agent_id, queue in env.simulator.open_orders.items()
        },
    )

    rl_01 = next(agent for agent in payload["portfolio_breakdown"] if agent["agent_id"] == "rl_01")
    rl_02 = next(agent for agent in payload["portfolio_breakdown"] if agent["agent_id"] == "rl_02")
    assert rl_01["starting_inventory"] == 0.0
    assert rl_01["starting_equity"] == 10000.0
    assert rl_01["total_pnl"] == pytest.approx(rl_01["realized_pnl"] + rl_01["unrealized_pnl"])
    assert "peak_equity" in rl_01
    assert "max_equity_drawdown" in rl_01
    assert "max_equity_drawdown_from_start_replay" in rl_01
    assert "min_equity_delta" in rl_01
    assert "max_pnl_drawdown_from_start" in rl_01
    assert "max_abs_inventory" in rl_01
    assert rl_02["starting_inventory"] == 0.0
    assert rl_02["starting_equity"] == 10000.0
    assert rl_02["total_pnl"] == pytest.approx(rl_02["realized_pnl"] + rl_02["unrealized_pnl"])
    assert "peak_equity" in rl_02
    assert "max_equity_drawdown" in rl_02
    assert "max_equity_drawdown_from_start_replay" in rl_02
    assert "min_equity_delta" in rl_02
    assert "max_pnl_drawdown_from_start" in rl_02
    assert "max_abs_inventory" in rl_02


def test_build_eval_config_can_add_learning_agent() -> None:
    config, horizon = eval_rl_agent.build_eval_config(
        "baseline",
        learning_agent_id="rl_01",
        add_learning_agent=True,
        learning_agent_template_id="trend_01",
    )

    assert horizon == config.market.event_horizon
    agent_ids = [agent.agent_id.value for agent in config.agents]
    assert agent_ids == ["maker_01", "retail_01", "informed_01", "trend_01", "rl_01"]


def test_build_eval_config_can_add_learning_and_frozen_agents() -> None:
    config, horizon = eval_rl_agent.build_eval_config(
        "baseline",
        learning_agent_id="rl_02",
        add_learning_agent=True,
        learning_agent_template_id="trend_01",
        frozen_agent_id="rl_01",
        add_frozen_agent=True,
        frozen_agent_template_id="trend_01",
    )

    assert horizon == config.market.event_horizon
    agent_ids = [agent.agent_id.value for agent in config.agents]
    assert agent_ids == ["maker_01", "retail_01", "informed_01", "trend_01", "rl_01", "rl_02"]
