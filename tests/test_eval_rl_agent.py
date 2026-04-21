from __future__ import annotations

from pathlib import Path

from marl_trading.configs import build_preset_config
from marl_trading.market.simulator import SyntheticMarketSimulator
from scripts import eval_rl_agent


def test_parse_args_requires_checkpoint() -> None:
    args = eval_rl_agent.parse_args(["--checkpoint", "model.zip"])

    assert args.checkpoint == Path("model.zip")
    assert args.algorithm == "auto"
    assert args.learning_agent_id == "trend_01"
    assert args.learning_agent_starting_inventory == 0.0
    assert args.phase_a_action_space is True
    assert args.include_cancel_action is False
    assert args.fixed_order_quantity == 1
    assert args.fixed_price_offset_ticks == 1
    assert args.reward_inactivity_penalty == 0.0
    assert args.reward_inventory_penalty == 0.0
    assert args.reward_inventory_risk_penalty == 0.0
    assert args.preset == "baseline"


def test_build_parser_help_describes_reward_shaping() -> None:
    help_text = eval_rl_agent.build_parser().format_help()

    assert "Reward shaping" in help_text
    assert "realized_pnl_delta - inactivity_penalty" in help_text
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
            "--inv-penalty",
            "0.1",
            "--inv-risk-penalty",
            "0.05",
        ]
    )

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
        learning_agent_starting_inventory=0.0,
        phase_a_action_space=True,
        include_cancel_action=False,
        fixed_order_quantity=1,
        fixed_price_offset_ticks=1,
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
    assert payload["metadata"]["learning_agent_starting_inventory"] == 0.0
    assert payload["metadata"]["phase_a_action_space"] is True
    assert payload["metadata"]["include_cancel_action"] is False
    assert payload["metadata"]["fixed_order_quantity"] == 1
    assert payload["metadata"]["fixed_price_offset_ticks"] == 1
    assert payload["metadata"]["reward_inactivity_penalty"] == 0.2
    assert payload["metadata"]["reward_inventory_penalty"] == 0.1
    assert payload["metadata"]["reward_inventory_risk_penalty"] == 0.05
    assert payload["metadata"]["reward_signal"] == "realized_pnl_delta"
    assert payload["metadata"]["reward_base_term"] == "realized_pnl_delta"
    assert payload["metadata"]["reward_formula"] == (
        "realized_pnl_delta - inactivity_penalty(if no trade) - abs(inventory) * reward_inventory_penalty - "
        "inventory^2 * reward_inventory_risk_penalty"
    )
    assert payload["metadata"]["reward_summary"] == (
        "realized_pnl_delta - 0.2 * inactivity(if no trade) - 0.1 * abs(inventory) - 0.05 * inventory^2"
    )
    assert payload["metadata"]["reward_shaping"]["inactivity_penalty"]["coefficient"] == 0.2
    assert payload["metadata"]["reward_shaping"]["linear_inventory_penalty"]["coefficient"] == 0.1
    assert payload["metadata"]["reward_shaping"]["quadratic_inventory_risk_penalty"]["coefficient"] == 0.05
    assert payload["summary"]["trade_count"] >= 0
    assert isinstance(payload["portfolio_breakdown"], list)
    assert payload["agents"] == payload["portfolio_breakdown"]


def test_build_rl_evaluation_payload_adjusts_learning_slot_starting_inventory(tmp_path: Path) -> None:
    config = build_preset_config("baseline")
    simulator = SyntheticMarketSimulator(config, horizon=24)
    result = simulator.run(horizon=24)

    payload = eval_rl_agent.build_rl_evaluation_payload(
        checkpoint_path=tmp_path / "ppo.zip",
        algorithm="ppo",
        preset_name="baseline",
        learning_agent_id="trend_01",
        learning_agent_starting_inventory=0.0,
        phase_a_action_space=True,
        include_cancel_action=False,
        fixed_order_quantity=1,
        fixed_price_offset_ticks=1,
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
