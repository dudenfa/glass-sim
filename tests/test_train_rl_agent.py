from __future__ import annotations

from pathlib import Path

import pytest

from scripts import train_rl_agent


def test_parse_args_defaults_to_trend_slot() -> None:
    args = train_rl_agent.parse_args(["--total-timesteps", "128"])

    assert args.algorithm == "ppo"
    assert args.preset == "baseline"
    assert args.learning_agent_id == "trend_01"
    assert args.add_learning_agent is False
    assert args.learning_agent_template_id is None
    assert args.learning_agent_starting_inventory == 0.0
    assert args.frozen_agent_checkpoint is None
    assert args.frozen_agent_id is None
    assert args.add_frozen_agent is False
    assert args.frozen_agent_template_id is None
    assert args.frozen_agent_starting_inventory is None
    assert args.train_seeds is None
    assert args.phase_a_action_space is True
    assert args.include_cancel_action is False
    assert args.fixed_order_quantity == 1
    assert args.fixed_price_offset_ticks == 1
    assert args.reward_equity_delta_coefficient == 0.0
    assert args.reward_inactivity_penalty == 0.0
    assert args.reward_inventory_penalty == 0.0
    assert args.reward_inventory_risk_penalty == 0.0
    assert args.total_timesteps == 128


def test_build_parser_help_describes_reward_shaping() -> None:
    help_text = train_rl_agent.build_parser().format_help()

    assert "Reward shaping" in help_text
    assert "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta" in help_text
    assert "--reward-equity-delta-coefficient" in help_text
    assert "--reward-inactivity-penalty" in help_text
    assert "--inv-penalty" in help_text
    assert "--reward-inventory-risk-penalty" in help_text


def test_parse_args_accepts_short_reward_aliases() -> None:
    args = train_rl_agent.parse_args(
        [
            "--total-timesteps",
            "128",
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
    args = train_rl_agent.parse_args(
        [
            "--total-timesteps",
            "128",
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


def test_parse_args_accepts_multi_seed_schedule() -> None:
    args = train_rl_agent.parse_args(
        [
            "--total-timesteps",
            "128",
            "--train-seeds",
            "1,2,7,8",
        ]
    )

    assert args.train_seeds == "1,2,7,8"
    assert train_rl_agent.parse_seed_schedule(args.train_seeds) == (1, 2, 7, 8)


def test_parse_args_accepts_add_learning_agent_mode() -> None:
    args = train_rl_agent.parse_args(
        [
            "--total-timesteps",
            "128",
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
    args = train_rl_agent.parse_args(
        [
            "--total-timesteps",
            "128",
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


def test_default_output_model_uses_preset_and_agent() -> None:
    checkpoint = train_rl_agent.default_checkpoint_path("baseline", "trend_01")

    assert checkpoint.name == "ppo_baseline_trend_01.zip"
    assert checkpoint.parent.name == "checkpoints"


def test_main_fails_cleanly_when_rl_dependencies_are_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise(_algorithm: str) -> tuple[object, object]:
        raise RuntimeError("missing optional RL deps")

    monkeypatch.setattr(train_rl_agent, "import_ppo_stack", _raise)

    with pytest.raises(SystemExit) as exc_info:
        train_rl_agent.main(
            [
                "--total-timesteps",
                "16",
                "--checkpoint",
                str(tmp_path / "ppo_test.zip"),
            ]
        )

    assert exc_info.value.code == 1


def test_build_training_metadata_includes_inventory_risk_penalty(tmp_path: Path) -> None:
    args = train_rl_agent.parse_args(
        [
            "--total-timesteps",
            "128",
            "--reward-inactivity-penalty",
            "0.2",
            "--reward-equity-delta-coefficient",
            "0.15",
            "--reward-inventory-penalty",
            "0.1",
            "--reward-inventory-risk-penalty",
            "0.05",
        ]
    )
    config, horizon = train_rl_agent.build_training_config(args.preset, seed=args.seed, horizon=args.horizon)
    metadata = train_rl_agent.build_training_metadata(
        args=args,
        config=config,
        effective_horizon=horizon,
        checkpoint_path=tmp_path / "ppo.zip",
    )

    assert metadata["reward_equity_delta_coefficient"] == 0.15
    assert metadata["reward_inactivity_penalty"] == 0.2
    assert metadata["reward_inventory_penalty"] == 0.1
    assert metadata["reward_inventory_risk_penalty"] == 0.05
    assert metadata["algorithm"] == "ppo"
    assert metadata["phase_a_action_space"] is True
    assert metadata["include_cancel_action"] is False
    assert metadata["fixed_order_quantity"] == 1
    assert metadata["fixed_price_offset_ticks"] == 1
    assert metadata["reward_signal"] == "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta"
    assert metadata["reward_base_term"] == "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta"
    assert metadata["reward_formula"] == (
        "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta - inactivity_penalty(if no trade) - "
        "abs(inventory) * reward_inventory_penalty - inventory^2 * reward_inventory_risk_penalty"
    )
    assert metadata["reward_summary"] == (
        "realized_pnl_delta + 0.15 * equity_delta - 0.2 * inactivity(if no trade) - 0.1 * abs(inventory) - 0.05 * inventory^2"
    )
    assert metadata["reward_shaping"]["equity_delta"]["coefficient"] == 0.15
    assert metadata["reward_shaping"]["inactivity_penalty"]["coefficient"] == 0.2
    assert metadata["reward_shaping"]["linear_inventory_penalty"]["coefficient"] == 0.1
    assert metadata["reward_shaping"]["quadratic_inventory_risk_penalty"]["coefficient"] == 0.05
    assert metadata["train_seeds"] == []
    assert metadata["add_learning_agent"] is False
    assert metadata["learning_agent_template_id"] is None
    assert metadata["frozen_agent_checkpoint"] is None
    assert metadata["frozen_agent_id"] is None
    assert metadata["add_frozen_agent"] is False
    assert metadata["frozen_agent_template_id"] is None
    assert metadata["frozen_agent_starting_inventory"] is None
    assert metadata["runtime_learning_agent_mode"] == "replace"
    assert metadata["runtime_frozen_agent_mode"] is None


def test_build_training_metadata_includes_multi_seed_schedule(tmp_path: Path) -> None:
    args = train_rl_agent.parse_args(
        [
            "--total-timesteps",
            "128",
            "--train-seeds",
            "3,4,9",
        ]
    )
    config, horizon = train_rl_agent.build_training_config(args.preset, seed=args.seed, horizon=args.horizon)
    metadata = train_rl_agent.build_training_metadata(
        args=args,
        config=config,
        effective_horizon=horizon,
        checkpoint_path=tmp_path / "ppo.zip",
    )

    assert metadata["train_seeds"] == [3, 4, 9]


def test_build_training_config_can_add_learning_agent() -> None:
    config, horizon = train_rl_agent.build_training_config(
        "baseline",
        learning_agent_id="rl_01",
        add_learning_agent=True,
        learning_agent_template_id="trend_01",
    )

    assert horizon == config.market.event_horizon
    agent_ids = [agent.agent_id.value for agent in config.agents]
    assert agent_ids == ["maker_01", "retail_01", "informed_01", "trend_01", "rl_01"]
    cloned = next(agent for agent in config.agents if agent.agent_id.value == "rl_01")
    assert cloned.agent_type == "trend_follower"


def test_build_training_config_can_add_learning_and_frozen_agents() -> None:
    config, horizon = train_rl_agent.build_training_config(
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
