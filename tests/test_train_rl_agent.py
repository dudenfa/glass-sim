from __future__ import annotations

from pathlib import Path

import pytest

from scripts import train_rl_agent


def test_parse_args_defaults_to_trend_slot() -> None:
    args = train_rl_agent.parse_args(["--total-timesteps", "128"])

    assert args.preset == "baseline"
    assert args.learning_agent_id == "trend_01"
    assert args.learning_agent_starting_inventory == 0.0
    assert args.total_timesteps == 128


def test_default_output_model_uses_preset_and_agent() -> None:
    checkpoint = train_rl_agent.default_checkpoint_path("baseline", "trend_01")

    assert checkpoint.name == "ppo_baseline_trend_01.zip"
    assert checkpoint.parent.name == "checkpoints"


def test_main_fails_cleanly_when_rl_dependencies_are_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise() -> tuple[object, object]:
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
