from __future__ import annotations

from pathlib import Path

from marl_trading.configs import build_preset_config
from marl_trading.market.simulator import SyntheticMarketSimulator
from scripts import eval_rl_agent


def test_parse_args_requires_checkpoint() -> None:
    args = eval_rl_agent.parse_args(["--checkpoint", "model.zip"])

    assert args.checkpoint == Path("model.zip")
    assert args.learning_agent_id == "trend_01"
    assert args.learning_agent_starting_inventory == 0.0
    assert args.preset == "baseline"


def test_normalize_checkpoint_load_path_strips_zip_suffix(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ppo_test.zip"

    normalized = eval_rl_agent._normalize_checkpoint_load_path(checkpoint)

    assert normalized.endswith("ppo_test")
    assert not normalized.endswith(".zip")


def test_build_rl_evaluation_payload_matches_market_health_shape(tmp_path: Path) -> None:
    config = build_preset_config("baseline")
    simulator = SyntheticMarketSimulator(config, horizon=24)
    result = simulator.run(horizon=24)

    payload = eval_rl_agent.build_rl_evaluation_payload(
        checkpoint_path=tmp_path / "ppo.zip",
        preset_name="baseline",
        learning_agent_id="trend_01",
        learning_agent_starting_inventory=0.0,
        result=result,
        config=config,
        horizon=24,
        deterministic=True,
        open_orders_by_agent={"trend_01": 1},
    )

    assert payload["preset"] == "baseline"
    assert payload["label"] == "baseline_rl"
    assert payload["metadata"]["learning_agent_id"] == "trend_01"
    assert payload["metadata"]["learning_agent_starting_inventory"] == 0.0
    assert payload["summary"]["trade_count"] >= 0
    assert isinstance(payload["portfolio_breakdown"], list)
    assert payload["agents"] == payload["portfolio_breakdown"]
