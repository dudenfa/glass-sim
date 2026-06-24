from __future__ import annotations

from scripts import train_marl_agents


def test_parse_ids_requires_three_unique_ids() -> None:
    assert train_marl_agents.parse_ids("marl_01,marl_02,marl_03") == ("marl_01", "marl_02", "marl_03")


def test_parser_accepts_frozen_opponents_for_ten_agent_training() -> None:
    args = train_marl_agents.build_parser().parse_args(
        [
            "--frozen-agent",
            "rl_01:checkpoints/rl_01.zip:trend_01:16",
            "--frozen-agent",
            "rl_02:checkpoints/rl_02.zip:trend_01:16",
        ]
    )

    specs = train_marl_agents.resolve_frozen_agent_specs(args)

    assert [spec.agent_id for spec in specs] == ["rl_01", "rl_02"]
    assert [spec.starting_inventory for spec in specs] == [16.0, 16.0]
