from __future__ import annotations

from scripts import run_agent_ranking_panel as panel


def test_parse_seeds_supports_ranges_and_preserves_order() -> None:
    assert panel.parse_seeds("31-33,35,32") == (31, 32, 33, 35)


def test_rl03_only_structure_builds_an_endowed_learning_agent_command() -> None:
    spec = next(item for item in panel.STRUCTURES if item.structure_id == "s1_rl03")

    argv = panel._eval_argv(spec, seed=31, horizon=100, preset="baseline", rl_inventory=16.0)

    assert argv[argv.index("--learning-agent-id") + 1] == "rl_03"
    assert argv[argv.index("--learning-agent-starting-inventory") + 1] == "16.0"
    assert "--frozen-agent" not in argv


def test_three_rl_structure_uses_both_endowed_frozen_opponents() -> None:
    spec = next(item for item in panel.STRUCTURES if item.structure_id == "s3_rl01_rl02_rl03")

    argv = panel._eval_argv(spec, seed=31, horizon=100, preset="baseline", rl_inventory=16.0)

    frozen_specs = [argv[index + 1] for index, value in enumerate(argv) if value == "--frozen-agent"]
    assert len(frozen_specs) == 2
    assert frozen_specs[0].startswith("rl_01:") and frozen_specs[0].endswith(":trend_01:16.0")
    assert frozen_specs[1].startswith("rl_02:") and frozen_specs[1].endswith(":trend_01:16.0")
