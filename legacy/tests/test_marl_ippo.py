from __future__ import annotations

from marl_trading.configs.defaults import default_simulation_config
from marl_trading.marl import IPPOTrainingConfig, JointIPPORunner, build_joint_training_config


def test_joint_training_config_adds_three_marl_agents_without_mutating_baseline() -> None:
    baseline = default_simulation_config()
    config = build_joint_training_config(
        preset_name="baseline",
        learner_ids=("marl_01", "marl_02", "marl_03"),
        seed=1,
    )

    assert [agent.agent_id.value for agent in baseline.agents] == ["maker_01", "retail_01", "informed_01", "trend_01"]
    assert [agent.agent_id.value for agent in config.agents][-3:] == ["marl_01", "marl_02", "marl_03"]


def test_joint_ippo_smoke_trains_all_three_policies() -> None:
    training = IPPOTrainingConfig(
        learner_ids=("marl_01", "marl_02", "marl_03"),
        horizon=80,
        train_seeds=(1,),
        total_timesteps=4,
        n_steps=4,
        batch_size=4,
    )
    config = build_joint_training_config(
        preset_name="baseline",
        learner_ids=training.learner_ids,
        seed=1,
    )

    result = JointIPPORunner(config, training).train()

    assert result["timesteps_per_agent"] == 4
    assert result["updates"] == 1
    assert result["learner_ids"] == ["marl_01", "marl_02", "marl_03"]
