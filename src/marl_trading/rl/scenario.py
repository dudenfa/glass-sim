from __future__ import annotations

from dataclasses import replace

from marl_trading.core.config import AgentConfig, SimulationConfig
from marl_trading.core.domain import AgentId


def prepare_learning_agent_config(
    config: SimulationConfig,
    *,
    learning_agent_id: str,
    add_learning_agent: bool = False,
    learning_agent_template_id: str | None = None,
) -> SimulationConfig:
    return prepare_runtime_agent_config(
        config,
        runtime_agent_id=learning_agent_id,
        add_runtime_agent=add_learning_agent,
        runtime_agent_template_id=learning_agent_template_id,
    )


def prepare_frozen_agent_config(
    config: SimulationConfig,
    *,
    frozen_agent_id: str,
    add_frozen_agent: bool = False,
    frozen_agent_template_id: str | None = None,
) -> SimulationConfig:
    return prepare_runtime_agent_config(
        config,
        runtime_agent_id=frozen_agent_id,
        add_runtime_agent=add_frozen_agent,
        runtime_agent_template_id=frozen_agent_template_id,
    )


def prepare_runtime_agent_config(
    config: SimulationConfig,
    *,
    runtime_agent_id: str,
    add_runtime_agent: bool = False,
    runtime_agent_template_id: str | None = None,
) -> SimulationConfig:
    resolved_runtime_agent_id = str(runtime_agent_id).strip()
    if not resolved_runtime_agent_id:
        raise ValueError("runtime_agent_id must be a non-empty string.")

    existing_ids = {agent.agent_id.value for agent in config.agents}
    if not add_runtime_agent:
        if resolved_runtime_agent_id not in existing_ids:
            raise KeyError(f"Unknown runtime agent id: {resolved_runtime_agent_id}")
        return config

    if resolved_runtime_agent_id in existing_ids:
        raise ValueError(
            f"Runtime agent id '{resolved_runtime_agent_id}' already exists in the preset. "
            "Choose a fresh id when using add-runtime-agent mode."
        )

    resolved_template_id = str(runtime_agent_template_id or "").strip()
    if not resolved_template_id:
        raise ValueError("runtime_agent_template_id is required when add_runtime_agent is enabled.")

    template_agent = next((agent for agent in config.agents if agent.agent_id.value == resolved_template_id), None)
    if template_agent is None:
        raise KeyError(f"Unknown learning agent template id: {resolved_template_id}")

    runtime_agent = _clone_agent_config(template_agent, agent_id=resolved_runtime_agent_id)
    return replace(config, agents=tuple(config.agents) + (runtime_agent,))


def _clone_agent_config(agent: AgentConfig, *, agent_id: str) -> AgentConfig:
    return replace(agent, agent_id=AgentId(agent_id))
