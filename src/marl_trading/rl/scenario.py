from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from marl_trading.core.config import AgentConfig, SimulationConfig
from marl_trading.core.domain import AgentId


@dataclass(frozen=True)
class FrozenAgentSpec:
    agent_id: str
    checkpoint_path: str
    add_agent: bool = False
    template_id: str | None = None
    starting_inventory: float | None = None

    def __post_init__(self) -> None:
        resolved_agent_id = str(self.agent_id).strip()
        resolved_checkpoint = str(self.checkpoint_path).strip()
        if not resolved_agent_id:
            raise ValueError("frozen agent id must be a non-empty string.")
        if not resolved_checkpoint:
            raise ValueError("frozen agent checkpoint path must be a non-empty string.")
        resolved_template = None if self.template_id is None else str(self.template_id).strip() or None
        object.__setattr__(self, "agent_id", resolved_agent_id)
        object.__setattr__(self, "checkpoint_path", resolved_checkpoint)
        object.__setattr__(self, "template_id", resolved_template)
        object.__setattr__(self, "add_agent", bool(self.add_agent))
        if self.starting_inventory is not None:
            object.__setattr__(self, "starting_inventory", float(self.starting_inventory))

    def with_resolved_checkpoint(self) -> "FrozenAgentSpec":
        return FrozenAgentSpec(
            agent_id=self.agent_id,
            checkpoint_path=str(Path(self.checkpoint_path).expanduser().resolve()),
            add_agent=self.add_agent,
            template_id=self.template_id,
            starting_inventory=self.starting_inventory,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "checkpoint_path": self.checkpoint_path,
            "add_agent": bool(self.add_agent),
            "template_id": self.template_id,
            "starting_inventory": self.starting_inventory,
        }


def parse_frozen_agent_spec(raw: str) -> FrozenAgentSpec:
    """Parse AGENT_ID:CHECKPOINT[:TEMPLATE_ID[:STARTING_INVENTORY]]."""
    parts = str(raw).split(":")
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError(
            "--frozen-agent expects AGENT_ID:CHECKPOINT[:TEMPLATE_ID[:STARTING_INVENTORY]]."
        )
    agent_id = parts[0].strip()
    checkpoint_path = parts[1].strip()
    template_id = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
    starting_inventory = None
    if len(parts) == 4 and parts[3].strip():
        starting_inventory = float(parts[3].strip())
    return FrozenAgentSpec(
        agent_id=agent_id,
        checkpoint_path=checkpoint_path,
        add_agent=template_id is not None,
        template_id=template_id,
        starting_inventory=starting_inventory,
    )


def prepare_frozen_agent_configs(
    config: SimulationConfig,
    *,
    frozen_agents: Sequence[FrozenAgentSpec],
) -> SimulationConfig:
    updated = config
    seen: set[str] = set()
    for frozen_agent in frozen_agents:
        if frozen_agent.agent_id in seen:
            raise ValueError(f"Duplicate frozen agent id: {frozen_agent.agent_id}")
        seen.add(frozen_agent.agent_id)
        updated = prepare_frozen_agent_config(
            updated,
            frozen_agent_id=frozen_agent.agent_id,
            add_frozen_agent=bool(frozen_agent.add_agent),
            frozen_agent_template_id=frozen_agent.template_id,
        )
    return updated


def build_frozen_agent_specs(
    *,
    frozen_agents: Sequence[str | FrozenAgentSpec] | None = None,
    frozen_agent_checkpoint_path: str | Path | None = None,
    frozen_agent_id: str | None = None,
    add_frozen_agent: bool = False,
    frozen_agent_template_id: str | None = None,
    frozen_agent_starting_inventory: float | None = None,
) -> tuple[FrozenAgentSpec, ...]:
    specs: list[FrozenAgentSpec] = []
    for raw_spec in frozen_agents or ():
        spec = raw_spec if isinstance(raw_spec, FrozenAgentSpec) else parse_frozen_agent_spec(str(raw_spec))
        specs.append(spec.with_resolved_checkpoint())

    legacy_checkpoint = None if frozen_agent_checkpoint_path is None else str(frozen_agent_checkpoint_path).strip()
    legacy_agent_id = None if frozen_agent_id is None else str(frozen_agent_id).strip()
    legacy_template_id = None if frozen_agent_template_id is None else str(frozen_agent_template_id).strip()
    legacy_has_options = bool(
        legacy_agent_id
        or legacy_checkpoint
        or bool(add_frozen_agent)
        or legacy_template_id
        or frozen_agent_starting_inventory is not None
    )
    if legacy_has_options:
        if not legacy_checkpoint:
            raise ValueError("--frozen-agent-checkpoint is required when configuring a frozen agent.")
        if not legacy_agent_id:
            raise ValueError("--frozen-agent-id is required when --frozen-agent-checkpoint is provided.")
        specs.append(
            FrozenAgentSpec(
                agent_id=legacy_agent_id,
                checkpoint_path=str(Path(legacy_checkpoint).expanduser().resolve()),
                add_agent=bool(add_frozen_agent),
                template_id=legacy_template_id or None,
                starting_inventory=frozen_agent_starting_inventory,
            )
        )

    seen: set[str] = set()
    for spec in specs:
        if spec.agent_id in seen:
            raise ValueError(f"Duplicate frozen agent id: {spec.agent_id}")
        seen.add(spec.agent_id)
    return tuple(specs)


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


def add_price_dumper_config(
    config: SimulationConfig,
    *,
    agent_id: str = "dumper_01",
) -> SimulationConfig:
    """Append an opt-in scripted seller for live market-stress demonstrations."""
    resolved_agent_id = str(agent_id).strip()
    if not resolved_agent_id:
        raise ValueError("price dumper agent id must be a non-empty string.")
    existing_ids = {agent.agent_id.value for agent in config.agents}
    if resolved_agent_id in existing_ids:
        raise ValueError(f"Price dumper agent id already exists: {resolved_agent_id}")
    dumper = AgentConfig(
        agent_id=AgentId(resolved_agent_id),
        agent_type="price_dumper",
        starting_cash=10_000.0,
        ruin_threshold=4_000.0,
        max_resting_orders=4,
    )
    turn_schedule_agent_ids = config.turn_schedule_agent_ids
    if turn_schedule_agent_ids is not None:
        turn_schedule_agent_ids = (*turn_schedule_agent_ids, resolved_agent_id)
    return replace(
        config,
        agents=(*config.agents, dumper),
        turn_schedule_agent_ids=turn_schedule_agent_ids,
    )


def exclude_agents_from_config(
    config: SimulationConfig,
    *,
    excluded_agent_ids: Sequence[str] = (),
    preserve_excluded_turn_slots: bool = False,
) -> SimulationConfig:
    resolved_ids = tuple(
        dict.fromkeys(str(agent_id).strip() for agent_id in excluded_agent_ids if str(agent_id).strip())
    )
    if not resolved_ids:
        return config
    existing_ids = {agent.agent_id.value for agent in config.agents}
    unknown_ids = sorted(set(resolved_ids) - existing_ids)
    if unknown_ids:
        raise KeyError(f"Unknown excluded agent id(s): {', '.join(unknown_ids)}")
    remaining_agents = tuple(agent for agent in config.agents if agent.agent_id.value not in resolved_ids)
    if not remaining_agents:
        raise ValueError("Cannot exclude every participant from the simulation config.")
    turn_schedule_agent_ids = config.turn_schedule_agent_ids
    if preserve_excluded_turn_slots:
        turn_schedule_agent_ids = tuple(agent.agent_id.value for agent in config.agents)
    return replace(config, agents=remaining_agents, turn_schedule_agent_ids=turn_schedule_agent_ids)


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
    turn_schedule_agent_ids = config.turn_schedule_agent_ids
    if turn_schedule_agent_ids is not None:
        turn_schedule_agent_ids = (*turn_schedule_agent_ids, resolved_runtime_agent_id)
    return replace(
        config,
        agents=tuple(config.agents) + (runtime_agent,),
        turn_schedule_agent_ids=turn_schedule_agent_ids,
    )


def _clone_agent_config(agent: AgentConfig, *, agent_id: str) -> AgentConfig:
    return replace(agent, agent_id=AgentId(agent_id))
