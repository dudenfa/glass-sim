from __future__ import annotations

from dataclasses import dataclass


def _validate_non_empty(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")
    return normalized


@dataclass(frozen=True)
class AgentId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_non_empty(self.value, "AgentId"))


@dataclass(frozen=True)
class AssetSymbol:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_non_empty(self.value, "AssetSymbol"))


@dataclass(frozen=True)
class OrderId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_non_empty(self.value, "OrderId"))


@dataclass(frozen=True)
class SimulationId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_non_empty(self.value, "SimulationId"))
