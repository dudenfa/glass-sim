"""Interfaces of the three pluggable parts of a learning environment.

``PerpTradingEnv`` (and the PettingZoo wrapper) build their observation,
action menu and reward from ``EnvConfig`` by default, but each part can be
replaced by any object that implements the matching protocol below, without
editing the library:

- an **observation encoder** turns the agent's ``MarketView`` into a vector;
- an **action space** lists the discrete actions, says which are valid now and
  turns an action index into orders;
- a **reward factory** is called once per episode with the market and the
  agent id and returns a tracker whose ``step()`` gives the reward since the
  previous decision.

The built-in implementations are ``ObservationEncoder``, ``ActionSpace`` /
``LadderActionSpace`` and ``RewardTracker``; subclassing them is often the
easiest way to start.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..market import PerpetualMarket
    from ..types import Action
    from ..views import MarketView


@runtime_checkable
class Encoder(Protocol):
    """Observation encoder. Optional attributes: ``feature_names`` (list of str)
    and ``bounds`` (low, high) for the observation box, default (-5, 5)."""

    @property
    def size(self) -> int: ...

    def encode(self, view: MarketView) -> np.ndarray: ...


@runtime_checkable
class Actions(Protocol):
    """Discrete action menu with validity masks. Index 0 must always be valid
    (conventionally ``hold``) so the agent can never be stuck."""

    names: list[str]

    @property
    def n(self) -> int: ...

    def mask(self, view: MarketView) -> np.ndarray: ...

    def decode(self, index: int, view: MarketView) -> list[Action]: ...


@runtime_checkable
class Reward(Protocol):
    """Per-episode reward tracker: ``reset()`` once the agent first decides,
    then ``step()`` after each decision returns ``(reward, info)``."""

    def reset(self) -> None: ...

    def step(self) -> tuple[float, dict[str, Any]]: ...


RewardFactory = Callable[["PerpetualMarket", str], Reward]
"""Builds the reward tracker of one agent for a new episode."""
