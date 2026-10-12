"""Save a run and replay it later, exactly, in the viewer or in code.

GLASS is deterministic: the same configuration and seed produce the same
market, event for event. A replay file therefore stores the *recipe* of a run
instead of its states:

- the ``MarketConfig`` (scenario, population, every parameter, the seed);
- agents added by hand after the market was created (``add_agent``), such as a
  trained RL agent, stored by class and constructor arguments or by checkpoint
  path;
- every call made from outside the simulation, in order and tied to the exact
  event at which it happened: decisions sent with ``act()`` (Gymnasium and
  PettingZoo environments, custom loops), ``execute()`` and ``shock_index()``;
- a fingerprint of the final state (events, trades, prices, fees, ...), so a
  replay can check that it reproduced the original run.

A replay file is small (kilobytes) and rebuilds everything: the order book with
owners, every account, every trade, at any moment.

    result = PerpetualMarket(config).run()
    result.save("runs/my_run")                 # writes replay.json next to the report
    replay = Replay.load("runs/my_run")
    again = replay.run()                       # identical SimulationResult
    replay.verify(again)                       # {"identical": True, ...}

    glass serve --replay runs/my_run           # watch it, pause, seek

Loading a file never runs arbitrary code: agent classes are only accepted if
they are ``Agent`` subclasses, and every other object must come from GLASS's
own modules. Agent classes must be importable, so agents defined in a notebook
or in the script that runs the simulation (module ``__main__``) make a run
non-replayable; move them to a module to replay such runs. RL checkpoints are
referenced by path and must exist where the replay is loaded.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import json
import time as wallclock
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from .agents.base import Agent

if TYPE_CHECKING:  # pragma: no cover
    from .market import PerpetualMarket
    from .results import SimulationResult

FORMAT = "glass-replay"
FORMAT_VERSION = 1
FILE_NAME = "replay.json"


class ReplayUnavailable(Exception):
    """The run contains something a replay file cannot describe (the message says what)."""


class ReplayError(Exception):
    """A replay file is invalid, refers to something unsafe or missing, or diverged from the original run."""


def _glass_version() -> str:
    try:
        from importlib.metadata import version

        return version("glass-sim")
    except Exception:  # pragma: no cover - running from a source tree without installation
        return "unknown"


# =============================================================================
# Restricted JSON codec
# =============================================================================


def _path_of(obj: Any) -> str:
    return f"{obj.__module__}:{obj.__qualname__}"


def _is_glass(module: str) -> bool:
    return module == "glass_sim" or module.startswith("glass_sim.")


def encode(value: Any) -> Any:
    """JSON-ready form of configs, actions and constructor arguments; raises ``ReplayUnavailable`` otherwise."""
    # Enums first: GLASS enums such as Side and Transparency are also str subclasses.
    if isinstance(value, enum.Enum):
        if not _is_glass(type(value).__module__):
            raise ReplayUnavailable(f"enum {_path_of(type(value))} is not part of GLASS")
        return {"__enum__": _path_of(type(value)), "value": value.value}
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value.item() if isinstance(value, np.generic) else value
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ReplayUnavailable("dictionaries with non-string keys cannot be saved")
        return {"__dict__": {k: encode(v) for k, v in value.items()}}
    if isinstance(value, type) and issubclass(value, Agent):
        if value.__module__ == "__main__":
            raise ReplayUnavailable(
                f"agent class {value.__qualname__} is defined in __main__ (a notebook or the running script); move it to an importable module to make the run replayable"
            )
        if "<locals>" in value.__qualname__:
            raise ReplayUnavailable(f"agent class {value.__qualname__} is defined inside a function; define it at module level to make the run replayable")
        return {"__agent_class__": _path_of(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if not _is_glass(type(value).__module__):
            raise ReplayUnavailable(f"{_path_of(type(value))} is not part of GLASS")
        fields = {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
        return {"__dataclass__": _path_of(type(value)), "fields": fields}
    raise ReplayUnavailable(f"cannot save a value of type {type(value).__module__}.{type(value).__qualname__}")


def _resolve(path: str) -> Any:
    module_name, _, qualname = path.partition(":")
    try:
        obj = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise ReplayError(f"cannot import {path}: {exc}") from exc
    return obj


def decode(data: Any) -> Any:
    """Inverse of ``encode``; refuses anything that is not a GLASS object or an ``Agent`` subclass."""
    if isinstance(data, list):
        return [decode(v) for v in data]
    if not isinstance(data, dict):
        return data
    if "__tuple__" in data:
        return tuple(decode(v) for v in data["__tuple__"])
    if "__dict__" in data:
        return {k: decode(v) for k, v in data["__dict__"].items()}
    if "__agent_class__" in data:
        cls = _resolve(data["__agent_class__"])
        if not (isinstance(cls, type) and issubclass(cls, Agent)):
            raise ReplayError(f"refusing {data['__agent_class__']}: not an Agent subclass")
        return cls
    if "__enum__" in data:
        if not _is_glass(data["__enum__"].partition(":")[0]):
            raise ReplayError(f"refusing {data['__enum__']}: not part of GLASS")
        cls = _resolve(data["__enum__"])
        if not (isinstance(cls, type) and issubclass(cls, enum.Enum)):
            raise ReplayError(f"refusing {data['__enum__']}: not an enum")
        return cls(data["value"])
    if "__dataclass__" in data:
        if not _is_glass(data["__dataclass__"].partition(":")[0]):
            raise ReplayError(f"refusing {data['__dataclass__']}: not part of GLASS")
        cls = _resolve(data["__dataclass__"])
        if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
            raise ReplayError(f"refusing {data['__dataclass__']}: not a dataclass")
        return cls(**{k: decode(v) for k, v in data["fields"].items()})
    raise ReplayError(f"unrecognized entry in replay file: {sorted(data)}")


# =============================================================================
# Agents added by hand
# =============================================================================


def _policy_classes():
    """The RL policy classes, or ``None`` when the RL extras are not installed (then no ``PolicyAgent`` can exist)."""
    try:
        from .rl.policy import HoldPolicy, ModelPolicy, PolicyAgent, RandomPolicy
    except ImportError:
        return None
    return HoldPolicy, ModelPolicy, PolicyAgent, RandomPolicy


def _agent_spec(agent: Agent) -> dict[str, Any]:
    """How to rebuild an agent added with ``add_agent``."""
    classes = _policy_classes()
    if classes is not None and isinstance(agent, classes[2]):
        HoldPolicy, ModelPolicy, PolicyAgent, RandomPolicy = classes
        policy = agent.policy
        if isinstance(policy, ModelPolicy):
            policy_spec = {"model": str(policy.path.resolve()), "deterministic": bool(policy.deterministic)}
        elif isinstance(policy, (HoldPolicy, RandomPolicy)):
            policy_spec = {"builtin": policy.name}
        else:
            raise ReplayUnavailable(
                f"PolicyAgent {agent.agent_id} uses a custom policy ({type(policy).__name__}); only checkpoints and the built-in policies can be saved"
            )
        for part in (agent.encoder, agent.actions):
            if not _is_glass(type(part).__module__) or not hasattr(part, "config"):
                raise ReplayUnavailable(f"PolicyAgent {agent.agent_id} uses a custom {type(part).__name__}")
        return {
            "policy_agent": {
                "policy": policy_spec,
                "wake_interval": agent.wake_interval,
                "encoder": encode(agent.encoder.config),
                "actions": encode(agent.actions.config),
                "agent_type": agent.agent_type,
            }
        }
    args, kwargs = getattr(agent, "_init_args", ((), {}))
    try:
        encoded_args = encode(list(args)), encode(dict(kwargs))
    except ReplayUnavailable as exc:
        raise ReplayUnavailable(f"agent {agent.agent_id} ({type(agent).__name__}) was built with arguments that cannot be saved: {exc}") from exc
    return {"class": encode(type(agent)), "args": encoded_args[0], "kwargs": encoded_args[1], "agent_type": agent.agent_type}


def _build_agent(spec: dict[str, Any]) -> Agent:
    if "policy_agent" in spec:
        try:
            from .rl.actions import make_action_space
            from .rl.observation import ObservationEncoder
            from .rl.policy import HoldPolicy, ModelPolicy, PolicyAgent, RandomPolicy
        except ImportError as exc:
            raise ReplayError(f"this run contains an RL agent; replaying it needs the RL extras (pip install glass-sim[rl]): {exc}") from exc

        p = spec["policy_agent"]
        policy_spec = p["policy"]
        if "model" in policy_spec:
            if not Path(policy_spec["model"]).exists():
                raise ReplayError(f"RL checkpoint not found: {policy_spec['model']}")
            policy = ModelPolicy(policy_spec["model"], policy_spec["deterministic"])
        else:
            policy = {"hold": HoldPolicy, "random": RandomPolicy}[policy_spec["builtin"]]()
        return PolicyAgent(
            policy,
            wake_interval=p["wake_interval"],
            encoder=ObservationEncoder(decode(p["encoder"])),
            actions=make_action_space(decode(p["actions"])),
            agent_type=p["agent_type"],
        )
    cls = decode(spec["class"])
    agent = cls(*decode(spec["args"]), **decode(spec["kwargs"]))
    agent.agent_type = spec["agent_type"]
    return agent


# =============================================================================
# Replay file
# =============================================================================


def _fingerprint(market: PerpetualMarket) -> dict[str, Any]:
    """Final-state numbers a faithful replay must reproduce exactly."""
    return {
        "events": market.events_processed,
        "time": market.time,
        "trades": len(market.recorder.trades),
        "last_price": market.last_price,
        "mark_price": market.mark_price,
        "open_interest": market.open_interest(),
        "fees_collected": market.fees_collected,
        "insurance_fund": market.insurance_fund,
        "liquidations": market.liquidation_count,
    }


@dataclasses.dataclass
class Replay:
    """The recipe of a run; ``build()`` / ``run()`` re-create it, ``verify`` checks the result."""

    data: dict[str, Any]
    source: Optional[str] = None

    # ----------------------------------------------------------- creation

    @classmethod
    def from_market(cls, market: PerpetualMarket) -> Replay:
        """Describe ``market`` as it is now; raises ``ReplayUnavailable`` if something cannot be saved."""
        added = []
        for entry in market.added_agents:
            agent = entry["agent"]
            added.append(
                {
                    "agent_id": agent.agent_id,
                    "collateral": entry["collateral"],
                    "private_signals": entry["private_signals"],
                    "external": entry["external"],
                    **_agent_spec(agent),
                }
            )
        log = []
        for item in market.control_log:
            row = {k: v for k, v in item.items() if k != "actions"}
            if "actions" in item:
                row["actions"] = encode(item["actions"])
            log.append(row)
        data = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "glass_version": _glass_version(),
            "config": encode(market.config),
            "keep_order_events": market.keep_order_events,
            "explicit_agents": len(market._explicit_agents) if market._explicit_agents is not None else None,
            "added_agents": added,
            "control_log": log,
            "end": {"events": market.events_processed, "time": market.time, "finished": market.finished},
            "fingerprint": _fingerprint(market),
        }
        return cls(data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Replay:
        """Read a replay file, or the ``replay.json`` inside a saved-run directory."""
        path = Path(path)
        if path.is_dir():
            path = path / FILE_NAME
        if not path.exists():
            raise ReplayError(f"no replay file at {path}")
        data = json.loads(path.read_text())
        if data.get("format") != FORMAT:
            raise ReplayError(f"{path} is not a GLASS replay file")
        if data.get("format_version", 0) > FORMAT_VERSION:
            raise ReplayError(f"{path} was written by a newer GLASS (format {data['format_version']})")
        # Validate now, so unsafe or missing classes are reported at load time.
        decode(data["config"])
        for spec in data["added_agents"]:
            if "class" in spec:
                decode(spec["class"])
        return cls(data, source=str(path))

    def save(self, path: Union[str, Path]) -> Path:
        """Write the replay file (a directory gets ``replay.json`` inside it)."""
        path = Path(path)
        if path.suffix != ".json":
            path.mkdir(parents=True, exist_ok=True)
            path = path / FILE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, separators=(",", ":")))
        return path

    # ----------------------------------------------------------- contents

    @property
    def config(self):
        """The ``MarketConfig`` of the run."""
        return decode(self.data["config"])

    @property
    def end_time(self) -> float:
        return float(self.data["end"]["time"])

    @property
    def fingerprint(self) -> dict[str, Any]:
        return dict(self.data["fingerprint"])

    def describe(self) -> str:
        """One line for logs and the viewer."""
        config = self.config
        hand = len(self.data["added_agents"])
        decisions = sum(1 for e in self.data["control_log"] if e["kind"] == "act")
        interventions = len(self.data["control_log"]) - decisions
        return (
            f"{config.name} seed {config.seed}, {config.transparency.value}, {self.end_time / 3600:.2f} h"
            f"{f', {hand} added agent(s)' if hand else ''}{f', {decisions} external decisions' if decisions else ''}"
            f"{f', {interventions} intervention(s)' if interventions else ''}"
        )

    # ------------------------------------------------------------ running

    def build(self) -> ReplayRun:
        """A fresh market positioned at the start of the run, with a driver that re-applies the log."""
        return ReplayRun(self)

    def run(self) -> SimulationResult:
        """Replay to the end and return the result."""
        return self.build().run()

    def verify(self, target: Union[PerpetualMarket, SimulationResult, ReplayRun]) -> dict[str, Any]:
        """Compare a replayed market with the fingerprint of the original run."""
        if isinstance(target, ReplayRun):
            target = target.market
        if hasattr(target, "recorder") and hasattr(target, "accounts") and not hasattr(target, "step"):
            current = target.replay.fingerprint if target.replay is not None else None
            if current is None:
                raise ReplayError("this result has no replay information to compare")
        else:
            current = _fingerprint(target)
        differences = {}
        for key, expected in self.fingerprint.items():
            got = current.get(key)
            same = got == expected if not isinstance(expected, float) else got is not None and abs(got - expected) <= 1e-9 * max(1.0, abs(expected))
            if not same:
                differences[key] = {"original": expected, "replay": got}
        recorded = self.data.get("glass_version")
        return {"identical": not differences, "differences": differences, "recorded_with": recorded, "replayed_with": _glass_version()}


class ReplayRun:
    """A market being replayed: ``step`` re-applies the recorded outside calls at their exact events."""

    def __init__(self, replay: Replay) -> None:
        from .market import PerpetualMarket

        self.replay = replay
        data = replay.data
        config = replay.config
        added = data["added_agents"]
        explicit = data.get("explicit_agents")
        agents = [_build_agent(spec) for spec in added]
        for agent, spec in zip(agents, added):
            agent.agent_id = spec["agent_id"]
        if explicit is not None:
            pairs = [(agent, spec["collateral"]) for agent, spec in zip(agents[:explicit], added[:explicit])]
            market = PerpetualMarket(config, agents=pairs, keep_order_events=data.get("keep_order_events", False))
            rest = list(zip(agents[explicit:], added[explicit:]))
        else:
            market = PerpetualMarket(config, keep_order_events=data.get("keep_order_events", False))
            rest = list(zip(agents, added))
        for agent, spec in rest:
            market.add_agent(agent, spec["collateral"], private_signals=spec["private_signals"], external=spec["external"])
        self.market = market
        self._log = data["control_log"]
        self._next = 0
        self._end_events = int(data["end"]["events"])

    # ---------------------------------------------------------------- state

    @property
    def done(self) -> bool:
        """True once the replay has reached the point where the original run was saved."""
        return self.market.events_processed >= self._end_events and self._next >= len(self._log)

    @property
    def end_time(self) -> float:
        return self.replay.end_time

    def _control_due(self) -> bool:
        if self._next >= len(self._log):
            return False
        entry = self._log[self._next]
        return entry["events"] == self.market.events_processed and (entry["kind"] != "act" or self.market.pending_decision is not None)

    # ------------------------------------------------------------- stepping

    def step(self) -> bool:
        """Apply the next recorded call or process the next event; False when the replay is complete."""
        market = self.market
        if self._control_due():
            entry = self._log[self._next]
            self._next += 1
            kind = entry["kind"]
            if kind == "act":
                if market.pending_decision != entry["agent_id"]:
                    raise ReplayError(f"replay diverged: expected a decision by {entry['agent_id']}, market waits for {market.pending_decision}")
                market.act(entry["agent_id"], decode(entry["actions"]))
            else:
                market.time = max(market.time, float(entry["time"]))
                if kind == "shock":
                    market.shock_index(entry["log_jump"], headline=entry["headline"])
                elif kind == "execute":
                    market.execute(entry["agent_id"], decode(entry["actions"]))
                else:
                    raise ReplayError(f"unknown entry {kind!r} in replay file")
            return True
        if market.pending_decision is not None:
            if market.events_processed >= self._end_events:
                return False
            raise ReplayError(f"replay diverged: {market.pending_decision} must decide but the file has no matching decision")
        if market.events_processed >= self._end_events or market.finished:
            return False
        market.step()
        return True

    def run_until(self, time: float, deadline: Optional[float] = None) -> bool:
        """Replay up to simulated ``time``; with ``deadline`` (a ``perf_counter`` value) stop early. True if ``time`` was reached."""
        market = self.market
        market._start()  # schedules the first events; idempotent, as in PerpetualMarket.run_until
        while True:
            if deadline is not None and wallclock.perf_counter() > deadline:
                return False
            if self._control_due() or (market.pending_decision is None and market._queue and market._queue[0][0] <= time and not self.done):
                if not self.step():
                    break
                continue
            break
        if not self.done and market.pending_decision is None:
            market.time = max(market.time, min(time, self.end_time))
        return True

    def run(self) -> SimulationResult:
        """Replay to the end and return the result."""
        while self.step():
            pass
        return self.market.results()

    def verify(self) -> dict[str, Any]:
        """Compare with the original run's fingerprint (meaningful once ``done``)."""
        return self.replay.verify(self.market)


def load(path: Union[str, Path]) -> Replay:
    """Shortcut for ``Replay.load``."""
    return Replay.load(path)
