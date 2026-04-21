from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from marl_trading.live.server import LiveServerConfig, parse_args, serve_market_view
from marl_trading.live.session import LiveMarketSession


class _FakePolicyAdapter:
    def action_for(self, observation):  # noqa: ARG002
        from marl_trading.rl.live import RuntimePolicyDecision, decode_policy_action

        raw_action = (0, 0, 0)
        return RuntimePolicyDecision(
            features=tuple(),
            raw_action=raw_action,
            rl_action=decode_policy_action(raw_action),
        )


def test_live_server_defaults_are_long_running() -> None:
    config = LiveServerConfig()
    assert config.horizon == 10_000
    assert config.autoplay is True
    assert config.preset == "baseline"


def test_live_server_cli_defaults_to_long_running_horizon(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["serve_market_view.py"])
    args = parse_args()
    assert args.horizon == 10_000
    assert args.preset == "baseline"


def test_live_server_cli_parses_runtime_rl_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_market_view.py",
            "--checkpoint",
            "checkpoints/ppo_baseline_trend_01.zip",
            "--learning-agent-id",
            "trend_01",
        ],
    )
    args = parse_args()
    assert args.checkpoint == Path("checkpoints/ppo_baseline_trend_01.zip")
    assert args.learning_agent_id == "trend_01"
    assert args.learning_agent_starting_inventory == 0.0


def test_live_server_endpoints() -> None:
    try:
        server = serve_market_view(
            LiveServerConfig(host="127.0.0.1", port=0, horizon=10, speed=10.0, autoplay=False),
        )
    except PermissionError as exc:  # pragma: no cover - sandbox-dependent
        pytest.skip(f"Socket binding is not permitted in this environment: {exc}")
    try:
        base_url = server.url
        with urlopen(f"{base_url}/app.js", timeout=5) as response:
            assert response.status == 200
        with urlopen(f"{base_url}/api/state", timeout=5) as response:
            state = json.loads(response.read().decode("utf-8"))
        assert state["session"]["step_index"] == 0
        assert state["session"]["status"] == "paused"

        request = Request(
            f"{base_url}/api/control",
            data=json.dumps({"action": "step"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            stepped = json.loads(response.read().decode("utf-8"))
        assert stepped["session"]["step_index"] == 1

        request = Request(
            f"{base_url}/api/control",
            data=json.dumps({"action": "speed", "speed": 2.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            speed_state = json.loads(response.read().decode("utf-8"))
        assert speed_state["session"]["steps_per_second"] == 2.0
    finally:
        server.stop()


def test_live_server_passes_runtime_rl_config_into_session(monkeypatch) -> None:
    fake_policy = _FakePolicyAdapter()
    monkeypatch.setattr(LiveMarketSession, "_load_ppo_policy", lambda self, path: fake_policy)

    try:
        server = serve_market_view(
            LiveServerConfig(
                host="127.0.0.1",
                port=0,
                horizon=10,
                speed=10.0,
                autoplay=False,
                checkpoint_path=Path("/tmp/fake_model.zip"),
                learning_agent_id="trend_01",
            ),
        )
    except PermissionError as exc:  # pragma: no cover - sandbox-dependent
        pytest.skip(f"Socket binding is not permitted in this environment: {exc}")
    try:
        state = server.session.state()
        trend_state = next(agent for agent in state["agents"] if agent["agent_id"] == "trend_01")
        assert trend_state["agent_type"] == "rl_agent"
        assert trend_state["rl_diagnostics"] is not None
    finally:
        server.stop()
