from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen

from marl_trading.live.server import LiveServerConfig, parse_args, serve_market_view


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


def test_live_server_endpoints() -> None:
    server = serve_market_view(
        LiveServerConfig(host="127.0.0.1", port=0, horizon=10, speed=10.0, autoplay=False),
    )
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
