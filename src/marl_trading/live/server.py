from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from marl_trading.configs import available_preset_names
from marl_trading.core.config import SimulationConfig
from marl_trading.live.session import LiveMarketSession


STATIC_ROOT = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class LiveServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    seed: int = 7
    horizon: int = 10_000
    speed: float = 4.0
    autoplay: bool = True
    open_browser: bool = False
    preset: str = "baseline"
    simulation_config: SimulationConfig | None = None
    checkpoint_path: Path | None = None
    learning_agent_id: str | None = None
    learning_agent_starting_inventory: float = 0.0


class _MarketViewHandler(BaseHTTPRequestHandler):
    server_version = "MarlTradingLive/0.1"

    def _server(self):
        return getattr(self.server, "market_server")

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_type, _ = mimetypes.guess_type(path.name)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._serve_file(STATIC_ROOT / "index.html")
            return
        if path == "/app.js":
            self._serve_file(STATIC_ROOT / "app.js")
            return
        if path == "/styles.css":
            self._serve_file(STATIC_ROOT / "styles.css")
            return
        if path == "/api/state":
            self._send_json(self._server().session.state())
            return
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown path")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown path")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        action = str(payload.get("action", "")).lower()
        server = self._server()
        if action == "play":
            server.session.play()
        elif action == "pause":
            server.session.pause()
        elif action == "step":
            server.session.step(int(payload.get("steps", 1)))
        elif action == "reset":
            seed = payload.get("seed")
            horizon = payload.get("horizon")
            server.session.reset(
                seed=None if seed is None else int(seed),
                horizon=None if horizon is None else int(horizon),
            )
        elif action == "speed":
            speed = payload.get("speed")
            if speed is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing speed")
                return
            server.session.set_speed(float(speed))
        else:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Unknown action: {action}")
            return

        self._send_json(server.session.state())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class MarketViewServer:
    def __init__(self, config: LiveServerConfig) -> None:
        self.config = config
        self.session = LiveMarketSession(
            config=config.simulation_config,
            horizon=config.horizon,
            step_delay_seconds=1.0 / max(config.speed, 0.1),
            autoplay=False,
            checkpoint_path=config.checkpoint_path,
            learning_agent_id=config.learning_agent_id,
            learning_agent_starting_inventory=config.learning_agent_starting_inventory,
        )
        self.httpd = _ReusableThreadingHTTPServer((config.host, config.port), _MarketViewHandler)
        self.httpd.market_server = self  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.httpd.server_address[1]}"

    def start(self) -> None:
        if self.config.autoplay:
            self.session.play()
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        if self.config.open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        self.session.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def serve_market_view(config: LiveServerConfig) -> MarketViewServer:
    server = MarketViewServer(config)
    server.start()
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local synthetic market live view.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument(
        "--preset",
        choices=available_preset_names(),
        default="baseline",
        help="Named preset from marl_trading.configs.presets.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Simulation seed.")
    parser.add_argument("--horizon", type=int, default=10_000, help="Number of steps in the live demo session.")
    parser.add_argument("--speed", type=float, default=4.0, help="Playback speed in steps per second.")
    parser.add_argument("--paused", action="store_true", help="Start paused instead of autoplaying.")
    parser.add_argument("--open-browser", action="store_true", help="Open the viewer in the default browser.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional PPO checkpoint to use for runtime control of one agent slot.",
    )
    parser.add_argument(
        "--learning-agent-id",
        default=None,
        help="Agent id replaced at runtime by the PPO policy when --checkpoint is provided.",
    )
    parser.add_argument(
        "--learning-agent-starting-inventory",
        type=float,
        default=0.0,
        help="Starting inventory for the runtime-replaced PPO slot only.",
    )
    return parser.parse_args()
