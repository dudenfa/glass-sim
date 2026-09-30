"""HTTP server for the live futures-market viewer (standard library only).

Run it with::

    python -m marl_trading.futures serve --scenario default --seed 1

then open http://127.0.0.1:8766. Endpoints:

- ``GET /api/state?bars_since=N``: market state, with bars from index N on
- ``GET /api/account?id=AGENT``: one account's detail (orders, fills, equity path)
- ``POST /api/control``: ``{"action": "play"|"pause"|"step"|"finish"|"speed"|"reset", ...}``
"""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .session import LiveSession

STATIC_ROOT = Path(__file__).resolve().parent / "static"


def _json_default(value: Any) -> Any:
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return str(value)


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class _Handler(BaseHTTPRequestHandler):
    server_version = "PerpViewer/1.0"

    @property
    def session(self) -> LiveSession:
        return self.server.session  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_clean(payload), default=_json_default, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, name: str) -> None:
        path = (STATIC_ROOT / name).resolve()
        if STATIC_ROOT not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif parsed.path == "/api/state":
            self._send_json(self.session.state(bars_since=int(query.get("bars_since", ["0"])[0])))
        elif parsed.path == "/api/account":
            try:
                self._send_json(self.session.account_detail(query.get("id", [""])[0]))
            except KeyError:
                self._send_json({"error": "unknown account"}, HTTPStatus.NOT_FOUND)
        elif parsed.path.startswith("/static/"):
            self._serve_static(parsed.path[len("/static/"):])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            self.session.control(payload)
        except (ValueError, KeyError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class FuturesViewer:
    """Owns a ``LiveSession`` and serves it over HTTP."""

    def __init__(self, session: LiveSession, host: str = "127.0.0.1", port: int = 8766) -> None:
        self.session = session
        self.httpd = _Server((host, port), _Handler)
        self.httpd.session = session  # type: ignore[attr-defined]
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self, open_browser: bool = False) -> None:
        self.session.start()
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url)

    def serve_forever(self, open_browser: bool = False) -> None:
        self.session.start()
        if open_browser:
            webbrowser.open(self.url)
        try:
            self.httpd.serve_forever()
        finally:
            self.stop()

    def stop(self) -> None:
        self.session.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
