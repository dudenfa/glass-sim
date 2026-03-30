from __future__ import annotations

import time

from marl_trading.live.server import LiveServerConfig, parse_args, serve_market_view


def main() -> int:
    args = parse_args()
    server = serve_market_view(
        LiveServerConfig(
            host=str(args.host),
            port=int(args.port),
            seed=int(args.seed),
            horizon=int(args.horizon),
            speed=float(args.speed),
            autoplay=not bool(args.paused),
            open_browser=bool(args.open_browser),
        ),
    )
    print(f"Live market viewer running at {server.url}")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
