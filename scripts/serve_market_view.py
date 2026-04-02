from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.configs import build_preset_config
from marl_trading.live.server import LiveServerConfig, parse_args, serve_market_view


def main() -> None:
    args = parse_args()
    simulation_config = build_preset_config(args.preset)
    if args.seed is not None:
      simulation_config = replace(simulation_config, seed=int(args.seed))
    server = serve_market_view(
        LiveServerConfig(
            host=args.host,
            port=args.port,
            seed=args.seed,
            horizon=args.horizon,
            speed=args.speed,
            autoplay=not args.paused,
            open_browser=args.open_browser,
            preset=args.preset,
            simulation_config=simulation_config,
        ),
    )
    print(f"Serving live market view at {server.url} using preset '{args.preset}'")
    print("Use Ctrl+C to stop.")
    try:
        while True:
            server.session.state()
            import time

            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
