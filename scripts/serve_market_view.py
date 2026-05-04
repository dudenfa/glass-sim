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
from marl_trading.rl.scenario import prepare_learning_agent_config


def main() -> None:
    args = parse_args()
    simulation_config = build_preset_config(args.preset)
    if args.seed is not None:
        simulation_config = replace(simulation_config, seed=int(args.seed))
    learning_agent_id = None
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
        learning_agent_id = str(args.learning_agent_id or "trend_01")
        simulation_config = prepare_learning_agent_config(
            simulation_config,
            learning_agent_id=learning_agent_id,
            add_learning_agent=bool(args.add_learning_agent),
            learning_agent_template_id=None if args.learning_agent_template_id is None else str(args.learning_agent_template_id),
        )
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
            checkpoint_path=checkpoint_path,
            learning_agent_id=learning_agent_id,
            learning_agent_starting_inventory=float(args.learning_agent_starting_inventory),
        ),
    )
    print(f"Serving live market view at {server.url} using preset '{args.preset}'")
    if checkpoint_path is not None:
        print(f"Runtime PPO control enabled for '{learning_agent_id}' from {checkpoint_path}")
        print(
            "Runtime PPO mode: "
            + ("add participant" if bool(args.add_learning_agent) else "replace scripted slot")
        )
        print(f"Runtime PPO starting inventory override: {float(args.learning_agent_starting_inventory):.2f}")
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
