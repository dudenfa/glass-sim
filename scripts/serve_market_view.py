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
from marl_trading.rl.scenario import (
    build_frozen_agent_specs,
    exclude_agents_from_config,
    prepare_frozen_agent_configs,
    prepare_learning_agent_config,
)


def main() -> None:
    args = parse_args()
    try:
        frozen_agent_specs = build_frozen_agent_specs(
            frozen_agents=tuple(args.frozen_agent or ()),
            frozen_agent_checkpoint_path=args.frozen_agent_checkpoint,
            frozen_agent_id=args.frozen_agent_id,
            add_frozen_agent=bool(args.add_frozen_agent),
            frozen_agent_template_id=args.frozen_agent_template_id,
            frozen_agent_starting_inventory=args.frozen_agent_starting_inventory,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for frozen_agent in frozen_agent_specs:
        if frozen_agent.agent_id == str(args.learning_agent_id or "").strip():
            raise SystemExit("frozen-agent-id must be different from learning-agent-id.")
    simulation_config = build_preset_config(args.preset)
    if args.seed is not None:
        simulation_config = replace(simulation_config, seed=int(args.seed))
    simulation_config = exclude_agents_from_config(
        simulation_config,
        excluded_agent_ids=tuple(args.exclude_agent or ()),
        preserve_excluded_turn_slots=bool(args.preserve_excluded_turn_slots),
    )
    learning_agent_id = None
    checkpoint_path = None
    simulation_config = prepare_frozen_agent_configs(simulation_config, frozen_agents=frozen_agent_specs)
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
            frozen_agents=frozen_agent_specs,
            frozen_agent_checkpoint_path=None,
            frozen_agent_id=None,
            frozen_agent_starting_inventory=None,
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
    for frozen_agent in frozen_agent_specs:
        print(f"Frozen runtime PPO opponent enabled for '{frozen_agent.agent_id}' from {frozen_agent.checkpoint_path}")
        print(
            "Frozen PPO mode: "
            + ("add participant" if bool(frozen_agent.add_agent) else "replace scripted slot")
        )
        if frozen_agent.starting_inventory is not None:
            print(f"Frozen PPO starting inventory override for '{frozen_agent.agent_id}': {float(frozen_agent.starting_inventory):.2f}")
    if args.exclude_agent:
        print("Excluded base-preset participant(s): " + ", ".join(args.exclude_agent))
    if args.preserve_excluded_turn_slots:
        print("Excluded participant turns are preserved as no-ops.")
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
