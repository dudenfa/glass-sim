"""Command line interface.

Examples::

    python -m marl_trading.futures scenarios
    python -m marl_trading.futures run --scenario default --seed 1 --duration 4h --out artifacts/futures/default_s1
    python -m marl_trading.futures serve --scenario default --seed 1 --open
    python -m marl_trading.futures compare "scenario=default seed=1" "scenario=default seed=1 transparency=anonymous"
    python -m marl_trading.futures compare "scenario=default" "scenario=anonymous" --seeds 1-10
    python -m marl_trading.futures compare artifacts/futures/run_a artifacts/futures/run_b --agents

    python -m marl_trading.futures train --name ppo_full_v1 --timesteps 500000
    python -m marl_trading.futures evaluate checkpoints/futures/ppo_full_v1 --seeds 31-40
    python -m marl_trading.futures serve --rl checkpoints/futures/ppo_full_v1

(``perpsim`` is the same command once the package is installed.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from typing import List, Optional

from . import scenarios
from .config import parse_duration
from .market import PerpetualMarket
from .types import Transparency


def _config_from_args(args: argparse.Namespace):
    overrides = {"seed": args.seed}
    if args.duration:
        overrides["duration"] = parse_duration(args.duration)
    config = scenarios.get(args.scenario, **overrides)
    if args.transparency:
        config = replace(config, transparency=Transparency(args.transparency))
    return config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", default="default", choices=scenarios.names())
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", default=None, help="e.g. 30m, 4h, 1d (default: scenario's own)")
    parser.add_argument("--transparency", choices=[t.value for t in Transparency], default=None)


def cmd_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    started = time.perf_counter()
    result = PerpetualMarket(config, keep_order_events=args.keep_orders).run()
    elapsed = time.perf_counter() - started
    result.print_report(agents=args.agents)
    print(f"\nsimulated {config.duration / 3600:.2f}h, {result.events_processed:,} events in {elapsed:.1f}s")
    if args.out:
        path = result.save(args.out)
        print(f"results written to {path} (report.txt has the full per-agent table)")
    if args.json:
        print(json.dumps(result.summary(), indent=2, default=str))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .live import FuturesViewer, LiveSession

    duration = parse_duration(args.duration) if args.duration else None
    checkpoints = [str(path if str(path).endswith(".zip") else f"{path}/model.zip") for path in args.rl]
    session = LiveSession(scenario=args.scenario, seed=args.seed, transparency=args.transparency, speed=args.speed, duration=duration, rl_checkpoints=checkpoints)
    session.playing = not args.paused
    viewer = FuturesViewer(session, host=args.host, port=args.port)
    print(f"Futures market viewer on {viewer.url}  (Ctrl+C to stop)")
    try:
        viewer.serve_forever(open_browser=args.open)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_panels, compare_runs, load_or_run, parse_seeds

    progress = (lambda message: print(f"  ran {message}", file=sys.stderr)) if args.verbose else None
    if args.seeds:
        text = compare_panels(args.left, args.right, parse_seeds(args.seeds), progress=progress)
    else:
        text = compare_runs(load_or_run(args.left), load_or_run(args.right), show_agents=args.agents)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    return 0


def cmd_scenarios(_args: argparse.Namespace) -> int:
    for name in scenarios.names():
        config = scenarios.get(name)
        agents = sum(spec.count for spec in config.agents)
        doc = (scenarios.SCENARIOS[name].__doc__ or "").strip().splitlines()
        print(f"{name:15s} agents={agents:4d}  transparency={config.transparency.value:10s} duration={config.duration / 3600:.1f}h  {doc[0] if doc else ''}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m marl_trading.futures", description="Transparent perpetual-futures market simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a simulation and print a report")
    _add_common(run)
    run.add_argument("--out", default=None, help="directory for summary/agents JSON and CSV time series")
    run.add_argument("--json", action="store_true", help="also print the summary as JSON")
    run.add_argument("--keep-orders", action="store_true", help="store every order event (larger output)")
    run.add_argument("--agents", action="store_true", help="also print the per-agent table")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="start the live browser viewer")
    _add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--speed", type=float, default=10.0, help="simulated seconds per real second")
    serve.add_argument("--paused", action="store_true")
    serve.add_argument("--open", action="store_true", help="open the browser")
    serve.add_argument("--rl", action="append", default=[], metavar="RUN_DIR_OR_ZIP", help="add a trained RL agent (repeatable); shown as rl_01, rl_02, ...")
    serve.set_defaults(func=cmd_serve)

    compare = sub.add_parser("compare", help="compare two runs, or two setups over a panel of seeds")
    compare.add_argument("left", help='run spec like "scenario=default seed=1 transparency=full duration=1h", or a saved run directory')
    compare.add_argument("right", help="run spec or saved run directory")
    compare.add_argument("--seeds", default=None, help='paired panel over seeds, e.g. "1-10" (run specs only; their seed is ignored)')
    compare.add_argument("--agents", action="store_true", help="list every agent instead of the largest changes")
    compare.add_argument("--out", default=None, help="also write the comparison to this file")
    compare.add_argument("--verbose", action="store_true", help="print progress while running panels")
    compare.set_defaults(func=cmd_compare)

    train = sub.add_parser("train", help="train an RL agent (see python -m marl_trading.futures.rl.train --help)", add_help=False)
    train.set_defaults(func=None, passthrough="train")
    evaluate = sub.add_parser("evaluate", help="evaluate a trained RL agent (see python -m marl_trading.futures.rl.evaluate --help)", add_help=False)
    evaluate.set_defaults(func=None, passthrough="evaluate")

    listing = sub.add_parser("scenarios", help="list available scenarios")
    listing.set_defaults(func=cmd_scenarios)

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("train", "evaluate"):
        if argv[0] == "train":
            from .rl.train import main as train_main

            return train_main(argv[1:])
        from .rl.evaluate import main as evaluate_main

        return evaluate_main(argv[1:])
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
