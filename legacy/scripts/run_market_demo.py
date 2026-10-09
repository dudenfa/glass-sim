from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.analysis import plot_market_replay, summarize_event_log
from marl_trading.configs import available_preset_names, build_preset_config
from marl_trading.market import MarketRunResult, SyntheticMarketSimulator, plot_market_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first scripted synthetic market demo.")
    parser.add_argument(
        "--preset",
        choices=available_preset_names(),
        default="baseline",
        help="Named preset from marl_trading.configs.presets.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the preset seed.")
    parser.add_argument("--horizon", type=int, default=240, help="Number of exchange events to simulate.")
    parser.add_argument(
        "--output-dir",
        default="market_demo_output",
        help="Directory where the event log and figures will be written.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write the event log and summary JSON, skipping plots.",
    )
    return parser.parse_args()


def run_demo(*, preset: str = "baseline", seed: int | None = None, horizon: int = 240) -> MarketRunResult:
    config = build_preset_config(preset)
    if seed is not None:
        config = replace(config, seed=int(seed))
    simulator = SyntheticMarketSimulator(config, horizon=horizon)
    return simulator.run(horizon=horizon)


def _write_outputs(result, output_dir: Path, summary_only: bool) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / "market_events.jsonl"
    summary_path = output_dir / "market_summary.json"
    result.event_log.save(event_log_path)
    summary = summarize_event_log(result.event_log)
    summary.update(result.summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paths = {
        "event_log": str(event_log_path),
        "summary": str(summary_path),
    }

    if summary_only:
        return paths

    world_path = output_dir / "market_world.png"
    replay_path = output_dir / "market_replay.png"
    depth_path = output_dir / "market_replay_depth.png"
    try:
        plot_market_world(result, world_path)
    except RuntimeError as exc:
        paths["world_plot_error"] = str(exc)
    else:
        paths["world_plot"] = str(world_path)

    try:
        plot_market_replay(result.event_log, output_path=replay_path, depth_output_path=depth_path)
    except RuntimeError as exc:
        paths["replay_plot_error"] = str(exc)
    else:
        paths["replay_plot"] = str(replay_path)
        if depth_path.exists():
            paths["replay_depth_plot"] = str(depth_path)

    return paths


def main() -> None:
    args = parse_args()
    result = run_demo(preset=args.preset, seed=args.seed, horizon=args.horizon)
    output_dir = Path(args.output_dir)
    paths = _write_outputs(result, output_dir, summary_only=args.summary_only)
    print(json.dumps({"summary": result.summary, "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
