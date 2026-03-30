from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.analysis import EventLog, plot_market_replay, summarize_event_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and plot a synthetic market event log.")
    parser.add_argument("event_log", help="Path to a JSON or JSONL event log.")
    parser.add_argument("--output", default="market_replay.png", help="Path to the main replay figure.")
    parser.add_argument(
        "--depth-output",
        default=None,
        help="Optional path for the order book depth heatmap. Defaults to <output>_depth.png when snapshots exist.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the event summary without generating figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_log = EventLog.load(args.event_log)
    summary = summarize_event_log(event_log)
    print(json.dumps(summary, indent=2))

    if args.summary_only:
        return

    output_path = Path(args.output)
    result = plot_market_replay(event_log, output_path=output_path, depth_output_path=args.depth_output)
    print(f"Saved replay figure to {output_path}")
    if result != output_path:
        print(f"Saved depth figure to {result}")


if __name__ == "__main__":
    main()
