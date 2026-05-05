from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.analysis import (
    build_agent_health_metrics,
    build_portfolio_health_rows,
    format_market_health_summary,
    format_portfolio_health_breakdown,
    summarize_market_health,
)
from marl_trading.configs import available_preset_names, get_preset
from marl_trading.market import SyntheticMarketSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a named market preset and print a compact health summary.")
    parser.add_argument(
        "--preset",
        choices=available_preset_names(),
        default="baseline",
        help="Named preset from marl_trading.configs.presets.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the preset seed.")
    parser.add_argument("--horizon", type=int, default=None, help="Override the preset event horizon.")
    parser.add_argument("--output", type=Path, default=None, help="Optional file path for the report output.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of compact text.")
    parser.add_argument(
        "--portfolio-breakdown",
        action="store_true",
        help="Include per-agent end-state portfolio metrics in the report.",
    )
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    return parser.parse_args()


def _preset_overview() -> str:
    lines = ["Available presets:"]
    for name in available_preset_names():
        preset = get_preset(name)
        lines.append(f"- {preset.name}: {preset.description}")
    return "\n".join(lines)


def build_market_health_report(
    preset_name: str,
    *,
    seed: int | None = None,
    horizon: int | None = None,
    portfolio_breakdown: bool = False,
) -> dict[str, Any]:
    preset = get_preset(preset_name)
    config = preset.build()
    if seed is not None:
        config = replace(config, seed=int(seed))

    effective_horizon = int(horizon if horizon is not None else config.market.event_horizon)
    simulator = SyntheticMarketSimulator(config, horizon=effective_horizon)
    result = simulator.run(horizon=effective_horizon)
    summary = summarize_market_health(result)
    report = format_market_health_summary(
        summary,
        preset_name=preset.name,
        seed=config.seed,
        horizon=effective_horizon,
    )
    portfolio_rows = None
    if portfolio_breakdown:
        final_mark_price = float(
            dict(getattr(result, "summary", {})).get("final_mark_price")
            or summary.final_midpoint
            or summary.final_fundamental
            or config.market.starting_mid_price
        )
        agent_metrics = build_agent_health_metrics(
            list(result.event_log.events),
            config.agents,
            starting_midpoint=float(config.market.starting_mid_price),
            final_mark_price=final_mark_price,
            open_orders_by_agent={agent_id: len(queue) for agent_id, queue in simulator.open_orders.items()},
        )
        portfolio_rows = build_portfolio_health_rows(
            result.final_portfolios,
            config.agents,
            starting_midpoint=float(config.market.starting_mid_price),
            agent_metrics=agent_metrics,
        )
        report = "\n\n".join(
            [
                report,
                format_portfolio_health_breakdown(portfolio_rows),
            ]
        )
    return {
        "preset": preset.name,
        "description": preset.description,
        "seed": config.seed,
        "horizon": effective_horizon,
        "summary": summary,
        "report": report,
        "portfolio_breakdown": portfolio_rows,
    }


def _serialize_report(payload: dict[str, Any]) -> dict[str, Any]:
    portfolio_breakdown = [row.to_dict() for row in payload["portfolio_breakdown"]] if payload.get("portfolio_breakdown") else None
    return {
        "preset": payload["preset"],
        "description": payload["description"],
        "seed": payload["seed"],
        "horizon": payload["horizon"],
        "report": payload["report"],
        "summary": payload["summary"].to_dict(),
        "portfolio_breakdown": portfolio_breakdown,
        "agents": portfolio_breakdown,
    }


def main() -> None:
    args = parse_args()
    if args.list_presets:
        text = _preset_overview()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return

    payload = build_market_health_report(
        args.preset,
        seed=args.seed,
        horizon=args.horizon,
        portfolio_breakdown=args.portfolio_breakdown,
    )
    output = json.dumps(_serialize_report(payload), indent=2) if args.json else payload["report"]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + ("\n" if not output.endswith("\n") else ""), encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds \n", file=sys.stderr)
