from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_agent_ranking_panel import MARKET_COLUMNS, build_ranking_payload


AGENT_METRICS = (
    ("medianRealizedPnl", "Median realized PnL", 2),
    ("medianUnrealizedPnl", "Median unrealized PnL", 2),
    ("medianTotalPnl", "Median total PnL", 2),
    ("medianPnlPct", "Median PnL %", 2),
    ("meanDrawdown", "Mean drawdown", 2),
    ("meanMaxAbsInventory", "Mean max |inventory|", 2),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two fixed-16 agent-ranking panels.")
    parser.add_argument("baseline_dir", type=Path, help="Baseline ranking panel directory.")
    parser.add_argument("treatment_dir", type=Path, help="Treatment ranking panel directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output Markdown path.")
    parser.add_argument("--html-output", type=Path, default=None, help="Optional self-contained HTML output path.")
    parser.add_argument("--title", default="Agent-Ranking Panel Comparison", help="Reader-facing comparison title.")
    parser.add_argument("--baseline-label", default=None, help="Reader-facing baseline panel label.")
    parser.add_argument("--treatment-label", default=None, help="Reader-facing treatment panel label.")
    return parser.parse_args(argv)


def build_comparison_payload(baseline_dir: Path, treatment_dir: Path) -> dict[str, Any]:
    baseline = build_ranking_payload(baseline_dir)
    treatment = build_ranking_payload(treatment_dir)
    baseline_seeds = set(baseline["manifest"].get("seeds", []))
    treatment_seeds = set(treatment["manifest"].get("seeds", []))
    if baseline_seeds and treatment_seeds and baseline_seeds != treatment_seeds:
        raise ValueError("Panels must use the same seed set for paired comparison.")

    treatment_market = {row["structureId"]: row for row in treatment["marketSummary"]}
    baseline_market = {row["structureId"]: row for row in baseline["marketSummary"]}
    structures = [structure_id for structure_id in baseline["structureOrder"] if structure_id in treatment_market]
    rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    for structure_id in structures:
        baseline_agents = {row["agentId"]: row for row in baseline["structureLeaderboards"].get(structure_id, [])}
        treatment_agents = {row["agentId"]: row for row in treatment["structureLeaderboards"].get(structure_id, [])}
        for agent_id in sorted(set(baseline_agents) & set(treatment_agents)):
            before, after = baseline_agents[agent_id], treatment_agents[agent_id]
            rows.append(
                {
                    "structureId": structure_id,
                    "structureLabel": baseline["labels"][structure_id],
                    "agentId": agent_id,
                    "agentType": before["agentType"],
                    "before": before,
                    "after": after,
                    "delta": {key: float(after[key]) - float(before[key]) for key, _, _ in AGENT_METRICS},
                }
            )
        before_market, after_market = baseline_market[structure_id], treatment_market[structure_id]
        market_rows.append(
            {
                "structureId": structure_id,
                "structureLabel": baseline["labels"][structure_id],
                "before": before_market,
                "after": after_market,
                "delta": {key: float(after_market[key]) - float(before_market[key]) for key, _, _ in MARKET_COLUMNS},
            }
        )
    return {
        "seedCount": baseline["seedCount"],
        "structures": structures,
        "agentRows": rows,
        "marketRows": market_rows,
        "treatmentExcludedAgents": treatment["manifest"].get("excluded_agent_ids", []),
    }


def _format(value: float, digits: int = 2) -> str:
    return f"{value:+,.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [f"# {payload['title']}", ""]
    lines.append(f"- Paired seed count: `{payload['seedCount']}`")
    lines.append(f"- Baseline: `{payload['baselineLabel']}`")
    lines.append(f"- Treatment: `{payload['treatmentLabel']}`")
    lines.append("- Values below are treatment-panel medians minus baseline-panel medians.")
    lines.append("")
    for structure_id in payload["structures"]:
        rows = [row for row in payload["agentRows"] if row["structureId"] == structure_id]
        lines.extend([f"## {rows[0]['structureLabel'] if rows else structure_id}", ""])
        lines.append("| Agent | Realized PnL | Unrealized PnL | Total PnL | PnL % | Drawdown | Max |inventory| |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            delta = row["delta"]
            lines.append(
                "| {agent} | {realized} | {unrealized} | {total} | {pct} | {drawdown} | {inventory} |".format(
                    agent=row["agentId"],
                    realized=_format(delta["medianRealizedPnl"]),
                    unrealized=_format(delta["medianUnrealizedPnl"]),
                    total=_format(delta["medianTotalPnl"]),
                    pct=_format(delta["medianPnlPct"]),
                    drawdown=_format(delta["meanDrawdown"]),
                    inventory=_format(delta["meanMaxAbsInventory"]),
                )
            )
        market = next(row for row in payload["marketRows"] if row["structureId"] == structure_id)
        lines.extend(["", "Market-quality delta:", ""])
        lines.append("| Trades | Spread availability | Mean spread | Volatility (bps) | Top-book liquidity | Final price |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        lines.append(
            "| {trades} | {availability} | {spread} | {volatility} | {liquidity} | {price} |".format(
                trades=_format(market["delta"]["trade_count"], 0),
                availability=_format(market["delta"]["spread_availability_ratio"], 3),
                spread=_format(market["delta"]["mean_spread"], 4),
                volatility=_format(market["delta"]["midpoint_return_volatility_bps"], 2),
                liquidity=_format(market["delta"]["mean_top_of_book_liquidity"], 2),
                price=_format(market["delta"]["final_midpoint"], 2),
            )
        )
        lines.append("")
    return "\n".join(lines)


def _html(payload: dict[str, Any]) -> str:
    body = html.escape(_markdown(payload))
    title = html.escape(str(payload["title"]))
    return f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{title}</title><style>body{{margin:0;background:#f4f6f7;color:#17232d;font-family:Arial,Helvetica,sans-serif}}main{{max-width:1100px;margin:auto;padding:32px 16px 56px}}h1{{font:500 2.5rem Georgia,serif}}pre{{margin-top:24px;padding:18px;background:#fff;border:1px solid #d4dde2;border-radius:6px;overflow:auto;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre}}</style><main><h1>{title}</h1><pre>{body}</pre></main></html>"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_comparison_payload(args.baseline_dir.resolve(), args.treatment_dir.resolve())
    payload.update(
        {
            "title": args.title,
            "baselineLabel": args.baseline_label or args.baseline_dir.name,
            "treatmentLabel": args.treatment_label or args.treatment_dir.name,
        }
    )
    output = args.output.resolve() if args.output else args.treatment_dir.resolve() / "with_vs_without_informed.md"
    output.write_text(_markdown(payload), encoding="utf-8")
    if args.html_output is not None:
        args.html_output.resolve().write_text(_html(payload), encoding="utf-8")
    print(f"Wrote comparison to {output}")


if __name__ == "__main__":
    main()
