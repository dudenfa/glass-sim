from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT_PATTERN = re.compile(r"(?P<structure>.+)_seed_(?P<seed>\d+)\.json$")
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
PALETTE = {
    "rl": "#7A828F",
    "marl": "#5477C4",
    "marl10": "#CC6F47",
    "scripted": "#B8A037",
}
STRUCTURE_SHORT = {
    "rl_trio_7": "Sequential RL / 7-agent",
    "marl7_in_7": "MARL trained in 7-agent / deployed in 7-agent",
    "marl10_in_7": "MARL trained in 10-agent / deployed in 7-agent",
    "marl7_in_10": "MARL trained in 7-agent / deployed in 10-agent",
    "marl10_in_10": "MARL trained in 10-agent / deployed in 10-agent",
}
GROUP_LABEL = {
    "rl": "Sequential RL",
    "marl": "MARL trained in 7-agent market",
    "marl10": "MARL trained in 10-agent market",
}
STRUCTURE_CODE = {
    "rl_trio_7": "RL / 7",
    "marl7_in_7": "MARL7 / 7",
    "marl10_in_7": "MARL10 / 7",
    "marl7_in_10": "MARL7 / 10",
    "marl10_in_10": "MARL10 / 10",
}
GROUP_CODE = {"rl": "RL", "marl": "MARL7", "marl10": "MARL10"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Stage A MARL technical report from evaluation JSON reports.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _median(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [_number(row.get(key)) for row in rows]
    return float(median(values)) if values else 0.0


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [_number(row.get(key)) for row in rows]
    return float(mean(values)) if values else 0.0


def _agent_group(agent_id: str) -> str:
    if agent_id in {"rl_01", "rl_02", "rl_03"}:
        return "rl"
    if agent_id.startswith("marl_vs_rl_"):
        return "marl10"
    if agent_id.startswith("marl_"):
        return "marl"
    return "scripted"


def _load_reports(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*_seed_*.json")):
        match = REPORT_PATTERN.match(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata", {}))
        reports.append(
            {
                "path": path,
                "structure_id": str(metadata.get("marl_stage_a_structure_id", match.group("structure"))),
                "structure_label": str(metadata.get("marl_stage_a_structure_label", match.group("structure"))),
                "seed": int(payload.get("seed", match.group("seed"))),
                "summary": dict(payload.get("summary", {})),
                "portfolio": list(payload.get("portfolio_breakdown", []) or []),
                "activity": list(payload.get("agent_activity", []) or []),
            }
        )
    if not reports:
        raise ValueError("No Stage A report files found.")
    return manifest, reports


def build_analysis(input_dir: Path) -> dict[str, Any]:
    manifest, reports = _load_reports(input_dir)
    structure_order = [str(item["structure_id"]) for item in manifest.get("structures", [])]
    labels = {str(item["structure_id"]): str(item["label"]) for item in manifest.get("structures", [])}
    rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []

    for report in reports:
        activity_by_id = {str(item.get("agent_id")): item for item in report["activity"] if isinstance(item, dict)}
        market_rows.append(
            {
                "structure_id": report["structure_id"],
                "structure_label": report["structure_label"],
                "seed": report["seed"],
                "trade_count": _number(report["summary"].get("trade_count")),
                "spread_availability": _number(report["summary"].get("spread_availability_ratio")),
                "mean_spread": _number(report["summary"].get("mean_spread")),
                "volatility_bps": _number(report["summary"].get("midpoint_return_volatility_bps")),
                "top_book_liquidity": _number(report["summary"].get("mean_top_of_book_liquidity")),
                "final_price": _number(report["summary"].get("final_midpoint")),
                "final_fundamental": _number(report["summary"].get("final_fundamental")),
            }
        )
        for portfolio in report["portfolio"]:
            if not isinstance(portfolio, dict):
                continue
            agent_id = str(portfolio.get("agent_id"))
            activity = dict(activity_by_id.get(agent_id, {}))
            starting_equity = _number(portfolio.get("starting_equity"))
            total_pnl = _number(portfolio.get("total_pnl", portfolio.get("equity_delta")))
            rows.append(
                {
                    "structure_id": report["structure_id"],
                    "structure_label": report["structure_label"],
                    "seed": report["seed"],
                    "agent_id": agent_id,
                    "agent_type": str(portfolio.get("agent_type", "unknown")),
                    "group": _agent_group(agent_id),
                    "starting_equity": starting_equity,
                    "realized_pnl": _number(portfolio.get("realized_pnl")),
                    "unrealized_pnl": _number(portfolio.get("unrealized_pnl")),
                    "total_pnl": total_pnl,
                    "pnl_pct": 100.0 * total_pnl / starting_equity if starting_equity else 0.0,
                    "drawdown": _number(portfolio.get("max_equity_drawdown")),
                    "max_abs_inventory": _number(portfolio.get("max_abs_inventory")),
                    "ending_inventory": _number(portfolio.get("ending_inventory")),
                    "submitted_orders": _number(activity.get("submitted_orders")),
                    "market_buy_orders": _number(activity.get("market_buy_orders")),
                    "market_sell_orders": _number(activity.get("market_sell_orders")),
                    "limit_buy_orders": _number(activity.get("limit_buy_orders")),
                    "limit_sell_orders": _number(activity.get("limit_sell_orders")),
                    "executed_buy_quantity": _number(activity.get("executed_buy_quantity")),
                    "executed_sell_quantity": _number(activity.get("executed_sell_quantity")),
                    "executed_quantity": _number(activity.get("executed_quantity")),
                }
            )

    run_groups: list[dict[str, Any]] = []
    by_run_group: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run_group[(row["structure_id"], row["seed"], row["group"])].append(row)
    for (structure_id, seed, group), members in by_run_group.items():
        starting_equity = sum(_number(row["starting_equity"]) for row in members)
        total_pnl = sum(_number(row["total_pnl"]) for row in members)
        run_groups.append(
            {
                "structure_id": structure_id,
                "structure_label": labels.get(structure_id, structure_id),
                "seed": seed,
                "group": group,
                "agent_count": len(members),
                "starting_equity": starting_equity,
                "realized_pnl": sum(_number(row["realized_pnl"]) for row in members),
                "unrealized_pnl": sum(_number(row["unrealized_pnl"]) for row in members),
                "total_pnl": total_pnl,
                "coalition_return_pct": 100.0 * total_pnl / starting_equity if starting_equity else 0.0,
            }
        )

    individual_summary: list[dict[str, Any]] = []
    by_individual: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["group"] != "scripted":
            by_individual[(row["structure_id"], row["agent_id"])].append(row)
    for (structure_id, agent_id), values in by_individual.items():
        individual_summary.append(
            {
                "structure_id": structure_id,
                "structure_label": labels.get(structure_id, structure_id),
                "agent_id": agent_id,
                "group": values[0]["group"],
                "median_realized_pnl": _median(values, "realized_pnl"),
                "median_unrealized_pnl": _median(values, "unrealized_pnl"),
                "median_total_pnl": _median(values, "total_pnl"),
                "median_pnl_pct": _median(values, "pnl_pct"),
                "mean_pnl_pct": _mean(values, "pnl_pct"),
                "median_drawdown": _median(values, "drawdown"),
                "mean_max_abs_inventory": _mean(values, "max_abs_inventory"),
                "median_submitted_orders": _median(values, "submitted_orders"),
                "median_executed_quantity": _median(values, "executed_quantity"),
                "median_net_executed_flow": _median(values, "executed_buy_quantity") - _median(values, "executed_sell_quantity"),
                "runs": len(values),
            }
        )
    individual_summary.sort(key=lambda row: (structure_order.index(row["structure_id"]), -row["median_pnl_pct"], row["agent_id"]))

    coalition_summary: list[dict[str, Any]] = []
    by_coalition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_groups:
        if row["group"] != "scripted":
            by_coalition[(row["structure_id"], row["group"])].append(row)
    for (structure_id, group), values in by_coalition.items():
        coalition_summary.append(
            {
                "structure_id": structure_id,
                "structure_label": labels.get(structure_id, structure_id),
                "group": group,
                "agent_count": int(values[0]["agent_count"]),
                "median_realized_pnl": _median(values, "realized_pnl"),
                "median_unrealized_pnl": _median(values, "unrealized_pnl"),
                "median_total_pnl": _median(values, "total_pnl"),
                "median_return_pct": _median(values, "coalition_return_pct"),
                "mean_return_pct": _mean(values, "coalition_return_pct"),
                "runs": len(values),
            }
        )
    coalition_summary.sort(key=lambda row: (structure_order.index(row["structure_id"]), row["group"]))

    behavior_summary = [
        row for row in individual_summary if row["group"] in {"marl", "marl10"}
    ]

    market_summary: list[dict[str, Any]] = []
    for structure_id in structure_order:
        values = [row for row in market_rows if row["structure_id"] == structure_id]
        if not values:
            continue
        market_summary.append(
            {
                "structure_id": structure_id,
                "structure_label": labels.get(structure_id, structure_id),
                "median_trades": _median(values, "trade_count"),
                "median_spread_availability": _median(values, "spread_availability"),
                "median_mean_spread": _median(values, "mean_spread"),
                "median_volatility_bps": _median(values, "volatility_bps"),
                "median_top_book_liquidity": _median(values, "top_book_liquidity"),
                "median_final_dislocation_bps": _median(
                    [
                        {"value": 10_000.0 * (row["final_price"] - row["final_fundamental"]) / row["final_fundamental"]}
                        for row in values if row["final_fundamental"]
                    ],
                    "value",
                ),
            }
        )

    paired_comparisons: list[dict[str, Any]] = []
    comparison_specs = (
        ("RL trio vs MARL7 in 7-agent market", "rl_trio_7", "rl", "marl7_in_7", "marl"),
        ("RL trio vs MARL10 in 7-agent market", "rl_trio_7", "rl", "marl10_in_7", "marl10"),
        ("MARL7 vs RL inside 10-agent market", "marl7_in_10", "rl", "marl7_in_10", "marl"),
        ("MARL10 vs RL inside 10-agent market", "marl10_in_10", "rl", "marl10_in_10", "marl10"),
    )
    return_by_key = {(row["structure_id"], row["seed"], row["group"]): row["coalition_return_pct"] for row in run_groups}
    for label, left_structure, left_group, right_structure, right_group in comparison_specs:
        differences: list[float] = []
        for seed in sorted({row["seed"] for row in run_groups}):
            left = return_by_key.get((left_structure, seed, left_group))
            right = return_by_key.get((right_structure, seed, right_group))
            if left is not None and right is not None:
                differences.append(float(right) - float(left))
        paired_comparisons.append(
            {
                "label": label,
                "difference_definition": "right minus left coalition return percentage points",
                "median_difference_pp": float(median(differences)) if differences else 0.0,
                "mean_difference_pp": float(mean(differences)) if differences else 0.0,
                "right_wins": sum(value > 0 for value in differences),
                "runs": len(differences),
            }
        )

    return {
        "manifest": manifest,
        "structure_order": structure_order,
        "structure_short": STRUCTURE_SHORT,
        "reports": len(reports),
        "seeds": sorted({report["seed"] for report in reports}),
        "individual_rows": rows,
        "market_rows": market_rows,
        "run_groups": run_groups,
        "individual_summary": individual_summary,
        "coalition_summary": coalition_summary,
        "behavior_summary": behavior_summary,
        "market_summary": market_summary,
        "paired_comparisons": paired_comparisons,
    }


def use_chart_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
            "grid.color": TOKENS["grid"], "grid.linewidth": 0.8,
            "font.family": "sans-serif", "patch.linewidth": 1.0,
        }
    )


def add_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    fig.subplots_adjust(top=0.75, left=0.24, right=0.96, bottom=0.15)
    left = ax.get_position().x0
    fig.text(left, 0.97, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def _save(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_charts(analysis: dict[str, Any], charts_dir: Path) -> list[dict[str, str]]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    use_chart_theme()
    charts: list[dict[str, str]] = []

    coalition_rows = analysis["coalition_summary"]
    labels = [f"{STRUCTURE_CODE[row['structure_id']]} | {GROUP_CODE[row['group']]}" for row in coalition_rows]
    values = [row["median_return_pct"] for row in coalition_rows]
    colors = [PALETTE[row["group"]] for row in coalition_rows]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    positions = list(range(len(labels)))
    ax.barh(positions, values, color=colors, edgecolor=TOKENS["ink"], linewidth=0.8)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set_xlabel("Median coalition return on combined starting equity (%)")
    for position, value in zip(positions, values):
        ax.text(value + (0.05 if value >= 0 else -0.05), position, f"{value:+.2f}%", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    add_header(fig, ax, "Coalition profitability by deployed population", "Median across held-out seeds 31-40. MARL7/MARL10 denote the market size used for training; full labels appear in the report table.")
    output = charts_dir / "coalition_returns.png"
    _save(fig, output)
    charts.append({"file": output.name, "title": "Coalition profitability"})

    learned = analysis["individual_summary"]
    labels = [f"{STRUCTURE_SHORT[row['structure_id']]} | {row['agent_id']}" for row in learned]
    values = [row["median_pnl_pct"] for row in learned]
    colors = [PALETTE[row["group"]] for row in learned]
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.36 * len(labels) + 1.8)))
    positions = list(range(len(labels)))
    ax.barh(positions, values, color=colors, edgecolor=TOKENS["ink"], linewidth=0.8)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.set_yticks(positions, labels, fontsize=7.5)
    ax.set_xlabel("Median individual return on starting equity (%)")
    add_header(fig, ax, "Individual learned-agent performance", "Each bar is one policy in its own deployed ecology; compare agents within a structure before comparing across agent counts.")
    output = charts_dir / "individual_returns.png"
    _save(fig, output)
    charts.append({"file": output.name, "title": "Individual policy returns"})

    behavior = analysis["behavior_summary"]
    labels = [f"{STRUCTURE_SHORT[row['structure_id']]} | {row['agent_id']}" for row in behavior]
    net_flow = [row["median_net_executed_flow"] for row in behavior]
    orders = [row["median_submitted_orders"] for row in behavior]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6.0, 0.36 * len(labels) + 1.8)), gridspec_kw={"wspace": 0.42})
    positions = list(range(len(labels)))
    flow_colors = [PALETTE["marl10"] if row["group"] == "marl10" else PALETTE["marl"] for row in behavior]
    axes[0].barh(positions, net_flow, color=flow_colors, edgecolor=TOKENS["ink"], linewidth=0.8)
    axes[0].axvline(0, color=TOKENS["ink"], linewidth=1.0)
    axes[0].set_yticks(positions, labels, fontsize=7.5)
    axes[0].set_xlabel("Median executed buys minus sells (shares)")
    axes[1].barh(positions, orders, color=flow_colors, edgecolor=TOKENS["ink"], linewidth=0.8)
    axes[1].set_yticks(positions, [])
    axes[1].set_xlabel("Median submitted orders")
    fig.text(0.125, 0.97, "MARL behavior: directional flow and activity", ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.125, 0.915, "Positive flow means net executed buying. This is descriptive evidence of role differentiation, not proof of cooperation.", ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    fig.subplots_adjust(top=0.78, left=0.31, right=0.97, bottom=0.12)
    output = charts_dir / "marl_behavior.png"
    _save(fig, output)
    charts.append({"file": output.name, "title": "MARL trading behavior"})

    market = analysis["market_summary"]
    labels = [STRUCTURE_SHORT[row["structure_id"]] for row in market]
    positions = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"wspace": 0.38})
    axes[0].bar(positions, [row["median_trades"] for row in market], color=PALETTE["marl"], edgecolor=TOKENS["ink"], linewidth=0.8)
    axes[0].set_xticks(positions, [str(index + 1) for index in positions])
    axes[0].set_xlabel("Structure (see table below)")
    axes[0].set_ylabel("Median trade count")
    axes[1].bar(positions, [row["median_spread_availability"] for row in market], color=PALETTE["marl10"], edgecolor=TOKENS["ink"], linewidth=0.8)
    axes[1].set_xticks(positions, [str(index + 1) for index in positions])
    axes[1].set_xlabel("Structure (see table below)")
    axes[1].set_ylabel("Median spread availability ratio")
    fig.text(0.125, 0.97, "Market activity and quoted-spread availability", ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.125, 0.915, "Structures are ordered: 1 RL trio / 7, 2 MARL7 / 7, 3 MARL10 / 7, 4 MARL7 / 10, 5 MARL10 / 10.", ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    fig.subplots_adjust(top=0.78, left=0.10, right=0.97, bottom=0.16)
    output = charts_dir / "market_quality.png"
    _save(fig, output)
    charts.append({"file": output.name, "title": "Market quality"})
    return charts


def _fmt(value: float, digits: int = 2, signed: bool = False) -> str:
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:,.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class=\"table-wrap\"><table><thead><tr>{header_html}</tr></thead><tbody>{body}</tbody></table></div>"


def render_html(analysis: dict[str, Any], output_dir: Path, charts: list[dict[str, str]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_by_label = {row["label"]: row for row in analysis["paired_comparisons"]}
    m7_vs_rl = comparison_by_label["RL trio vs MARL7 in 7-agent market"]
    m10_vs_rl = comparison_by_label["RL trio vs MARL10 in 7-agent market"]
    m7_crowd = comparison_by_label["MARL7 vs RL inside 10-agent market"]
    m10_crowd = comparison_by_label["MARL10 vs RL inside 10-agent market"]
    top_marl = max(analysis["behavior_summary"], key=lambda row: row["median_pnl_pct"])

    coalition_table = _table(
        ["Deployed population", "Cohort", "Median realized", "Median unrealized", "Median total PnL", "Median return"],
        [
            [
                html.escape(STRUCTURE_SHORT[row["structure_id"]]), html.escape(GROUP_LABEL[row["group"]]),
                _fmt(row["median_realized_pnl"], signed=True), _fmt(row["median_unrealized_pnl"], signed=True),
                _fmt(row["median_total_pnl"], signed=True), f"{_fmt(row['median_return_pct'], signed=True)}%",
            ]
            for row in analysis["coalition_summary"]
        ],
    )
    behavior_table = _table(
        ["Deployed population", "MARL policy", "Median return", "Realized PnL", "Net executed flow", "Submitted orders", "Mean max |inventory|"],
        [
            [
                html.escape(STRUCTURE_SHORT[row["structure_id"]]), html.escape(row["agent_id"]),
                f"{_fmt(row['median_pnl_pct'], signed=True)}%", _fmt(row["median_realized_pnl"], signed=True),
                _fmt(row["median_net_executed_flow"], signed=True), _fmt(row["median_submitted_orders"], digits=0),
                _fmt(row["mean_max_abs_inventory"]),
            ]
            for row in analysis["behavior_summary"]
        ],
    )
    market_table = _table(
        ["#", "Population", "Median trades", "Spread availability", "Mean spread", "Volatility (bps)", "Top-book liquidity", "Final price-fundamental (bps)"],
        [
            [
                str(index + 1), html.escape(STRUCTURE_SHORT[row["structure_id"]]), _fmt(row["median_trades"], digits=0),
                _fmt(row["median_spread_availability"], digits=3), _fmt(row["median_mean_spread"], digits=4),
                _fmt(row["median_volatility_bps"]), _fmt(row["median_top_book_liquidity"]),
                _fmt(row["median_final_dislocation_bps"], signed=True),
            ]
            for index, row in enumerate(analysis["market_summary"])
        ],
    )
    paired_table = _table(
        ["Comparison", "Median difference", "Mean difference", "MARL-side wins"],
        [
            [
                html.escape(row["label"]), f"{_fmt(row['median_difference_pp'], signed=True)} pp",
                f"{_fmt(row['mean_difference_pp'], signed=True)} pp", f"{row['right_wins']}/{row['runs']}",
            ]
            for row in analysis["paired_comparisons"]
        ],
    )
    chart_html = {chart["title"]: f'<img src="charts/{chart["file"]}" alt="{html.escape(chart["title"])}">' for chart in charts}
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage A MARL Evaluation</title><style>
:root {{ --ink:#1F2430; --muted:#6F768A; --paper:#FCFCFD; --panel:#fff; --line:#D7DBE7; --blue:#5477C4; --orange:#CC6F47; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; }}
main {{ width:min(1080px,calc(100% - 32px)); margin:0 auto; padding:42px 0 70px; }} header {{ border-bottom:2px solid var(--ink); padding-bottom:24px; margin-bottom:32px; }} h1 {{ font-size:2.25rem; margin:0 0 8px; }} h2 {{ font-size:1.35rem; margin:0 0 12px; }} h3 {{ font-size:1.02rem; margin:0 0 8px; }} p,li {{ line-height:1.6; }} .deck {{ color:var(--muted); max-width:850px; margin:0; }} section {{ margin:34px 0; }} .callout {{ background:#EAF1FE; border-left:4px solid var(--blue); padding:14px 16px; line-height:1.55; }} figure {{ margin:18px 0 24px; }} figure img {{ width:100%; display:block; border:1px solid var(--line); background:var(--panel); }} figcaption {{ color:var(--muted); font-size:.9rem; margin-top:7px; line-height:1.5; }} .table-wrap {{ overflow-x:auto; border:1px solid var(--line); background:var(--panel); }} table {{ width:100%; border-collapse:collapse; font-size:.84rem; min-width:730px; }} th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }} th {{ color:var(--muted); font-size:.72rem; letter-spacing:.04em; text-transform:uppercase; }} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }} tr:last-child td {{ border-bottom:0; }} code {{ background:#F1F3F7; padding:2px 4px; }} .small {{ color:var(--muted); font-size:.9rem; }} @media(max-width:640px) {{ main {{ width:min(100% - 22px,1080px); padding-top:26px; }} h1 {{ font-size:1.75rem; }} }}
</style></head><body><main data-report-audience="technical">
<header data-contract-section="title"><h1>Stage A MARL Evaluation</h1><p class="deck">Independent PPO market agents evaluated across five deployed populations, held-out seeds 31-40, and a 10,000-step synthetic limit-order-book horizon.</p></header>
<section data-contract-section="technical-summary"><h2>Technical Summary</h2>
<div class="callout"><strong>This report tests joint self-interested MARL, not explicit cooperation.</strong> Relative to the sequential RL trio in the matched seven-agent ecology, the 7-agent-trained MARL coalition differs by <strong>{_fmt(m7_vs_rl['median_difference_pp'], signed=True)} percentage points</strong> of coalition return at the median; the 10-agent-trained MARL coalition differs by <strong>{_fmt(m10_vs_rl['median_difference_pp'], signed=True)} pp</strong>. In the crowded ten-agent ecology, MARL minus RL coalition return is <strong>{_fmt(m7_crowd['median_difference_pp'], signed=True)} pp</strong> for the 7-agent-trained MARL population and <strong>{_fmt(m10_crowd['median_difference_pp'], signed=True)} pp</strong> for the 10-agent-trained population. The strongest individual MARL result is <strong>{html.escape(top_marl['agent_id'])}</strong> in <strong>{html.escape(STRUCTURE_SHORT[top_marl['structure_id']])}</strong>, with median return <strong>{_fmt(top_marl['median_pnl_pct'], signed=True)}%</strong>.</div></section>

<section data-contract-section="key-findings"><h2>Coalition returns show whether joint training improves the three-agent cohort</h2><p>Returns are normalized by combined starting equity, so a three-agent MARL coalition is compared fairly with the three-agent RL coalition. The seven-agent contrasts share seeds and exogenous paths but have different endogenous market paths; the crowded contrasts place RL and MARL in the exact same runs.</p><figure>{chart_html['Coalition profitability']}<figcaption>Median coalition return by structure and learned cohort.</figcaption></figure>{coalition_table}<p class="small">The paired differences below are MARL minus RL. Positive values favor MARL, but this is evidence from one trained realization of each population rather than a distribution across independent training restarts.</p>{paired_table}</section>

<section><h2>Individual MARL behavior is heterogeneous, but it is not cooperation evidence</h2><p>Individual returns identify who performs best inside each ecology. The activity ledger shows whether MARL policies take distinct directional and trading-frequency roles. For example, the top MARL policy above has median net executed flow of {_fmt(top_marl['median_net_executed_flow'], signed=True)} shares and {_fmt(top_marl['median_submitted_orders'], digits=0)} submitted orders. Different roles can emerge under independent objectives, but they do not prove intention, communication, or coalition optimization.</p><figure>{chart_html['Individual policy returns']}<figcaption>Median individual return for each learned policy within its deployed structure.</figcaption></figure><figure>{chart_html['MARL trading behavior']}<figcaption>Order flow and activity of MARL policies. Positive net flow means net buying.</figcaption></figure>{behavior_table}</section>

<section><h2>Adding six learned policies changes the market ecology</h2><p>Trade count and spread availability are market-quality guardrails. A learned cohort should not be called better solely because it earns PnL if it only does so by eliminating usable quotes or creating extreme dislocations.</p><figure>{chart_html['Market quality']}<figcaption>Structures map to the numbered rows in the market table.</figcaption></figure>{market_table}</section>

<section data-contract-section="scope-data-and-metric-definitions"><h2>Scope and metric definitions</h2><p>The dataset contains <strong>{analysis['reports']} deterministic evaluations</strong>: five population structures across held-out seeds <code>{', '.join(str(seed) for seed in analysis['seeds'])}</code>. Every learned participant starts with 16 shares and the same starting-cash convention. Total PnL is ending equity minus starting equity; realized and unrealized PnL are reported separately. Coalition return is total PnL divided by combined starting equity. Spread availability is the fraction of snapshots with a valid two-sided quoted spread.</p></section>

<section data-contract-section="methodology"><h2>Method: fixed policy evaluation with a shared-market rollout</h2><p>Each population is evaluated deterministically. One PPO checkpoint is the runtime-controlled policy and its peers are frozen runtime checkpoints; this is only an implementation device, not online training. For each seed, all structures share the same exogenous seed, while trades, order-book state, and realized PnL remain endogenous to the deployed population. The Stage A MARL policies are independent MaskablePPO policies trained jointly with individual rewards, separate critics, separate rollout buffers, and no shared team reward.</p></section>

<section data-contract-section="limitations-uncertainty-and-robustness-checks"><h2>Limitations and interpretation guardrails</h2><ul><li>Ten held-out seeds measure evaluation variability but do not estimate variability across independent MARL training runs.</li><li>Seven-agent and ten-agent results answer different ecological questions because agents receive different decision opportunities at a fixed horizon.</li><li>Participant order is deterministic. In the current ten-agent setup, the frozen RL agents act before MARL in every cycle. Each agent gets the same number of turns, but MARL acts on a book already changed by RL; the reported gap is therefore an ordered-ecology result, not yet an order-neutral policy-quality result.</li><li>Higher MARL coalition PnL is not proof of cooperation; Stage A contains no communication channel, centralized critic, or shared reward.</li><li>The scripted fundamental-reactive agent remains a benchmark participant in every treatment; this report does not make a private-information claim.</li></ul></section>

<section data-contract-section="recommended-next-steps"><h2>Recommended next steps</h2><ol><li>Use the report's best Stage A population as the individual-objective benchmark, without promotion until the user reviews the seed-level evidence.</li><li>Run at least two independent MARL training seeds before making an algorithm-level claim.</li><li>Implement the Stage B cooperative objective with separate <code>marl_coop_*</code> checkpoints and evaluate it using this same panel.</li></ol></section>

<section data-contract-section="further-questions"><h2>Further questions</h2><p>Does any apparent MARL specialization persist across independently trained populations, and does it improve coalition return without weakening liquidity? The next cooperative treatment should answer that directly rather than infer cooperation from aligned PnL.</p></section>
</main></body></html>"""
    output = output_dir / "stage_a_marl_evaluation_report.html"
    output.write_text(document, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "technical_report"
    analysis = build_analysis(input_dir)
    charts = render_charts(analysis, output_dir / "charts")
    summary_path = output_dir / "stage_a_marl_analysis.json"
    summary_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    source_notes = {
        "report": "Stage A MARL Evaluation",
        "input_dir": str(input_dir),
        "reports": len(analysis["individual_rows"]),
        "source_manifest": str(input_dir / "manifest.json"),
        "metrics": ["per-agent realized/unrealized/total PnL", "coalition return", "drawdown", "inventory exposure", "submitted orders", "executed order flow", "market quality"],
        "chart_renderer": "Matplotlib static PNG fallback; Seaborn is not installed in the project virtual environment.",
        "charts": charts,
    }
    (output_dir / "source_notes.json").write_text(json.dumps(source_notes, indent=2) + "\n", encoding="utf-8")
    report_path = render_html(analysis, output_dir, charts)
    print(f"Wrote report to {report_path}")
    print(f"Wrote analysis to {summary_path}")


if __name__ == "__main__":
    main()
