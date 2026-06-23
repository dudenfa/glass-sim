from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


_REPORT_PATTERN = re.compile(r"(?P<structure>.+)_seed_(?P<seed>\d+)\.json$")
MARKET_COLUMNS = (
    ("trade_count", "Trades", 0),
    ("spread_availability_ratio", "Spread availability", 3),
    ("mean_spread", "Mean spread", 4),
    ("midpoint_return_volatility_bps", "Volatility (bps)", 2),
    ("mean_top_of_book_liquidity", "Top-book liquidity", 2),
    ("final_midpoint", "Final price", 2),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dashboard and markdown ranking from an agent-ranking panel.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--html-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args(argv)


def _number(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return median(_number(row.get(key)) for row in rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(_number(row.get(key)) for row in rows)


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def build_ranking_payload(input_dir: Path) -> dict[str, Any]:
    directory = input_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    manifest_path = directory / "manifest.json"
    manifest = _load_report(manifest_path) if manifest_path.exists() else {}
    structure_order = [item["structure_id"] for item in manifest.get("structures", []) if isinstance(item, dict)]
    labels = {
        str(item["structure_id"]): str(item.get("label", item["structure_id"]))
        for item in manifest.get("structures", [])
        if isinstance(item, dict) and item.get("structure_id") is not None
    }

    reports: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*_seed_*.json")):
        match = _REPORT_PATTERN.match(path.name)
        if match is None:
            continue
        payload = _load_report(path)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        structure_id = str(metadata.get("ranking_structure_id", match.group("structure")))
        label = str(metadata.get("ranking_structure_label", labels.get(structure_id, structure_id)))
        labels[structure_id] = label
        reports.append(
            {
                "path": str(path),
                "structureId": structure_id,
                "structureLabel": label,
                "seed": int(payload.get("seed", match.group("seed"))),
                "summary": dict(payload.get("summary", {})),
                "portfolio": list(payload.get("portfolio_breakdown", []) or []),
            }
        )
    if not reports:
        raise ValueError("No ranking report files found.")

    rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    for report in reports:
        market_rows.append(
            {
                "structureId": report["structureId"],
                "structureLabel": report["structureLabel"],
                "seed": report["seed"],
                **{key: _number(report["summary"].get(key)) for key, _, _ in MARKET_COLUMNS},
            }
        )
        for portfolio in report["portfolio"]:
            if not isinstance(portfolio, dict):
                continue
            starting_equity = _number(portfolio.get("starting_equity"))
            total_pnl = _number(portfolio.get("total_pnl", portfolio.get("equity_delta")))
            rows.append(
                {
                    "structureId": report["structureId"],
                    "structureLabel": report["structureLabel"],
                    "seed": report["seed"],
                    "agentId": str(portfolio.get("agent_id")),
                    "agentType": str(portfolio.get("agent_type", "unknown")),
                    "startingEquity": starting_equity,
                    "realizedPnl": _number(portfolio.get("realized_pnl")),
                    "unrealizedPnl": _number(portfolio.get("unrealized_pnl")),
                    "totalPnl": total_pnl,
                    "pnlPct": 100.0 * total_pnl / starting_equity if starting_equity else 0.0,
                    "maxDrawdown": _number(portfolio.get("max_equity_drawdown")),
                    "maxAbsInventory": _number(portfolio.get("max_abs_inventory")),
                    "endingEquity": _number(portfolio.get("ending_equity")),
                }
            )

    by_run: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[(row["structureId"], row["seed"])].append(row)
    for run_rows in by_run.values():
        ordered = sorted(run_rows, key=lambda row: (-float(row["pnlPct"]), str(row["agentId"])))
        for rank, row in enumerate(ordered, start=1):
            row["pnlPctRank"] = rank

    by_structure_agent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_structure_agent[(row["structureId"], row["agentId"])].append(row)
        by_agent[row["agentId"]].append(row)

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "runs": len(group),
            "medianRealizedPnl": _median(group, "realizedPnl"),
            "medianUnrealizedPnl": _median(group, "unrealizedPnl"),
            "medianTotalPnl": _median(group, "totalPnl"),
            "medianPnlPct": _median(group, "pnlPct"),
            "meanPnlPct": _mean(group, "pnlPct"),
            "medianRank": _median(group, "pnlPctRank"),
            "wins": sum(int(row["pnlPctRank"]) == 1 for row in group),
            "meanDrawdown": _mean(group, "maxDrawdown"),
            "meanMaxAbsInventory": _mean(group, "maxAbsInventory"),
        }

    structure_leaderboards: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (structure_id, agent_id), group in by_structure_agent.items():
        structure_leaderboards[structure_id].append(
            {
                "agentId": agent_id,
                "agentType": group[0]["agentType"],
                **aggregate(group),
            }
        )
    for leaderboard in structure_leaderboards.values():
        leaderboard.sort(key=lambda row: (-float(row["medianPnlPct"]), float(row["medianRank"]), str(row["agentId"])))

    overall = [
        {"agentId": agent_id, "agentType": group[0]["agentType"], **aggregate(group)}
        for agent_id, group in by_agent.items()
    ]
    overall.sort(key=lambda row: (-float(row["medianPnlPct"]), float(row["medianRank"]), str(row["agentId"])))

    market_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in market_rows:
        market_by_structure[row["structureId"]].append(row)
    market_summary = []
    for structure_id, group in market_by_structure.items():
        market_summary.append(
            {
                "structureId": structure_id,
                "structureLabel": labels[structure_id],
                **{key: _median(group, key) for key, _, _ in MARKET_COLUMNS},
            }
        )
    ordered_structure_ids = [item for item in structure_order if item in labels]
    ordered_structure_ids.extend(sorted(set(labels) - set(ordered_structure_ids)))
    market_summary.sort(key=lambda row: ordered_structure_ids.index(row["structureId"]))

    return {
        "manifest": manifest,
        "seedCount": len({report["seed"] for report in reports}),
        "reportCount": len(reports),
        "structureOrder": ordered_structure_ids,
        "labels": labels,
        "rows": rows,
        "structureLeaderboards": dict(structure_leaderboards),
        "overall": overall,
        "marketSummary": market_summary,
    }


def _format(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Fixed-16 Agent Ranking Panel", ""]
    lines.append(f"- Seeds: `{payload['seedCount']}`")
    lines.append(f"- Completed runs: `{payload['reportCount']}`")
    lines.append("- Ranking metric: median total PnL as a percentage of each agent's starting equity.")
    lines.append("")
    for structure_id in payload["structureOrder"]:
        lines.extend([f"## {payload['labels'][structure_id]}", ""])
        lines.append("| Rank | Agent | Median PnL % | Realized | Unrealized | Total PnL | Wins | Median rank |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for rank, row in enumerate(payload["structureLeaderboards"].get(structure_id, []), start=1):
            lines.append(
                "| {rank} | {agent} | {pnl_pct} | {realized} | {unrealized} | {total} | {wins}/{runs} | {median_rank} |".format(
                    rank=rank,
                    agent=row["agentId"],
                    pnl_pct=_format(row["medianPnlPct"]),
                    realized=_format(row["medianRealizedPnl"]),
                    unrealized=_format(row["medianUnrealizedPnl"]),
                    total=_format(row["medianTotalPnl"]),
                    wins=row["wins"],
                    runs=row["runs"],
                    median_rank=_format(row["medianRank"], 1),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "The cross-structure table is a robustness summary, not a single universal leaderboard: agents appear in different opponent sets and may have different numbers of observations. Use the per-structure leaderboards for direct competition claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _dashboard_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fixed-16 Agent Ranking Panel</title><style>
:root {{ --ink:#18242e; --muted:#64727d; --paper:#f4f6f7; --surface:#fff; --line:#d4dde2; --good:#117b62; --bad:#c75d4b; --accent:#416ba8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Arial,Helvetica,sans-serif; }}
main {{ width:min(1420px,calc(100% - 32px)); margin:auto; padding:30px 0 52px; }} header {{ border-bottom:2px solid var(--ink); padding-bottom:20px; }}
h1,h2,h3,p {{ margin:0; }} h1 {{ font:500 clamp(2rem,5vw,3.4rem) Georgia,serif; }} h2 {{ font-size:1.12rem; }} h3 {{ font-size:.95rem; }} .lede,.note {{ color:var(--muted); line-height:1.5; }} .lede {{ max-width:860px; margin-top:10px; }}
.grid {{ display:grid; gap:16px; }} .cards {{ grid-template-columns:repeat(4,minmax(0,1fr)); margin-top:22px; }} .card,.panel {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; }} .card {{ padding:14px; }} .label {{ color:var(--muted); font-size:.75rem; letter-spacing:.07em; text-transform:uppercase; }} .value {{ margin-top:9px; font:500 1.7rem Georgia,serif; }}
.section {{ margin-top:30px; }} .section-head {{ display:flex; justify-content:space-between; gap:16px; align-items:baseline; margin-bottom:12px; }} .section-head p {{ color:var(--muted); font-size:.84rem; }}
.panel {{ overflow-x:auto; }} table {{ width:100%; min-width:900px; border-collapse:collapse; font-size:.84rem; }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap; text-align:right; }} th {{ color:var(--muted); font-size:.7rem; letter-spacing:.06em; text-transform:uppercase; }} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }} tr:last-child td {{ border-bottom:0; }} .positive {{ color:var(--good); font-weight:700; }} .negative {{ color:var(--bad); font-weight:700; }}
.controls {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }} button {{ border:1px solid var(--line); background:var(--surface); color:var(--ink); border-radius:4px; padding:8px 10px; cursor:pointer; }} button[aria-pressed="true"] {{ background:var(--accent); border-color:var(--accent); color:#fff; }} .note {{ margin-top:18px; font-size:.82rem; }}
@media(max-width:900px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }} @media(max-width:520px) {{ main {{ width:min(100% - 20px,1420px); }} .cards {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<header><h1>Fixed-16 Agent Ranking Panel</h1><p class="lede">Current scripted and learned policies evaluated as the hybrid ecology grows from four scripted participants to seven participants with three learned policies.</p></header>
<section class="grid cards" id="cards"></section>
<section class="section"><div class="section-head"><h2>Cross-Structure Robustness</h2><p>Useful context, but not a direct head-to-head leaderboard.</p></div><div class="panel"><table id="overall"></table></div></section>
<section class="section"><div class="section-head"><h2>Within-Structure Leaderboards</h2><p>Directly comparable participants share every run and exogenous path.</p></div><div class="controls" id="structureControls"></div><div class="panel"><table id="leaderboard"></table></div></section>
<section class="section"><div class="section-head"><h2>Market Context</h2><p>Median market-quality outcomes by structure.</p></div><div class="panel"><table id="market"></table></div></section>
<section class="section"><div class="section-head"><h2>Seed-Level PnL Detail</h2><p>Realized, unrealized, total PnL, return, risk, and inventory exposure.</p></div><div class="panel"><table id="seedDetail"></table></div></section>
<p class="note">Rankings use total PnL divided by each agent's starting equity. Realized and unrealized PnL remain visible separately because a favorable end-of-run mark is not the same thing as realized trading profit. The exogenous news and fundamental RNG stream is held fixed for each shared seed across structures.</p>
</main><script>const data={data}; const fmt=(v,d=2)=>Number(v).toLocaleString(undefined,{{minimumFractionDigits:d,maximumFractionDigits:d}}); const pct=(v)=>`${{fmt(v)}}%`; const cls=(v)=>Number(v)>=0?'positive':'negative';
const order=data.structureOrder; let selected=order[0];
function table(id,headers,body) {{ document.getElementById(id).innerHTML=`<thead><tr>${{headers.map(x=>`<th>${{x}}</th>`).join('')}}</tr></thead><tbody>${{body}}</tbody>`; }}
function cells(row) {{ return `<td>${{row}}</td>`; }}
document.getElementById('cards').innerHTML=[['Structures',order.length],['Holdout seeds',data.seedCount],['Completed runs',data.reportCount],['RL endowment',`${{data.manifest.rl_starting_inventory ?? 16}} shares`]].map(([l,v])=>`<article class="card"><div class="label">${{l}}</div><div class="value">${{v}}</div></article>`).join('');
table('overall',['Agent','Type','Runs','Median PnL %','Median PnL','Wins','Median rank','Mean drawdown'],data.overall.map(r=>`<tr>${{cells(r.agentId)}}${{cells(r.agentType)}}${{cells(r.runs)}}<td class="${{cls(r.medianPnlPct)}}">${{pct(r.medianPnlPct)}}</td><td class="${{cls(r.medianTotalPnl)}}">${{fmt(r.medianTotalPnl)}}</td>${{cells(`${{r.wins}}/${{r.runs}}`)}}${{cells(fmt(r.medianRank,1))}}${{cells(fmt(r.meanDrawdown))}}</tr>`).join(''));
function renderSelected() {{ const leaders=data.structureLeaderboards[selected]||[]; table('leaderboard',['Rank','Agent','Type','Runs','Median realized','Median unrealized','Median total PnL','Median PnL %','Wins','Median rank','Mean max |inventory|'],leaders.map((r,i)=>`<tr>${{cells(i+1)}}${{cells(r.agentId)}}${{cells(r.agentType)}}${{cells(r.runs)}}<td class="${{cls(r.medianRealizedPnl)}}">${{fmt(r.medianRealizedPnl)}}</td><td class="${{cls(r.medianUnrealizedPnl)}}">${{fmt(r.medianUnrealizedPnl)}}</td><td class="${{cls(r.medianTotalPnl)}}">${{fmt(r.medianTotalPnl)}}</td><td class="${{cls(r.medianPnlPct)}}">${{pct(r.medianPnlPct)}}</td>${{cells(`${{r.wins}}/${{r.runs}}`)}}${{cells(fmt(r.medianRank,1))}}${{cells(fmt(r.meanMaxAbsInventory))}}</tr>`).join('')); const rows=data.rows.filter(r=>r.structureId===selected).sort((a,b)=>a.seed-b.seed||a.pnlPctRank-b.pnlPctRank); table('seedDetail',['Seed','Rank','Agent','Type','Start equity','Realized','Unrealized','Total PnL','PnL %','Max drawdown','Max |inventory|'],rows.map(r=>`<tr>${{cells(r.seed)}}${{cells(r.pnlPctRank)}}${{cells(r.agentId)}}${{cells(r.agentType)}}${{cells(fmt(r.startingEquity))}}<td class="${{cls(r.realizedPnl)}}">${{fmt(r.realizedPnl)}}</td><td class="${{cls(r.unrealizedPnl)}}">${{fmt(r.unrealizedPnl)}}</td><td class="${{cls(r.totalPnl)}}">${{fmt(r.totalPnl)}}</td><td class="${{cls(r.pnlPct)}}">${{pct(r.pnlPct)}}</td>${{cells(fmt(r.maxDrawdown))}}${{cells(fmt(r.maxAbsInventory))}}</tr>`).join('')); document.querySelectorAll('#structureControls button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.structure===selected))); }}
document.getElementById('structureControls').innerHTML=order.map(id=>`<button data-structure="${{id}}" aria-pressed="false">${{data.labels[id]}}</button>`).join(''); document.querySelectorAll('#structureControls button').forEach(b=>b.addEventListener('click',()=>{{selected=b.dataset.structure;renderSelected();}}));
table('market',['Structure','Trades','Spread availability','Mean spread','Volatility (bps)','Top-book liquidity','Final price'],data.marketSummary.map(r=>`<tr>${{cells(r.structureLabel)}}${{cells(fmt(r.trade_count,0))}}${{cells(fmt(r.spread_availability_ratio,3))}}${{cells(fmt(r.mean_spread,4))}}${{cells(fmt(r.midpoint_return_volatility_bps))}}${{cells(fmt(r.mean_top_of_book_liquidity))}}${{cells(fmt(r.final_midpoint))}}</tr>`).join('')); renderSelected();</script></body></html>"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    payload = build_ranking_payload(input_dir)
    html_output = args.html_output.resolve() if args.html_output else input_dir / "agent_ranking_dashboard.html"
    summary_output = args.summary_output.resolve() if args.summary_output else input_dir / "agent_ranking_summary.md"
    html_output.write_text(_dashboard_html(payload), encoding="utf-8")
    summary_output.write_text(_markdown(payload), encoding="utf-8")
    print(f"Wrote dashboard to {html_output}")
    print(f"Wrote markdown summary to {summary_output}")


if __name__ == "__main__":
    main()
