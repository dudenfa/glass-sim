from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


MARKET_METRICS = (
    ("trade_count", "Trades", True, 0),
    ("spread_availability_ratio", "Spread availability", True, 3),
    ("mean_spread", "Mean spread", False, 4),
    ("midpoint_return_volatility_bps", "Volatility (bps)", False, 2),
    ("mean_top_of_book_liquidity", "Top-book liquidity", True, 2),
)
FINAL_PRICE_KEY = "final_midpoint"
RL_AGENT_IDS = ("rl_01", "rl_02", "rl_03")
_SEED_PATTERN = re.compile(r"seed_(\d+)_all_cashonly\.json$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained dashboard for the RL inventory-endowment ablation.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing seed_N_all_cashonly.json and seed_N_all_endowed_16.json reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to inventory_ablation_dashboard.html in the input directory.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _agent_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("portfolio_breakdown", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("agent_id")): dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("agent_id") is not None
    }


def _value(payload: dict[str, Any], key: str) -> float:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return 0.0
    value = summary.get(key)
    return float(value) if value is not None else 0.0


def build_dashboard_payload(input_dir: Path) -> dict[str, Any]:
    directory = input_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    runs: list[dict[str, Any]] = []
    for cash_path in sorted(directory.glob("seed_*_all_cashonly.json")):
        match = _SEED_PATTERN.search(cash_path.name)
        if match is None:
            continue
        seed = int(match.group(1))
        endowed_path = directory / f"seed_{seed}_all_endowed_16.json"
        if not endowed_path.exists():
            continue
        cash = _load_json(cash_path)
        endowed = _load_json(endowed_path)
        cash_agents = _agent_rows(cash)
        endowed_agents = _agent_rows(endowed)
        runs.append(
            {
                "seed": seed,
                "cash": {
                    key: _value(cash, key)
                    for key in (*[key for key, _, _, _ in MARKET_METRICS], FINAL_PRICE_KEY)
                },
                "endowed": {
                    key: _value(endowed, key)
                    for key in (*[key for key, _, _, _ in MARKET_METRICS], FINAL_PRICE_KEY)
                },
                "pnl": {
                    agent_id: {
                        "cash": float(cash_agents.get(agent_id, {}).get("total_pnl", 0.0)),
                        "endowed": float(endowed_agents.get(agent_id, {}).get("total_pnl", 0.0)),
                    }
                    for agent_id in RL_AGENT_IDS
                },
            }
        )
    if not runs:
        raise ValueError(
            "No paired reports found. Expected seed_N_all_cashonly.json and seed_N_all_endowed_16.json files."
        )

    metrics: list[dict[str, Any]] = []
    for key, label, higher_is_better, digits in MARKET_METRICS:
        cash_values = [run["cash"][key] for run in runs]
        endowed_values = [run["endowed"][key] for run in runs]
        deltas = [endowed - cash for cash, endowed in zip(cash_values, endowed_values)]
        metrics.append(
            {
                "key": key,
                "label": label,
                "higherIsBetter": higher_is_better,
                "digits": digits,
                "cashMean": mean(cash_values),
                "endowedMean": mean(endowed_values),
                "deltaMean": mean(deltas),
                "deltaMedian": median(deltas),
                "wins": sum(delta * (1 if higher_is_better else -1) > 0 for delta in deltas),
            }
        )

    agents: list[dict[str, Any]] = []
    for agent_id in RL_AGENT_IDS:
        cash_values = [run["pnl"][agent_id]["cash"] for run in runs]
        endowed_values = [run["pnl"][agent_id]["endowed"] for run in runs]
        deltas = [endowed - cash for cash, endowed in zip(cash_values, endowed_values)]
        agents.append(
            {
                "agentId": agent_id,
                "cashMean": mean(cash_values),
                "endowedMean": mean(endowed_values),
                "deltaMean": mean(deltas),
                "deltaMedian": median(deltas),
                "wins": sum(delta > 0 for delta in deltas),
            }
        )

    return {
        "seedCount": len(runs),
        "runs": runs,
        "metrics": metrics,
        "agents": agents,
    }


def _dashboard_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inventory Endowment Ablation</title>
  <style>
    :root {{
      --ink: #15202b;
      --muted: #66727e;
      --line: #d8dee4;
      --paper: #f5f7f8;
      --surface: #ffffff;
      --cash: #d76a52;
      --endowed: #138a72;
      --accent: #4469a8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: Arial, Helvetica, sans-serif; }}
    main {{ width: min(1320px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 48px; }}
    header {{ border-bottom: 2px solid var(--ink); padding-bottom: 20px; margin-bottom: 22px; }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-family: Georgia, serif; font-size: clamp(2rem, 5vw, 3.4rem); font-weight: 500; letter-spacing: 0; }}
    h2 {{ font-size: 1.05rem; }}
    .lede {{ margin-top: 10px; color: var(--muted); max-width: 760px; line-height: 1.5; }}
    .legend {{ display: flex; gap: 18px; margin-top: 18px; color: var(--muted); font-size: .86rem; }}
    .legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 7px; border-radius: 50%; }}
    .legend .cash::before {{ background: var(--cash); }}
    .legend .endowed::before {{ background: var(--endowed); }}
    .grid {{ display: grid; gap: 16px; }}
    .summary-grid {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
    .card, .chart-card, .table-card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
    .card {{ padding: 14px; min-height: 122px; }}
    .card-label {{ color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; }}
    .card-value {{ margin-top: 10px; font-family: Georgia, serif; font-size: 1.62rem; }}
    .card-detail {{ margin-top: 7px; color: var(--muted); font-size: .82rem; line-height: 1.35; }}
    .section {{ margin-top: 28px; }}
    .section-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 12px; }}
    .section-head p {{ color: var(--muted); font-size: .86rem; }}
    .charts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .chart-card {{ padding: 14px; min-width: 0; }}
    .chart-card h3 {{ font-size: .93rem; }}
    .chart-card p {{ margin-top: 4px; color: var(--muted); font-size: .78rem; }}
    .chart {{ width: 100%; margin-top: 10px; }}
    .agent-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .table-card {{ overflow-x: auto; }}
    table {{ width: 100%; min-width: 900px; border-collapse: collapse; font-size: .83rem; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    .positive {{ color: var(--endowed); font-weight: 700; }}
    .negative {{ color: var(--cash); font-weight: 700; }}
    .note {{ margin-top: 22px; color: var(--muted); font-size: .82rem; line-height: 1.45; }}
    @media (max-width: 900px) {{ .summary-grid, .agent-grid, .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Inventory Endowment Ablation</h1>
      <p class="lede">Paired evaluation of the same learned policies in a seven-agent synthetic market: all RL agents start cash-only versus each starting with 16 shares.</p>
      <div class="legend"><span class="cash">Cash-only</span><span class="endowed">Endowed (16 shares each)</span></div>
    </header>
    <section class="grid summary-grid" id="metricCards"></section>
    <section class="section">
      <div class="section-head"><h2>Market Quality By Seed</h2><p>One line is one paired holdout seed.</p></div>
      <div class="grid charts" id="marketCharts"></div>
    </section>
    <section class="section">
      <div class="section-head"><h2>Learned-Agent PnL By Seed</h2><p>PNL is measured relative to each agent's own starting equity.</p></div>
      <div class="grid agent-grid" id="agentCharts"></div>
    </section>
    <section class="section">
      <div class="section-head"><h2>Paired Differences</h2><p>Endowed minus cash-only.</p></div>
      <div class="table-card"><table id="differenceTable"></table></div>
    </section>
    <p class="note">This dashboard is exploratory. Starting inventory changes total initial wealth, so aggregate equity is intentionally excluded. The current simulator may also produce different exogenous fundamental paths across configurations sharing one numerical seed; interpret results as a holdout distribution comparison rather than a perfect same-path counterfactual.</p>
  </main>
  <script>
    const data = {data};
    const colors = {{ cash: '#d76a52', endowed: '#138a72', grid: '#d8dee4', ink: '#15202b', muted: '#66727e' }};
    const metricCards = document.getElementById('metricCards');
    const marketCharts = document.getElementById('marketCharts');
    const agentCharts = document.getElementById('agentCharts');
    const differenceTable = document.getElementById('differenceTable');
    const fmt = (value, digits = 2) => Number(value).toLocaleString(undefined, {{ minimumFractionDigits: digits, maximumFractionDigits: digits }});
    const sign = (value, digits = 2) => `${{value >= 0 ? '+' : '-'}}${{fmt(Math.abs(value), digits)}}`;
    const svg = (tag, attributes = {{}}) => {{ const node = document.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value)); return node; }};
    const metricDigits = (key) => data.metrics.find((metric) => metric.key === key)?.digits ?? 2;

    function renderSummary() {{
      metricCards.innerHTML = data.metrics.map((metric) => {{
        const direction = metric.higherIsBetter ? 1 : -1;
        const improved = metric.deltaMean * direction > 0;
        return `<article class="card"><div class="card-label">${{metric.label}}</div><div class="card-value ${{improved ? 'positive' : 'negative'}}">${{sign(metric.deltaMean, metric.digits)}}</div><div class="card-detail">mean paired delta<br>${{metric.wins}}/${{data.seedCount}} seeds improve</div></article>`;
      }}).join('');
    }}

    function renderSlopeChart(container, title, subtitle, entries, digits) {{
      const width = 560, height = 270, top = 34, bottom = 32, left = 74, right = 52;
      const values = entries.flatMap((entry) => [entry.cash, entry.endowed]);
      const min = Math.min(...values), max = Math.max(...values);
      const span = Math.max(0.0001, max - min);
      const y = (value) => top + (1 - (value - min) / span) * (height - top - bottom);
      const chart = svg('svg', {{ viewBox: `0 0 ${{width}} ${{height}}`, class: 'chart', role: 'img', 'aria-label': title }});
      for (let index = 0; index <= 4; index += 1) {{
        const value = max - (span * index) / 4;
        const yPos = y(value);
        chart.append(svg('line', {{ x1: left, x2: width - right, y1: yPos, y2: yPos, stroke: colors.grid, 'stroke-width': 1 }}));
        const label = svg('text', {{ x: left - 8, y: yPos + 4, 'text-anchor': 'end', fill: colors.muted, 'font-size': 10 }});
        label.textContent = fmt(value, digits); chart.append(label);
      }}
      const labels = [[left + 90, 'cash-only'], [width - right - 90, 'endowed']];
      labels.forEach(([x, labelText]) => {{ const label = svg('text', {{ x, y: height - 8, 'text-anchor': 'middle', fill: colors.muted, 'font-size': 11 }}); label.textContent = labelText; chart.append(label); }});
      entries.forEach((entry, index) => {{
        const x1 = left + 90, x2 = width - right - 90, y1 = y(entry.cash), y2 = y(entry.endowed);
        chart.append(svg('line', {{ x1, x2, y1, y2, stroke: '#aeb8c1', 'stroke-width': 1.5, opacity: .78 }}));
        chart.append(svg('circle', {{ cx: x1, cy: y1, r: 4, fill: colors.cash }}));
        chart.append(svg('circle', {{ cx: x2, cy: y2, r: 4, fill: colors.endowed }}));
        const seedLabel = svg('text', {{ x: width - right + 7, y: y2 + 4, fill: colors.muted, 'font-size': 10 }});
        seedLabel.textContent = entry.seed; chart.append(seedLabel);
      }});
      const card = document.createElement('article'); card.className = 'chart-card';
      card.innerHTML = `<h3>${{title}}</h3><p>${{subtitle}}</p>`; card.append(chart); container.append(card);
    }}

    function renderCharts() {{
      data.metrics.forEach((metric) => renderSlopeChart(
        marketCharts,
        metric.label,
        `${{metric.wins}}/${{data.seedCount}} seeds improve; mean delta ${{sign(metric.deltaMean, metric.digits)}}`,
        data.runs.map((run) => ({{ seed: run.seed, cash: run.cash[metric.key], endowed: run.endowed[metric.key] }})),
        metric.digits,
      ));
      data.agents.forEach((agent) => renderSlopeChart(
        agentCharts,
        agent.agentId,
        `${{agent.wins}}/${{data.seedCount}} seeds improve; mean PnL delta ${{sign(agent.deltaMean, 2)}}`,
        data.runs.map((run) => ({{ seed: run.seed, cash: run.pnl[agent.agentId].cash, endowed: run.pnl[agent.agentId].endowed }})),
        0,
      ));
    }}

    function renderTable() {{
      const tableMetrics = [{{ key: 'final_midpoint', label: 'Final price', digits: 2 }}, ...data.metrics];
      const header = ['Seed', ...tableMetrics.map((metric) => metric.label), ...data.agents.map((agent) => `${{agent.agentId}} PnL`)];
      const body = data.runs.map((run) => {{
        const cells = [String(run.seed)];
        tableMetrics.forEach((metric) => cells.push(run.endowed[metric.key] - run.cash[metric.key]));
        data.agents.forEach((agent) => cells.push(run.pnl[agent.agentId].endowed - run.pnl[agent.agentId].cash));
        return cells;
      }});
      differenceTable.innerHTML = `<thead><tr>${{header.map((label) => `<th>${{label}}</th>`).join('')}}</tr></thead><tbody>${{body.map((row) => `<tr>${{row.map((value, index) => {{ if (index === 0) return `<td>${{value}}</td>`; const metric = index <= tableMetrics.length ? tableMetrics[index - 1] : null; const better = metric?.higherIsBetter === undefined ? Number(value) > 0 : Number(value) * (metric.higherIsBetter ? 1 : -1) > 0; const digits = metric ? metric.digits : 2; return `<td class="${{better ? 'positive' : 'negative'}}">${{sign(Number(value), digits)}}</td>`; }}).join('')}}</tr>`).join('')}}</tbody>`;
    }}

    renderSummary(); renderCharts(); renderTable();
  </script>
</body>
</html>"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    payload = build_dashboard_payload(input_dir)
    output_path = args.output.resolve() if args.output is not None else input_dir / "inventory_ablation_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_dashboard_html(payload), encoding="utf-8")
    print(f"Wrote inventory ablation dashboard to {output_path}")


if __name__ == "__main__":
    main()
