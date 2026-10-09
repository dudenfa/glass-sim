"""Evaluate several agents trading together, on held-out seeds.

    python -m glass_sim.rl.marl_evaluate checkpoints/glass/marl3_s0 \\
        --single checkpoints/glass/ppo_full_v2 --seeds 31-40

Each treatment adds the same number N of extra participants (same capital and
decision rate) to the same market, so the treatments are directly comparable:

- ``marl``: the N policies trained together (``marl_01.zip`` ... in the run);
- ``single_copies``: N copies of one single-agent policy trained alone;
- ``scripted_mm``: N scripted market makers (fair control);
- ``none``: no extra participants (market reference).

Reports per-agent and coalition returns (combined PnL / combined capital),
action mixes (specialization), fill quality, participant-type returns and
market quality, as medians over seeds with paired comparisons.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .. import scenarios
from ..agents import MarketMaker
from ..config import parse_duration
from ..market import PerpetualMarket
from ..types import Transparency
from .evaluate import MARKET_KEYS, _parse_seeds, behaviour_metrics
from .policy import policy_agent_from_checkpoint


def _agents_for(treatment: str, n: int, marl_dir: Optional[Path], single_model: Optional[Path], wake: float):
    agents = []
    for i in range(n):
        if treatment == "marl":
            agent = policy_agent_from_checkpoint(marl_dir / f"marl_{i + 1:02d}" / "model.zip", agent_type="marl_agent")
        elif treatment == "single_copies":
            agent = policy_agent_from_checkpoint(single_model, agent_type="rl_copy")
        elif treatment == "scripted_mm":
            agent = MarketMaker(wake_interval=wake)
            agent.agent_type = "control_mm"
        else:
            raise ValueError(treatment)
        agent.agent_id = f"x_{i + 1:02d}"
        agents.append(agent)
    return agents


def run(treatment: str, seed: int, n: int, duration: float, scenario: str, transparency: Optional[str], marl_dir: Optional[Path], single_model: Optional[Path], collateral: float, wake: float) -> Dict[str, Any]:
    config = scenarios.get(scenario, seed=seed, duration=duration)
    if transparency:
        config = replace(config, transparency=Transparency(transparency))
    market = PerpetualMarket(config)
    agents = [] if treatment == "none" else _agents_for(treatment, n, marl_dir, single_model, wake)
    for agent in agents:
        market.add_agent(agent, collateral)
    result = market.run()
    summary = result.summary()
    row: Dict[str, Any] = {
        "seed": seed,
        "treatment": treatment,
        "market": {k: summary.get(k) for k in MARKET_KEYS},
        "group_returns": {g["agent_type"]: round(g["return_pct"], 4) for g in result.group_table()},
        "agents": {},
    }
    if agents:
        snaps = {a.agent_id: a for a in result.accounts}
        pnl = 0.0
        for agent in agents:
            snap = snaps[agent.agent_id]
            pnl += snap.total_pnl
            info = {
                "return_pct": snap.return_pct,
                "max_drawdown_pct": snap.max_drawdown_pct,
                "liquidations": snap.liquidations,
                "maker_share": snap.maker_share,
                "trade_count": snap.trade_count,
                "actions": agent.action_distribution() if hasattr(agent, "action_distribution") else {},
            }
            info.update(behaviour_metrics(result, agent.agent_id))
            row["agents"][agent.agent_id] = info
        row["coalition_return_pct"] = 100.0 * pnl / (collateral * len(agents))
    return row


def _med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def format_report(report: Dict[str, Any]) -> str:
    rows = report["rows"]
    seeds = report["seeds"]
    treatments = list(dict.fromkeys(r["treatment"] for r in rows))
    by = {(r["treatment"], r["seed"]): r for r in rows}
    active = [t for t in treatments if t != "none"]
    lines = [
        f"Multi-agent evaluation: {report['n']} extra participants per treatment, seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}), {report['duration'] / 3600:.1f}h, transparency {report['transparency']}",
        f"marl run: {report.get('marl_dir')}   single model: {report.get('single_model')}",
        "",
        "Coalition return % (combined PnL / combined capital), median over seeds",
    ]
    for t in active:
        vals = [by[(t, s)]["coalition_return_pct"] for s in seeds]
        lines.append(f"  {t:14s} median {_med(vals):+7.2f}   mean {statistics.mean(vals):+7.2f}   positive in {sum(v > 0 for v in vals)}/{len(vals)}")
    if len(active) > 1:
        lines.append("")
        lines.append("Paired coalition differences (percentage points)")
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                d = [by[(a, s)]["coalition_return_pct"] - by[(b, s)]["coalition_return_pct"] for s in seeds]
                lines.append(f"  {a} - {b}: median {statistics.median(d):+.2f}, {a} better in {sum(x > 0 for x in d)}/{len(d)}")
    lines.append("")
    lines.append("Per agent (median over seeds)")
    lines.append(f"  {'treatment':14s} {'agent':6s} {'return %':>9s} {'maxDD%':>7s} {'maker%':>7s} {'long':>5s} {'short':>5s} {'|pos|':>6s} {'edge':>6s} {'mk60':>6s}  top actions")
    for t in active:
        for agent_id in sorted(by[(t, seeds[0])]["agents"]):
            infos = [by[(t, s)]["agents"][agent_id] for s in seeds]
            m = lambda k: _med([i.get(k) for i in infos])
            actions = {}
            for i in infos:
                for name, share in i["actions"].items():
                    actions.setdefault(name, []).append(share)
            top = sorted(((statistics.median(v), k) for k, v in actions.items()), reverse=True)[:3]
            top_text = ", ".join(f"{k} {v:.2f}" for v, k in top) if top else "(scripted)"
            fmt = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"
            lines.append(
                f"  {t:14s} {agent_id:6s} {m('return_pct'):+9.2f} {fmt(m('max_drawdown_pct')):>7s} {100 * m('maker_share'):7.0f} {fmt(m('time_long'), 2):>5s} {fmt(m('time_short'), 2):>5s} {fmt(m('mean_abs_position'), 0):>6s} {fmt(m('edge_bps')):>6s} {fmt(m('markout_60s_bps')):>6s}  {top_text}"
            )
    lines.append("")
    lines.append("Participant-type returns % (median over seeds)")
    types = sorted({k for r in rows for k in r["group_returns"]})
    lines.append(f"  {'type':20s}" + "".join(f"{t:>15s}" for t in treatments))
    for agent_type in types:
        vals = [_med([by[(t, s)]["group_returns"].get(agent_type) for s in seeds]) for t in treatments]
        lines.append(f"  {agent_type:20s}" + "".join(f"{v:+15.2f}" if v is not None else f"{'—':>15s}" for v in vals))
    lines.append("")
    lines.append("Market quality (median over seeds)")
    lines.append(f"  {'metric':26s}" + "".join(f"{t:>15s}" for t in treatments))
    for key in MARKET_KEYS:
        vals = [_med([by[(t, s)]["market"][key] for s in seeds]) for t in treatments]
        lines.append(f"  {key:26s}" + "".join(f"{v:15.3f}" if v is not None else f"{'—':>15s}" for v in vals))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate several agents trading together")
    parser.add_argument("marl_dir", nargs="?", default=None, help="run directory from marl.py (omit to skip the marl treatment)")
    parser.add_argument("--single", default=None, help="single-agent run directory or model.zip for single_copies")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seeds", default="31-40")
    parser.add_argument("--duration", default="4h")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--transparency", default=None, choices=["anonymous", "attributed", "full"])
    parser.add_argument("--treatments", default="marl,single_copies,scripted_mm,none")
    parser.add_argument("--collateral", type=float, default=100_000.0)
    parser.add_argument("--wake", type=float, default=5.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    marl_dir = Path(args.marl_dir) if args.marl_dir else None
    single = None
    if args.single:
        single = Path(args.single)
        if single.is_dir():
            single = single / "model.zip"
    treatments = [t for t in args.treatments.split(",") if t]
    if marl_dir is None:
        treatments = [t for t in treatments if t != "marl"]
    if single is None:
        treatments = [t for t in treatments if t != "single_copies"]
    seeds = _parse_seeds(args.seeds)
    duration = parse_duration(args.duration)
    rows = []
    for seed in seeds:
        for t in treatments:
            rows.append(run(t, seed, args.n, duration, args.scenario, args.transparency, marl_dir, single, args.collateral, args.wake))
            coalition = rows[-1].get("coalition_return_pct")
            print(f"  seed {seed} {t:14s}" + (f" coalition {coalition:+.2f}%" if coalition is not None else ""), flush=True)
    report = {"n": args.n, "seeds": seeds, "duration": duration, "transparency": args.transparency or "scenario default", "marl_dir": str(marl_dir), "single_model": str(single), "rows": rows}
    text = format_report(report)
    print("\n" + text)
    out = Path(args.out) if args.out else (marl_dir / f"eval_{args.seeds}_{args.transparency or 'default'}" if marl_dir else Path("artifacts/glass") / f"multi_eval_{args.seeds}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(text + "\n")
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
