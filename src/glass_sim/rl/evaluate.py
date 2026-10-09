"""Evaluate a trained agent on held-out seeds, next to baselines.

    python -m glass_sim.rl.evaluate checkpoints/ppo_full_v1 --seeds 31-40

For every seed the same market (same scenario, seed and duration, hence the
same exogenous world) is run once per *treatment*:

- ``model``: the trained policy (deterministic);
- ``random``: a uniformly random policy over valid actions;
- ``hold``: a policy that never trades;
- ``none``: the market without the extra participant (market-quality reference);
- ``scripted_mm``: the fair control: a scripted ``MarketMaker`` with the same
  capital and the same mean decision interval as the learner;
- ``scripted_mm_fast``: the same scripted market maker at its native speed
  (decision every ~1.5 s), i.e. with a speed advantage over the learner;
- ``blind``: the trained policy without its private signals (signal inputs are
  zero). Only meaningful for agents trained with private signals: the gap
  between ``model`` and ``blind`` measures how much the agent uses them.

All treatments with a learner use the same capital, wake rate, observation
and action spaces. Results are reported per seed and as medians, with paired
model-minus-baseline differences, and written to ``<run>/eval_<seeds>/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from .. import scenarios
from ..agents import MarketMaker
from ..config import parse_duration
from ..market import PerpetualMarket
from ..types import Transparency
from .policy import HoldPolicy, ModelPolicy, PolicyAgent, RandomPolicy, encoder_and_actions_from_config, load_run_config

LEARNER_ID = "rl_01"
MARKET_KEYS = ("median_spread_bps", "two_sided_book_fraction", "last_index_mean_abs_bps", "return_vol_1m_bps", "volume_contracts", "liquidations")


def _parse_seeds(text: str) -> list[int]:
    """``"31-40"`` or ``"1,3,5"`` to a list of seeds."""
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        elif part.strip():
            seeds.append(int(part))
    return seeds


def run_treatment(
    run_config: dict[str, Any], treatment: str, seed: int, duration: float, model: Optional[ModelPolicy] = None, transparency: Optional[str] = None
) -> dict[str, Any]:
    """Run one treatment on one seed and collect learner, participant-type and market metrics."""
    env = run_config["env"]
    config = scenarios.get(env.get("scenario", "default"), seed=seed, duration=duration)
    level = transparency or env.get("transparency")
    if level:
        config = replace(config, transparency=Transparency(level))
    if env.get("news_rate_per_hour") is not None:
        config = replace(config, news=replace(config.news, rate_per_hour=env["news_rate_per_hour"]))
    agent = None
    if treatment in ("scripted_mm", "scripted_mm_fast"):
        agent = MarketMaker(wake_interval=float(env.get("wake_interval", 5.0)) if treatment == "scripted_mm" else 1.5)
        agent.agent_type = "control_mm"
        agent.agent_id = LEARNER_ID
    elif treatment != "none":
        encoder, actions = encoder_and_actions_from_config(run_config)
        policy = {"model": model, "blind": model, "random": RandomPolicy(), "hold": HoldPolicy()}[treatment]
        agent = PolicyAgent(policy, wake_interval=float(env.get("wake_interval", 5.0)), encoder=encoder, actions=actions)
        agent.agent_id = LEARNER_ID
    market = PerpetualMarket(config)
    if agent is not None:
        market.add_agent(agent, float(env.get("collateral", 100_000.0)), private_signals=bool(env.get("private_signals")) and treatment != "blind")
    result = market.run()
    summary = result.summary()
    groups = {g["agent_type"]: g for g in result.group_table()}
    row: dict[str, Any] = {
        "seed": seed,
        "treatment": treatment,
        "market": {key: summary.get(key) for key in MARKET_KEYS},
        "group_returns": {k: round(v["return_pct"], 4) for k, v in groups.items()},
    }
    if agent is not None:
        snap = next(a for a in result.accounts if a.agent_id == LEARNER_ID)
        row["learner"] = {
            "return_pct": snap.return_pct,
            "total_pnl": snap.total_pnl,
            "realized_pnl": snap.realized_pnl,
            "fees_paid": snap.fees_paid,
            "funding_paid": snap.funding_paid,
            "max_drawdown_pct": snap.max_drawdown_pct,
            "max_abs_position": snap.max_abs_position,
            "max_leverage": snap.max_leverage,
            "liquidations": snap.liquidations,
            "trade_count": snap.trade_count,
            "maker_share": snap.maker_share,
            "final_position": snap.position,
            "active": snap.active,
            "actions": agent.action_distribution() if hasattr(agent, "action_distribution") else {},
        }
        row["learner"].update(behaviour_metrics(result, LEARNER_ID))
    return row


def behaviour_metrics(result, agent_id: str) -> dict[str, Any]:
    """How the agent trades: time long/short, typical position, and fill quality.

    ``edge_bps``: signed distance between the fill price and the mid at the fill
    (positive = bought below / sold above the mid). ``markout_{10,60}s_bps``:
    signed mid move 10 s / 60 s after the fill relative to the fill price
    (positive = the trade was profitable at that horizon; a drop from edge to
    markout measures adverse selection).
    """
    import numpy as np

    positions = np.asarray(result.recorder.account_position.get(agent_id, [0]), dtype=float)
    mids = np.asarray([np.nan if m is None else m for m in result.recorder.bars["mid"]], dtype=float)
    edge, m10, m60 = [], [], []
    for trade in result.recorder.trades:
        if agent_id not in (trade.buyer, trade.seller):
            continue
        sign = 1.0 if trade.buyer == agent_id else -1.0
        i = min(int(trade.time), len(mids) - 1)
        if np.isnan(mids[i]):
            continue
        edge.append(sign * (mids[i] - trade.price) / mids[i] * 1e4)
        for horizon, store in ((10, m10), (60, m60)):
            j = min(i + horizon, len(mids) - 1)
            if not np.isnan(mids[j]):
                store.append(sign * (mids[j] - trade.price) / trade.price * 1e4)
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {
        "time_long": float((positions > 0).mean()),
        "time_short": float((positions < 0).mean()),
        "time_flat": float((positions == 0).mean()),
        "mean_abs_position": float(np.abs(positions).mean()),
        "edge_bps": mean(edge),
        "markout_10s_bps": mean(m10),
        "markout_60s_bps": mean(m60),
    }


def _median(values: Sequence[float]) -> Optional[float]:
    """Median ignoring ``None``; ``None`` when nothing is left."""
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def evaluate(
    run_dir: str,
    seeds: Sequence[int],
    duration: float = 4 * 3600.0,
    treatments: Sequence[str] = ("model", "random", "hold", "none"),
    transparency: Optional[str] = None,
    model_file: str = "model.zip",
    progress: bool = True,
) -> dict[str, Any]:
    """Run every treatment on every seed; a shared seed means a shared exogenous world across treatments."""
    run_path = Path(run_dir)
    model_path = run_path / model_file
    run_config = load_run_config(model_path)
    model = ModelPolicy(model_path) if ("model" in treatments or "blind" in treatments) else None
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for treatment in treatments:
            rows.append(run_treatment(run_config, treatment, seed, duration, model=model, transparency=transparency))
            if progress:
                learner = rows[-1].get("learner")
                extra = f" return {learner['return_pct']:+.2f}%" if learner else ""
                print(f"  seed {seed} {treatment:6s}{extra}", flush=True)
    return {
        "run": str(run_path),
        "model": model_file,
        "seeds": list(seeds),
        "duration": duration,
        "transparency": transparency or run_config["env"].get("transparency") or "scenario default",
        "rows": rows,
    }


def format_evaluation(report: dict[str, Any]) -> str:
    """Text report of an ``evaluate`` result."""
    rows = report["rows"]
    treatments = list(dict.fromkeys(r["treatment"] for r in rows))
    learner_treatments = [t for t in treatments if t != "none"]
    by = {(r["treatment"], r["seed"]): r for r in rows}
    seeds = report["seeds"]
    lines = [
        f"Evaluation of {report['run']} ({report['model']})",
        f"seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}), duration {report['duration'] / 3600:.1f}h, transparency {report['transparency']}",
        "",
        "Learner (median over seeds)",
        f"  {'treatment':18s} {'return %':>9s} {'pnl':>10s} {'fees':>9s} {'funding':>9s} {'maxDD %':>8s} {'max|pos|':>9s} {'maxLev':>7s} {'liqs':>5s} {'fills':>7s} {'maker %':>8s}",
    ]
    for t in learner_treatments:
        ls = [by[(t, s)]["learner"] for s in seeds]
        med = lambda k, ls=ls: _median([row[k] for row in ls])
        lines.append(
            f"  {t:18s} {med('return_pct'):+9.2f} {med('total_pnl'):+10,.0f} {med('fees_paid'):9,.0f} {med('funding_paid'):+9,.0f} {med('max_drawdown_pct'):8.1f} "
            f"{med('max_abs_position'):9,.0f} {med('max_leverage'):7.1f} {med('liquidations'):5.0f} {med('trade_count'):7,.0f} {100 * med('maker_share'):8.0f}"
        )
    if "model" in learner_treatments:
        lines.append("")
        lines.append("Learner behaviour (median over seeds)")
        lines.append(f"  {'treatment':18s} {'long':>6s} {'short':>6s} {'|pos|':>8s} {'edge':>7s} {'mk10s':>7s} {'mk60s':>7s}")
        for t in learner_treatments:
            ls = [by[(t, s)]["learner"] for s in seeds]
            m = lambda k, ls=ls: _median([row.get(k) for row in ls])
            fmt = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
            lines.append(
                f"  {t:18s} {fmt(m('time_long')):>6s} {fmt(m('time_short')):>6s} {fmt(m('mean_abs_position'), 0):>8s} {fmt(m('edge_bps'), 1):>7s} {fmt(m('markout_10s_bps'), 1):>7s} {fmt(m('markout_60s_bps'), 1):>7s}"
            )
        lines.append("")
        lines.append("Model minus baseline, paired by seed (learner return, percentage points)")
        for base in [t for t in learner_treatments if t != "model"]:
            diffs = [by[("model", s)]["learner"]["return_pct"] - by[(base, s)]["learner"]["return_pct"] for s in seeds]
            wins = sum(1 for d in diffs if d > 0)
            lines.append(
                f"  vs {base:8s} median {statistics.median(diffs):+7.2f}  mean {statistics.mean(diffs):+7.2f}  model better in {wins}/{len(diffs)} seeds"
            )
        lines.append("")
        lines.append("Model return per seed: " + "  ".join(f"{s}:{by[('model', s)]['learner']['return_pct']:+.2f}%" for s in seeds))
        lines.append("")
        lines.append("Model action mix (median share over seeds)")
        names = list(by[("model", seeds[0])]["learner"]["actions"].keys())
        lines.append("  " + "  ".join(f"{n}={_median([by[('model', s)]['learner']['actions'][n] for s in seeds]):.3f}" for n in names))
    lines.append("")
    lines.append("Participant-type returns, % (median over seeds)")
    types = sorted({k for r in rows for k in r["group_returns"]})
    lines.append(f"  {'type':20s}" + "".join(f"{t:>10s}" for t in treatments))
    for agent_type in types:
        vals = [_median([by[(t, s)]["group_returns"].get(agent_type) for s in seeds]) for t in treatments]
        lines.append(f"  {agent_type:20s}" + "".join(f"{v:+10.2f}" if v is not None else f"{'—':>10s}" for v in vals))
    lines.append("")
    lines.append("Market quality (median over seeds)")
    lines.append(f"  {'metric':26s}" + "".join(f"{t:>12s}" for t in treatments))
    for key in MARKET_KEYS:
        vals = [_median([by[(t, s)]["market"][key] for s in seeds]) for t in treatments]
        lines.append(f"  {key:26s}" + "".join(f"{v:12.3f}" if v is not None else f"{'—':>12s}" for v in vals))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point (``glass evaluate``); writes the report next to the checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate a trained futures-market agent on held-out seeds")
    parser.add_argument("run_dir", help="directory written by glass_sim.rl.train")
    parser.add_argument("--seeds", default="31-40")
    parser.add_argument("--duration", default="4h")
    parser.add_argument("--treatments", default="model,scripted_mm,scripted_mm_fast,random,hold,none")
    parser.add_argument("--transparency", default=None, choices=["anonymous", "attributed", "full"])
    parser.add_argument("--model-file", default="model.zip")
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    report = evaluate(args.run_dir, seeds, parse_duration(args.duration), args.treatments.split(","), args.transparency, args.model_file)
    text = format_evaluation(report)
    print("\n" + text)
    tag = f"eval_{args.seeds.replace(',', '_')}_{args.transparency or 'default'}_{Path(args.model_file).stem}"
    out = Path(args.run_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(text + "\n")
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
