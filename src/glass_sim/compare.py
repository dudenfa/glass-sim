"""Compare runs: two single runs side by side, or two setups over a panel of seeds.

A *run spec* is a string of ``key=value`` pairs, for example
``"scenario=default seed=1 transparency=anonymous duration=1h"``, or the path
of a directory written by ``SimulationResult.save`` (``glass run --out``).

Single comparison: ``delta = right - left`` and ``%`` relative to ``|left|``.

Panel comparison (``seeds``): both setups are run on the same seeds. Because
the exogenous world is identical for a shared seed, each seed gives a paired
difference. The report shows the median of each side, the median and mean of
the paired differences, and in how many seeds the right side was higher.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from . import scenarios
from .config import MarketConfig, parse_duration
from .market import PerpetualMarket
from .report import SUMMARY_SECTIONS
from .results import SavedRun, SimulationResult
from .types import Transparency

RunLike = Union[SimulationResult, SavedRun]


# ----------------------------------------------------------------- specs


def parse_spec(spec: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for token in spec.split():
        if "=" not in token:
            raise ValueError(f"expected key=value in run spec, got {token!r}")
        key, value = token.split("=", 1)
        values[key.strip()] = value.strip()
    unknown = set(values) - {"scenario", "seed", "transparency", "duration"}
    if unknown:
        raise ValueError(f"unknown run-spec keys: {', '.join(sorted(unknown))}")
    return values


def config_from_spec(spec: str, seed: Optional[int] = None) -> MarketConfig:
    values = parse_spec(spec)
    overrides: Dict[str, Any] = {"seed": int(seed if seed is not None else values.get("seed", 1))}
    if "duration" in values:
        overrides["duration"] = parse_duration(values["duration"])
    config = scenarios.get(values.get("scenario", "default"), **overrides)
    if "transparency" in values:
        config = replace(config, transparency=Transparency(values["transparency"]))
    return config


def load_or_run(spec: str) -> RunLike:
    path = Path(spec)
    if path.is_dir():
        return SavedRun.load(path)
    return PerpetualMarket(config_from_spec(spec)).run()


def _parts(run: RunLike) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(run, SavedRun):
        return run.summary, run.agents, run.groups
    return run.summary(), run.agent_table(), run.group_table()


def _label(run: RunLike) -> str:
    summary = _parts(run)[0]
    return f"{summary.get('scenario')} seed={summary.get('seed')} {summary.get('transparency')}"


# ----------------------------------------------------------- single run


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def _delta_row(label: str, left: Any, right: Any, width: int = 32) -> str:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        delta = right - left
        pct = f"{100.0 * delta / abs(left):+.1f}%" if left else "—"
        return f"  {label:{width}s} {_fmt(left):>14s} {_fmt(right):>14s} {('+' if delta > 0 else '') + _fmt(delta):>14s} {pct:>9s}"
    return f"  {label:{width}s} {_fmt(left):>14s} {_fmt(right):>14s}"


def compare_runs(left: RunLike, right: RunLike, show_agents: bool = False, top_agents: int = 15) -> str:
    ls, la, lg = _parts(left)
    rs, ra, rg = _parts(right)
    header = f"  {'':32s} {'left':>14s} {'right':>14s} {'delta':>14s} {'%':>9s}"
    lines = [f"left:  {_label(left)}", f"right: {_label(right)}", ""]
    for title, fields in SUMMARY_SECTIONS:
        lines.append(title)
        lines.append(header)
        for key, label, _ in fields:
            lines.append(_delta_row(label, ls.get(key), rs.get(key)))
        lines.append("")

    lines.append("Return on capital by participant type (%)")
    lines.append(header)
    lgroups = {g["agent_type"]: g for g in lg}
    rgroups = {g["agent_type"]: g for g in rg}
    for agent_type in sorted(set(lgroups) | set(rgroups)):
        lines.append(_delta_row(agent_type, lgroups.get(agent_type, {}).get("return_pct"), rgroups.get(agent_type, {}).get("return_pct")))
    lines.append("")

    lagents = {a["agent_id"]: a for a in la}
    ragents = {a["agent_id"]: a for a in ra}
    common = [agent_id for agent_id in lagents if agent_id in ragents]
    common.sort(key=lambda i: -abs(ragents[i]["total_pnl"] - lagents[i]["total_pnl"]))
    shown = common if show_agents else common[:top_agents]
    title = "Every agent present in both runs" if show_agents else f"Agents with the largest PnL change (top {len(shown)} of {len(common)})"
    lines.append(title)
    lines.append(f"  {'agent':16s} {'type':18s} {'pnl left':>12s} {'pnl right':>12s} {'delta':>12s} {'ret L%':>8s} {'ret R%':>8s} {'maxDD L':>8s} {'maxDD R':>8s}")
    for agent_id in shown:
        a, b = lagents[agent_id], ragents[agent_id]
        lines.append(
            f"  {agent_id:16s} {a['agent_type']:18s} {a['total_pnl']:+12,.0f} {b['total_pnl']:+12,.0f} {b['total_pnl'] - a['total_pnl']:+12,.0f} "
            f"{a['return_pct']:+8.2f} {b['return_pct']:+8.2f} {a.get('max_drawdown_pct', 0):8.1f} {b.get('max_drawdown_pct', 0):8.1f}"
        )
    only_left = sorted(set(lagents) - set(ragents))
    only_right = sorted(set(ragents) - set(lagents))
    if only_left or only_right:
        lines.append("")
        if only_left:
            lines.append(f"  only in left:  {', '.join(only_left)}")
        if only_right:
            lines.append(f"  only in right: {', '.join(only_right)}")
    return "\n".join(lines)


# ------------------------------------------------------------ seed panel

PANEL_METRICS = [key for _, fields in SUMMARY_SECTIONS for key, _, _ in fields if key not in ("final_last", "final_mark", "final_index", "agents", "events_processed")]


def parse_seeds(text: str) -> List[int]:
    """``"1-10"``, ``"1,3,5"`` or ``"1-5,20"``."""
    seeds: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        elif part.strip():
            seeds.append(int(part))
    return seeds


def run_panel(spec: str, seeds: Sequence[int], progress: Optional[Callable[[str], None]] = None) -> List[SimulationResult]:
    results = []
    for seed in seeds:
        results.append(PerpetualMarket(config_from_spec(spec, seed=seed)).run())
        if progress:
            progress(f"{spec} seed={seed}")
    return results


def _panel_rows(results: Sequence[SimulationResult]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        summary = result.summary()
        row = {key: summary.get(key) for key in PANEL_METRICS}
        for group in result.group_table():
            row[f"return_pct:{group['agent_type']}"] = group["return_pct"]
        rows.append(row)
    return rows


def compare_panels(left_spec: str, right_spec: str, seeds: Sequence[int], progress: Optional[Callable[[str], None]] = None) -> str:
    left = _panel_rows(run_panel(left_spec, seeds, progress))
    right = _panel_rows(run_panel(right_spec, seeds, progress))
    keys = list(dict.fromkeys(list(left[0].keys()) + list(right[0].keys())))
    lines = [
        f"left:  {left_spec}",
        f"right: {right_spec}",
        f"seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]}); paired difference = right - left on the same seed",
        "",
        f"  {'metric':36s} {'median L':>12s} {'median R':>12s} {'med diff':>12s} {'mean diff':>12s} {'R > L':>7s}",
    ]
    for key in keys:
        pairs = [(l.get(key), r.get(key)) for l, r in zip(left, right)]
        pairs = [(a, b) for a, b in pairs if isinstance(a, (int, float)) and isinstance(b, (int, float))]
        if not pairs:
            continue
        diffs = [b - a for a, b in pairs]
        wins = sum(1 for d in diffs if d > 0)
        label = key.replace("return_pct:", "return % ") if key.startswith("return_pct:") else key
        lines.append(
            f"  {label:36s} {_fmt(statistics.median(a for a, _ in pairs)):>12s} {_fmt(statistics.median(b for _, b in pairs)):>12s} "
            f"{_fmt(statistics.median(diffs)):>12s} {_fmt(statistics.mean(diffs)):>12s} {wins:>3d}/{len(pairs):<3d}"
        )
    return "\n".join(lines)
