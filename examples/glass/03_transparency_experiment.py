"""Paired experiment: the same markets under three transparency levels.

    PYTHONPATH=src python examples/glass/03_transparency_experiment.py

For each seed, the exogenous world (index path, news, private signals) is
identical across the three treatments, because exogenous processes use their
own random streams. Differences in outcomes therefore come from what agents
can see, not from luck.
"""

import statistics
import sys

from glass_sim import PerpetualMarket, Transparency, parse_duration, scenarios

SEEDS = range(1, int(sys.argv[1]) + 1) if len(sys.argv) > 1 else range(1, 6)
LEVELS = (Transparency.ANONYMOUS, Transparency.ATTRIBUTED, Transparency.FULL)
METRICS = ("median_spread_bps", "last_index_mean_abs_bps", "return_vol_1m_bps", "liquidations", "volume_contracts")
GROUPS = ("market_maker", "retail", "informed", "whale", "flow_follower", "copy_trader", "liquidation_hunter")

rows = {level: [] for level in LEVELS}
for seed in SEEDS:
    for level in LEVELS:
        result = PerpetualMarket(scenarios.default(seed=seed, transparency=level, duration=parse_duration("4h"))).run()
        summary = result.summary()
        groups = {g["agent_type"]: g["return_pct"] for g in result.group_table()}
        rows[level].append({**{m: summary[m] for m in METRICS}, **{f"ret_{g}": groups.get(g, 0.0) for g in GROUPS}})
        print(f"seed {seed} {level.value:10s} done", file=sys.stderr)

keys = list(rows[LEVELS[0]][0].keys())
print(f"\nmedians over {len(list(SEEDS))} seeds")
print(f"{'metric':28s}" + "".join(f"{level.value:>14s}" for level in LEVELS))
for key in keys:
    values = [statistics.median(r[key] for r in rows[level]) for level in LEVELS]
    print(f"{key:28s}" + "".join(f"{v:14.3f}" for v in values))
