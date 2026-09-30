"""Run the reference market for one simulated hour and inspect the results.

    PYTHONPATH=src python examples/futures/01_quickstart.py
"""

from marl_trading.futures import PerpetualMarket, parse_duration, scenarios

config = scenarios.default(seed=1, duration=parse_duration("1h"))
result = PerpetualMarket(config).run()

# A compact human-readable report.
result.print_report()

# Everything is also available as data.
summary = result.summary()
print("\nmedian spread (bps):", round(summary["median_spread_bps"], 2))
print("liquidations:", summary["liquidations"])

best = result.agent_table()[0]
print(f"best account: {best['agent_id']} ({best['agent_type']}) {best['return_pct']:+.2f}%")

# Stylized facts of the simulated trade price.
for horizon, stats in result.stylized_facts().items():
    print(horizon, "excess kurtosis", round(stats["excess_kurtosis"] or 0, 2), "ACF |r| lag1", round(stats["acf_abs_return_lag1"] or 0, 3))

# With pandas installed, every recorded series is a DataFrame.
frames = result.to_frames()
print(frames["bars"][["close", "mark", "index", "spread", "open_interest"]].tail())
