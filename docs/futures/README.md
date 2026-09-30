# Transparent perpetual-futures market simulator

`marl_trading.futures` is a multi-agent simulator of a **perpetual futures
exchange**: a public limit order book where agents trade long and short with
leverage, post margin, pay and receive funding, and get liquidated. The level
of **transparency** (what every agent can see about the others) is an
experimental setting, from a classic anonymous exchange to a fully transparent
on-chain venue in the style of Hyperliquid.

Prices, liquidity, volatility and liquidation cascades are not replayed from
data. They emerge from the agents' interaction.

- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Writing an agent](#writing-an-agent)
- [The scripted ecology](#the-scripted-ecology)
- [Configuration](#configuration)
- [Results and analysis](#results-and-analysis)
- [Live viewer](#live-viewer)
- [Running experiments](#running-experiments)
- [Driving agents from outside (RL)](#driving-agents-from-outside-rl)
- [Validation](#validation)

## Quick start

```bash
pip install -e ".[dev]"          # numpy is the only runtime dependency; pandas is optional
python -m marl_trading.futures run --scenario default --seed 1
python -m marl_trading.futures serve --open
```

```python
from marl_trading.futures import PerpetualMarket, scenarios

result = PerpetualMarket(scenarios.default(seed=1)).run()
result.print_report()
frames = result.to_frames()      # pandas DataFrames: bars, trades, liquidations, funding, equity, ...
```

The reference scenario simulates 4 hours with 88 agents (about 130,000
events) in roughly 5 seconds on a laptop.

Runnable examples live in [`examples/futures/`](../../examples/futures).

## Concepts

### Time and events

The simulator is a discrete-event simulation in continuous time (seconds).
Every agent wakes up on its own random clock (a Poisson process by default).
It looks at the market and sends actions. Actions reach the exchange after a
small random latency, so the market may have moved in between. There are no
"turns": which agent acts first is decided by the clocks, not by a fixed order.

At equal timestamps events are processed in a fixed priority (news, index
update, private signals, funding, order arrivals, agent wakeups, recording) and
then by insertion order, so a run is exactly reproducible from its seed.

### Randomness and paired experiments

The index path, the news and the private signals use their own random streams.
Each agent has its own stream derived from the seed and its id. Two consequences:

- two runs with the same seed face **exactly the same exogenous world**, even
  if the agents or the transparency level differ, so treatments can be compared
  seed by seed;
- adding or removing one agent does not change any other agent's random choices.

### Instrument

A linear perpetual future (`ContractSpec`): tick size 0.01, contract size 1,
initial price 100, maximum leverage 40x, maker fee 1.5 bps, taker fee 4.5 bps.

### Orders

| action | meaning |
|---|---|
| `LimitOrder(side, qty, price, tif=GTC)` | rests in the book until filled or canceled |
| `LimitOrder(..., tif=IOC)` | trades what it can immediately, the rest is canceled |
| `LimitOrder(..., tif=POST_ONLY)` | rejected if it would trade immediately |
| `MarketOrder(side, qty)` | takes liquidity, at most 5% away from the mark price |
| `reduce_only=True` | only reduces the position (market and IOC orders) |
| `Cancel(order_id)` / `CancelAll(side=None)` | cancel resting orders |

Matching is price-time priority. Self-trade prevention cancels the resting
order instead of trading an agent against itself.

### Margin

Each account has collateral (cash), a signed position and an average entry price.

- `equity = collateral + unrealized PnL at the mark price`
- initial margin uses the **worst-case position** if all resting orders on one
  side filled: `IM = max(|pos + open buys|, |pos - open sells|) * mark / max_leverage`.
  An order that would push IM above equity is rejected. Orders that reduce risk
  are always accepted.
- maintenance margin: `MM = |pos| * mark * mmr`, with `mmr = 0.5 / max_leverage`
  (the Hyperliquid convention).

### Liquidation

When equity falls below maintenance margin:

1. all the account's orders are canceled;
2. the position is closed with a market order (up to 10% from the mark);
3. whatever the book cannot absorb is **auto-deleveraged** against the most
   profitable, most levered opposite positions at the mark price;
4. a penalty (1% of closed notional, capped at remaining equity) goes to the
   **insurance fund**; negative equity (bad debt) is paid by the fund;
5. an account left with less than 5% of its initial collateral leaves the market.

Liquidation fills move the price, which can trigger further liquidations, so
cascades emerge naturally.

### Index, mark and funding

- The **index** is the external spot reference. Its log price follows a
  diffusion with stochastic volatility (volatility clustering). News items
  (a Poisson process) make it jump.
- The **mark price** is the median of (book mid, last trade, index + 150 s
  moving-average basis), which is robust to manipulation. It is used for margin,
  PnL and liquidations.
- **Funding**: the premium `(mark - index) / index` is sampled every 5 s.
  Every hour, `rate = clip(mean premium + interest, ±cap)` and each account pays
  `position * index * rate`: longs pay shorts when the perp trades above the index.

### Information and transparency

| information | ANONYMOUS | ATTRIBUTED | FULL |
|---|:-:|:-:|:-:|
| aggregated book, trade prices and sizes | ✓ | ✓ | ✓ |
| mark, index, funding, open interest | ✓ | ✓ | ✓ |
| owner of every resting order | | ✓ | ✓ |
| buyer and seller of every trade | | ✓ | ✓ |
| every account's position, entry, leverage, PnL, liquidation price | | | ✓ |

Agents always see their own account and orders. Asking for hidden information
raises `TransparencyError`; `view.can_see(level)` lets an agent adapt.

With `pseudonymous_ids=True`, public ids are wallet-like hashes, so agents
cannot read another agent's type from its name.

**Private information** is separate from transparency. Agents created with
`AgentSpec(private_signals=True)` receive, for each news item, a noisy estimate
of its size 20 to 90 seconds before it becomes public.

## Writing an agent

```python
from marl_trading.futures import Agent, MarketOrder, Side

class Contrarian(Agent):
    agent_type = "contrarian"
    default_prefix = "contra"

    def __init__(self, wake_interval=15.0, size=5):
        super().__init__(wake_interval=wake_interval)
        self.size = size

    def on_wakeup(self, view, rng):
        prices = view.price_history(60)            # last 60 seconds, one sample per second
        if len(prices) < 60:
            return []
        side = Side.SELL if prices[-1] > prices[0] else Side.BUY
        return [MarketOrder(side, self.size)]
```

Add it to any scenario:

```python
from dataclasses import replace
from marl_trading.futures import AgentSpec, PerpetualMarket, scenarios

base = scenarios.default(seed=1)
config = replace(base, agents=base.agents + (AgentSpec(Contrarian, count=5, collateral=(5_000, 50_000)),))
result = PerpetualMarket(config).run()
```

Hooks an agent can implement:

| hook | when |
|---|---|
| `on_start(view, rng)` | once at time 0 |
| `on_wakeup(view, rng)` | at each wakeup: the main decision point |
| `on_fill(fill, view)` | when one of its orders trades (bookkeeping only) |
| `on_news(news, view, rng)` | when public news is released |
| `on_signal(signal, view, rng)` | when a private signal arrives (subscribed agents only) |
| `on_liquidation(view)` | after the agent was liquidated |
| `next_wakeup_delay(rng)` | override for a non-Poisson clock |

Always use the `rng` argument for randomness, never the global generators.

`AgentSpec` builds groups of agents: `count`, `collateral` (fixed or a
`(low, high)` range sampled log-uniformly), constructor `params`, and
`randomize=("param", ...)` to sample listed `(low, high)` params per agent.

## The scripted ecology

`scenarios.default()` contains 88 agents of nine types:

| type | count | role |
|---|---:|---|
| `MarketMaker` | 3 | multi-level two-sided quotes around a blend of mid and index; spread widens with index volatility and after news; skews to shed inventory |
| `NoiseTrader` (retail) | 60 | random direction with a mild trend-chasing bias, leverage 3–40x, random holding time, take-profit, no stop-loss |
| `MomentumTrader` | 8 | position proportional to a fast/slow EMA crossover of the trade price |
| `BasisArbitrageur` | 4 | sells the perp above the index and buys below it with price-limited orders; anchors the perp to the index |
| `InformedTrader` | 2 | trades ahead of news on private signals |
| `Whale` | 1 | multi-hour accumulation and unwind campaigns executed with TWAP child orders |
| `FlowFollower` | 3 | ATTRIBUTED or FULL: follows the account with the strongest one-directional flow on the attributed trade tape (wallet watching) |
| `CopyTrader` | 6 | FULL only: mirrors the most profitable visible account taking leveraged directional risk |
| `LiquidationHunter` | 1 | FULL only: pushes the price into visible liquidation clusters when the forced flow outweighs the cost of the push |

Other scenarios change one thing at a time: `anonymous`, `attributed`, `calm`,
`volatile`, `retail_frenzy`, `no_arbitrage`, plus `small` for fast tests.
List them with `python -m marl_trading.futures scenarios`.

## Configuration

`MarketConfig` groups everything:

| field | type | purpose |
|---|---|---|
| `contract` | `ContractSpec` | tick, contract size, initial price, max leverage, mmr, fees |
| `index` | `IndexConfig` | daily volatility, drift, update interval, volatility clustering |
| `news` | `NewsConfig` | news rate, jump size, private-signal lead time and noise |
| `funding` | `FundingConfig` | interval, sampling, interest rate, cap |
| `liquidation` | `LiquidationConfig` | penalty, insurance fund, exit threshold |
| `latency` | `LatencyConfig` | order latency base and jitter |
| `transparency` | `Transparency` | ANONYMOUS, ATTRIBUTED or FULL |
| `agents` | `tuple[AgentSpec]` | the population |
| `duration`, `seed`, `record_interval` | | run length, seed, sampling of time series |
| `pseudonymous_ids` | `bool` | hash public ids |

All config objects are frozen dataclasses: derive variants with
`dataclasses.replace`.

## Results and analysis

`PerpetualMarket.run()` returns a `SimulationResult`:

- `summary()`: liquidity (two-sided book fraction, spread, depth), price
  discovery (mark/last vs index), volatility, open interest, liquidations, bad
  debt, funding, fees, and `conservation_error` (see [Validation](#validation));
- `stylized_facts()`: excess kurtosis and autocorrelation of returns and of
  absolute returns at 10 s, 60 s and 300 s;
- `agent_table()` / `group_table()`: per account and per agent type: PnL
  (realized, unrealized), return, fees, funding, path risk (peak and worst
  equity, max drawdown, largest position, highest leverage), maker share,
  order counts and liquidations;
- `report(agents=True)` / `print_report(agents=True)`: the text report with
  every agent;
- `to_frames()`: pandas DataFrames (`bars`, `trades`, `liquidations`,
  `funding`, `news`, `equity`, `positions`, and `orders` if recorded);
- `save(directory)`: `report.txt` (full text report), JSON summaries and CSV
  time series. `SavedRun.load(directory)` reads a saved run back.

Every field is explained in [METRICS.md](METRICS.md).

From the command line:

```bash
perpsim run --scenario volatile --seed 3 --duration 2h --agents --out artifacts/futures/volatile_s3
```

### Comparing runs

```bash
# two single runs (run specs or saved directories)
perpsim compare "scenario=default seed=1" "scenario=default seed=1 transparency=anonymous"
perpsim compare artifacts/futures/run_a artifacts/futures/run_b --agents

# two setups over a paired seed panel: medians, paired differences, win counts
perpsim compare "scenario=default" "scenario=anonymous" --seeds 1-10 --out comparison.txt
```

In Python: `marl_trading.futures.compare.compare_runs(left, right)` and
`compare_panels(left_spec, right_spec, seeds)`.

## Live viewer

```bash
python -m marl_trading.futures serve --scenario default --seed 1 --speed 30 --open
```

Opens http://127.0.0.1:8766 with:

- candlestick chart (1 s to 15 min) with mark and index lines, news and
  liquidation markers, volume and open interest; scroll to zoom, drag to pan;
- the order book, with each level's owners shown as colored dots (hover for
  details), and the trade tape with buyer and seller;
- tabs: every account's position and liquidation price, results by participant
  type, liquidations, funding history, news, order flow, live market-quality
  and stylized-fact statistics, and an explanation of the mechanics;
- click any account for its equity curve, position, parameters, orders and fills;
- controls: play/pause (space bar), step, run to end, speed, and a new session
  with another scenario, transparency level or seed.

The researcher always sees everything. The viewer states what the **agents**
can see at the current transparency level.

## Running experiments

Because exogenous randomness is shared across treatments, experiments are
naturally paired:

```python
from marl_trading.futures import PerpetualMarket, Transparency, scenarios

for seed in range(1, 11):
    for level in Transparency:
        result = PerpetualMarket(scenarios.default(seed=seed, transparency=level)).run()
        ...
```

See [`examples/futures/03_transparency_experiment.py`](../../examples/futures/03_transparency_experiment.py).

For controlled interventions, `market.shock_index(log_jump)` applies an
exogenous jump at the current time, and `market.execute(agent_id, actions)`
injects orders immediately. Combine them with `market.run_until(time)`.

## Driving agents from outside (RL)

`ExternalAgent` connects a learning algorithm:

```python
from marl_trading.futures import AgentSpec, ExternalAgent

spec = AgentSpec(ExternalAgent, collateral=50_000, params={"wake_interval": 5.0}, external=True, prefix="rl")
market = PerpetualMarket(replace(base, agents=base.agents + (spec,)))
while (agent_id := market.run_until_decision()) is not None:
    view = market.view(agent_id)           # observation
    market.act(agent_id, policy(view))     # action; reward from view.me.equity changes
```

Several external agents can be present. The market pauses at each one's
wakeup, which is the natural shape of a turn-based multi-agent (PettingZoo AEC)
environment. `ExternalAgent(policy=f)` runs a trained policy inline instead.

## Validation

- **Accounting conservation.** The sum of all equities, plus the insurance
  fund, plus collected fees, always equals total deposits plus the initial fund.
  Every trade and funding payment has two sides. `summary()["conservation_error"]`
  reports the discrepancy, and the tests check it at every recorded second of a run.
- **Unit tests** cover the order book (priority, partial fills, self-trade
  prevention), account arithmetic (averaging, flips, liquidation price),
  margin, IOC/post-only/reduce-only rules, liquidation with bad debt,
  auto-deleveraging, funding direction, transparency gating, reproducibility,
  paired exogenous paths and the external-agent loop: `pytest tests/test_futures_*.py`.
- **Stylized facts.** The reference ecology produces heavy-tailed returns whose
  kurtosis falls with the horizon, near-zero autocorrelation of 1-minute
  returns and positive autocorrelation of absolute returns. Calibration against
  published stylized-fact values is future work.
