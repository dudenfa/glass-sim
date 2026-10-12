# Metrics reference

Every field in the futures-market outputs: the text report (`glass run`,
`result.print_report()`), `summary.json`, `agents.json`, `groups.json`, the
comparison reports (`glass compare`) and the viewer.

Units: prices in quote currency (the contract starts at 100), quantities in
contracts, money in quote currency, `bps` = basis points (0.01%).

Source of truth in code: `analysis.py` (market metrics, stylized facts),
`results.py` (summary, groups), `account.py` (per-agent fields), `report.py`
and `compare.py` (formatting). Update this file when fields change.

## Market summary (`summary.json`)

### Activity

| field | meaning |
|---|---|
| `trades` | number of trades (each maker-taker match is one trade; ADL transfers included) |
| `volume_contracts` | total traded contracts |
| `trades_per_minute` | trades divided by simulated minutes |
| `mean_open_interest`, `max_open_interest` | open interest = sum of all long positions (= sum of all shorts), sampled every second |

### Liquidity

| field | meaning |
|---|---|
| `two_sided_book_fraction` | share of seconds where both a best bid and a best ask existed |
| `median_spread_bps` | median of (ask - bid) / average mid, over two-sided seconds |
| `mean_spread` | mean ask - bid in price units |
| `mean_bid_depth`, `mean_ask_depth` | contracts resting within 50 ticks of the best price, averaged over seconds |

### Prices

| field | meaning |
|---|---|
| `return_vol_1m_bps` | standard deviation of 1-minute log returns of the last trade price |
| `last_index_mean_abs_bps` / `last_index_max_abs_bps` | mean / max absolute gap between last trade and index; how well the perp tracks its reference |
| `mark_index_mean_abs_bps` | same for the mark price (smoother by construction) |
| `final_last`, `final_mark`, `final_index` | prices at the end of the run |

### Risk and funding

| field | meaning |
|---|---|
| `liquidations` | number of liquidation events |
| `liquidated_contracts` | contracts closed by liquidations |
| `adl_contracts` | part of those closed by auto-deleveraging because the book could not absorb them |
| `bad_debt` | losses beyond liquidated accounts' collateral, paid by the insurance fund |
| `insurance_fund_final` | fund at the end (starts at 250,000; grows with penalties, shrinks with bad debt) |
| `funding_settlements`, `mean_funding_rate` | number of hourly settlements and their average rate (per interval; positive = longs paid shorts) |
| `fees_collected` | maker plus taker fees paid to the exchange |

### Population and checks

| field | meaning |
|---|---|
| `agents`, `active_agents_final` | participants, and those still in the market at the end (liquidated below 5% of initial collateral means out) |
| `orders_accepted`, `orders_rejected`, `orders_canceled` | order lifecycle counts; rejection reasons are in the order events |
| `events_processed` | discrete events simulated |
| `conservation_error` | sum of equities + insurance fund + fees - deposits - initial fund. Must be ~0 (float rounding, e.g. 1e-8); anything larger means an accounting bug |

## Liquidation events (`liquidations.csv`, `result.liquidations`, viewer tab)

One row per liquidation, in the order they happened:

| field | meaning |
|---|---|
| `time` | simulated time of the liquidation |
| `agent_id` | the liquidated account |
| `side_closed` | side of the forced order: `sell` closed a long, `buy` closed a short |
| `quantity` | contracts the account held when it was liquidated |
| `mark_price` | mark price at the moment this account was liquidated; in a cascade, later rows see the mark moved by the earlier ones |
| `avg_fill_price` | average price of the forced close against the book (empty if nothing could be filled) |
| `adl_quantity` | contracts the book could not absorb, closed by auto-deleveraging at `mark_price` |
| `bad_debt` | negative collateral left after the close, paid by the insurance fund |
| `penalty` | liquidation penalty paid into the insurance fund (1% of closed notional, capped at the remaining collateral) |
| `equity_before` | the account's equity at `mark_price` just before the close |
| `deactivated` | whether the account left the market (collateral below 5% of its initial deposit) |

## Stylized facts (`stylized_facts.json`)

For horizons 10 s, 60 s and 300 s, computed on log returns of the last trade price:

| field | meaning | real markets |
|---|---|---|
| `n_returns` | number of returns at that horizon | read statistics only with dozens or more |
| `zero_return_fraction` | share of intervals without a price change | |
| `std_bps` | return standard deviation | |
| `excess_kurtosis` | tail heaviness (0 for a normal distribution) | positive, shrinking at longer horizons |
| `acf_return_lag1` | autocorrelation of consecutive returns | about 0 (slightly negative at very short horizons from bid-ask bounce) |
| `acf_abs_return_lag{1,5,20}` | autocorrelation of absolute returns | positive and slowly decaying (volatility clustering) |

## Per agent (`agents.json`, report table, viewer)

### State at the end

| field | meaning |
|---|---|
| `agent_id`, `public_id`, `agent_type` | internal id, id shown to other agents (hashed if `pseudonymous_ids`), type |
| `active` | still in the market |
| `collateral` | cash: deposit + realized PnL - fees - funding paid |
| `position` | signed contracts (long > 0) |
| `entry_price` | average entry of the current position |
| `mark_price` | mark used for the valuation |
| `unrealized_pnl` | position × (mark - entry) × contract size |
| `realized_pnl` | PnL locked in by reducing or closing positions (before fees and funding) |
| `equity` | collateral + unrealized PnL |
| `notional` | abs(position) × mark × contract size |
| `leverage` | notional / equity |
| `liquidation_price` | mark at which equity would equal maintenance margin (empty if flat or unreachable) |
| `fees_paid`, `funding_paid` | totals; funding positive = paid, negative = received |
| `initial_collateral` | starting deposit |
| `total_pnl`, `return_pct` | equity - initial collateral, and the same as % of initial collateral |
| `open_orders` | resting orders at the end |

Identity: `total_pnl = realized_pnl + unrealized_pnl - fees_paid - funding_paid - liquidation penalties`
(plus bad debt forgiven, for liquidated accounts).

### Path risk (updated every second and at liquidation)

| field | meaning |
|---|---|
| `peak_equity`, `min_equity` | highest and lowest equity reached |
| `max_drawdown`, `max_drawdown_pct` | largest fall from a previous equity peak, in money and in % of that peak |
| `max_abs_position` | largest absolute position held |
| `max_leverage` | highest leverage reached |
| `first_liquidation_time` | simulated time of the first liquidation, if any |
| `liquidations` | number of times liquidated |

### Activity

| field | meaning |
|---|---|
| `trade_count` | fills (one per match the account took part in) |
| `volume` | traded notional in quote currency |
| `maker_quantity`, `taker_quantity`, `maker_share` | contracts traded as liquidity provider vs taker, and the maker fraction |
| `orders_accepted`, `orders_rejected`, `orders_canceled` | order lifecycle counts for this account |

## Per participant type (`groups.json`)

| field | meaning |
|---|---|
| `agents`, `inactive` | members, and members out of the market |
| `initial_collateral`, `total_pnl`, `return_pct` | summed capital and PnL; return is capital-weighted (total PnL / total capital) |
| `median_return_pct` | median of members' returns; robust to one big account |
| `realized_pnl`, `fees_paid`, `funding_paid`, `volume`, `liquidations` | sums |
| `mean_max_drawdown_pct` | average of members' max drawdown % |
| `maker_share` | maker contracts / all contracts traded by the group |
| `net_position` | sum of members' positions at the end |

## Comparisons (`glass compare`)

- **Two runs:** `left`, `right`, `delta = right - left`, `%` relative to `|left|`,
  for every summary field, return by type, and the agents whose PnL changed most
  (`--agents` lists all agents present in both runs).
- **Seed panel (`--seeds`):** both setups run on the same seeds. For each metric:
  `median L`, `median R`, `med diff` = median of the per-seed paired differences,
  `mean diff`, and `R > L` = in how many seeds the right side was higher.
  The median of paired differences is not the difference of medians; both are
  shown because they answer different questions.

Because exogenous randomness (index, news, private signals) is shared by seed,
paired differences isolate the effect of the treatment rather than luck.
