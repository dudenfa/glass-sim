# Changelog

## 0.1.0 (2026-10)

First public release of GLASS.

- Discrete-event perpetual-futures exchange: price-time priority order book,
  GTC / IOC / post-only / reduce-only orders, self-trade prevention, fees,
  per-agent Poisson clocks and order latency, exact reproducibility.
- Cross-margin accounts, liquidation with auto-deleveraging and insurance
  fund, index price with stochastic volatility and news, robust mark price,
  hourly funding; accounting conservation checked by the tests.
- Selectable transparency (anonymous, attributed, full) enforced by the
  agents' `MarketView`.
- Nine scripted trader types and eight scenarios.
- Results, text reports, path-risk metrics, stylized facts, run and paired
  seed-panel comparisons; `glass` command line; live browser viewer.
- Reinforcement learning: Gymnasium environment with masked actions and an
  exit-value reward, MaskablePPO training and held-out evaluation with fair
  controls, multi-agent IPPO trainer and evaluation.
- Experiments and figures of the accompanying paper (`experiments/`, `paper/`).

The earlier spot-market thesis code (V1) is kept, unmaintained, in `legacy/`.
