# Changelog

## Unreleased

- Pluggable RL components: `PerpTradingEnv(encoder=..., actions=..., reward=...)`
  accepts custom observation encoders, action menus and reward trackers
  (interfaces in `glass_sim/rl/components.py`); `make_env` forwards them.
- Ladder action menu (`ActionConfig(kind="ladder")`, `--actions ladder`):
  market orders and post-only quotes at several price offsets and sizes,
  optionally several resting quotes per side. Checkpoints record the menu.
- PettingZoo environment (`glass_sim.rl.pettingzoo_env.GlassAECEnv`): several
  learners in one market through the AEC API, with action masks, liquidation
  terminations and pluggable components; passes PettingZoo's API and seed
  tests. PettingZoo joins the `rl` extra. New example `05_pettingzoo.py`.
- `MultiAgentMarket` accepts custom encoder, action menu and reward.
- Tutorial notebooks (`notebooks/`): a first market, agents and paired
  experiments, reinforcement learning; saved with outputs. New `notebooks`
  install extra.

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

The earlier spot-market thesis code (V1) is kept, unmaintained, in `legacy/`.
