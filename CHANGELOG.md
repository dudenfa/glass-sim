# Changelog

## 0.2.0 (2026-10-12)

- Fix: saving a run that contains an agent added by hand failed without the
  RL extras installed (the replay code imported the RL policy module, which
  needs Gymnasium); such runs are now replayable with the core install, and
  a replay that does contain an RL agent explains that it needs the extras.
- Docs: the user guide states that the retail group loses money by design
  (typically 15–30% of its capital per 4-hour session); the docstring of
  `AccountView.max_additional_notional` says what it returns (margin
  headroom, not notional). CI also runs on Python 3.14.
- Docs: the evaluation protocol discloses that scripted agents (the control
  market maker included) react to public news through the `on_news` hook
  while a learner acts only at its next decision, a timing advantage of a
  few seconds for the control around each news item.
- Docs: the liquidation hunter is described as what it is, an economic rule
  that fires rarely in calm markets and several times in the `volatile`
  scenario, rather than an always-active strategy.
- Every scenario now blinds agents to names: other accounts appear under
  wallet-like hashed ids (`pseudonymous_ids=True`), as on a real transparent
  exchange, so what an agent knows about another participant is only what
  the transparency level reveals. Reports and the viewer still show the real
  ids. Example 2 and tutorial 2 find "the crowd" from equity and leverage
  instead of from a name prefix. Runs differ slightly from earlier versions
  where agents broke ties by public id.
- Fix: `PerpetualMarket.reset()` on a market built with ready-made `agents`
  re-used the same agent objects, whose internal state carried into the new
  session; it now raises and asks for a new market. A reset with a new seed
  also updates the configuration, so results and replays report that seed.
- Fix: when several accounts were liquidated in the same pass of a cascade,
  all of them were priced at the mark from before the pass; each forced
  close, its penalty and any auto-deleveraging now use the mark as moved by
  the previous liquidation, and an account that the previous liquidation
  pushed back above water is no longer liquidated. Runs with simultaneous
  liquidations are therefore not bit-identical to those of earlier versions.
- Fix: scenario functions crashed when an override named a field the
  scenario sets itself (`calm(news=...)`, `anonymous(name="x")`,
  `retail_frenzy(agents=...)`); overrides now always win.
- Viewer: a new session's scenario, transparency level and seed are chosen
  in a dialog whose fields start empty, before the session starts; the
  running settings are shown read-only in the control bar, and the dialog
  opens by itself when a session ends. A restart button rewatches the same
  session from the start. Fix: a `reset` request that left out the transparency level rebuilt
  the market at the scenario's default level instead of the running one
  (for example after `glass serve --transparency anonymous`); missing fields
  now keep their current value and bad values are rejected before anything
  is rebuilt.
- Fix: in the multi-agent trainer (`glass_sim.rl.marl`), a learner's last
  transition of an episode cut by the time limit bootstrapped from the first
  decision of the next episode on the same worker, and its advantage chained
  into that episode. Such transitions are now marked truncated, valued from
  the state the episode actually ended in, and never joined to the next one.
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
- Exact replays: `result.save()` writes `replay.json` (the configuration,
  agents added by hand, every outside call such as `act()`, `execute()` and
  `shock_index()`, and a fingerprint of the final state). `Replay.load(...)`
  rebuilds the run event for event and verifies it; `glass serve --replay`
  plays it back in the viewer and `glass replay` re-runs and checks it.
  Loading never builds objects outside GLASS other than `Agent` subclasses.
- Viewer: click the timeline to jump to any moment (live sessions and replays).
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
