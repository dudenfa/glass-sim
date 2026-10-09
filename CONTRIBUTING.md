# Contributing to GLASS

Thanks for your interest in GLASS. Bug reports, new agents, new scenarios,
analysis tools and documentation improvements are all welcome.

## Development setup

```bash
git clone https://github.com/dudenfa/glass-sim.git
cd glass-sim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q          # run the test suite
ruff check .       # lint
```

`.[all]` installs the RL extras (PyTorch via Stable-Baselines3). For work on
the simulator only, `pip install -e ".[dev]"` is enough; RL tests are then
skipped automatically.

## Guidelines

- **Reproducibility first.** Agents must draw randomness only from the `rng`
  argument they receive; never use the global `random` / `numpy.random`.
  Runs with the same seed must stay identical.
- **Accounting must be conserved.** Any change to the exchange, margin,
  liquidation or funding code must keep `conservation_error` at ~0; the tests
  check it every simulated second.
- **Respect transparency.** Agents read the market only through `MarketView`;
  new information exposed to agents must be gated by the transparency level.
- **Tests and docs.** Add a test for new behaviour and update
  `docs/user-guide.md` (and `docs/metrics.md` for new output fields).
- **Style.** Type hints, docstrings that explain *why*, `ruff check .` clean.

## Adding an agent or a scenario

- New scripted agents go in `src/glass_sim/agents/scripted.py` (or a new
  module under `agents/`) and are exported from `glass_sim/agents/__init__.py`.
- New scenarios go in `src/glass_sim/scenarios.py`. Keep scenarios as
  variations of `default()` that change one thing, so they stay comparable.

## Reporting a bug

Please include the command or script, the scenario and seed, the GLASS
version (`pip show glass-sim`) and the full error message.
