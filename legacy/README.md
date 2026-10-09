# Legacy: the V1 spot-market thesis code

This folder keeps the first version of the project, which preceded GLASS: a
one-asset **spot** market (no leverage, no shorting) with a scripted ecology,
single-agent PPO agents (RL01-RL03) and a first concurrent multi-agent (IPPO)
experiment. It is preserved so that the thesis results remain reproducible.
It is **not maintained** and is not part of the `glass-sim` package.

The exact code that produced the V1 results is also available at the git tag
`v0.1-spot-thesis`.

| folder | contents |
|---|---|
| `src/marl_trading/` | the V1 simulator, RL boundary, live viewer and analysis code |
| `scripts/` | V1 command-line entry points (demo, health reports, training, evaluation, panels) |
| `tests/` | V1 test suite (a few tests were already failing at the time of freezing) |
| `checkpoints/` | V1 trained agents (`rl_01` ... `rl_03`, Stage A MARL) |
| `docs/` | `THESIS.md` (V1 manifesto), `PROJECT_STATE_V1.md` (full V1 research log), agent catalog, metrics, commands, preset summaries |

## Running it

From the repository root:

```bash
pip install -r legacy/requirements.txt
PYTHONPATH=legacy/src python legacy/scripts/run_market_health.py --preset baseline --seed 7 --horizon 10000
PYTHONPATH=legacy/src python legacy/scripts/serve_market_view.py --paused --port 8765
PYTHONPATH=legacy/src pytest legacy/tests
```

Paths to checkpoints inside the V1 documents refer to the original layout
(`checkpoints/...`); they now live under `legacy/checkpoints/`.

Why the project moved on from V1 to GLASS is explained in
[`docs/research/PROJECT_STATE.md`](../docs/research/PROJECT_STATE.md), section 2.
