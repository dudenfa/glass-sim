# marl-trading

Synthetic one-asset market simulation for a university thesis on multi-agent reinforcement learning in trading.

The current repo contains the first public prototype:
- a spot-only synthetic market
- a central limit order book
- heterogeneous scripted agents
- a live local market viewer
- replay and analysis utilities

This project is the MARL continuation of the earlier thesis work in `../algo-trading-drl`.

## Current Focus

The main research question is:

> In a synthetic market where price, liquidity, and volatility emerge from agent interaction through an order book, what behaviors and regimes can we observe and study?

Right now the repo is focused on:
- building a stable synthetic market ecology
- visualizing market behavior live
- preparing the ground for RL and then MARL agents

## Repo Docs

- `THESIS.md`: initial whitepaper / manifesto
- `PROJECT_STATE.md`: working memory and implementation state

## Quick Start

Python `3.9+` is expected.

Create a virtual environment, activate it, and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Notes:
- `numpy` is required by the simulator and live viewer
- `Pillow` is used for the saved world/replay images
- for optional test/development tools:

  ```bash
  python3 -m pip install -e ".[dev]"
  ```

- `pytest` is included by `.[dev]`
- `matplotlib` is still optional for extra static analysis plots

## Run The Live Viewer

From the repo root:

```bash
PYTHONPATH=src python3 scripts/serve_market_view.py --paused --port 8765
```

Then open:

`http://127.0.0.1:8765`

Useful variant:

```bash
PYTHONPATH=src python3 scripts/serve_market_view.py --port 8765
```

That starts the market in autoplay mode.

## Run A Scripted Demo

```bash
PYTHONPATH=src python3 scripts/run_market_demo.py --seed 7 --horizon 240 --output-dir artifacts/demo_seed7_h240
```

This writes:
- event logs
- summary JSON
- optional replay figures

## Replay A Saved Event Log

```bash
PYTHONPATH=src python3 scripts/replay_market.py artifacts/demo_seed7_h240/market_events.jsonl
```

## Run Tests

```bash
PYTHONPATH=src pytest
```

## Project Layout

```text
src/marl_trading/
  agents/      scripted agent policies
  exchange/    matching engine and order book
  market/      simulator, fundamentals, news
  portfolio/   spot portfolio and reservation logic
  live/        local web viewer
  analysis/    replay and plotting tools

scripts/
  serve_market_view.py
  run_market_demo.py
  replay_market.py
```

## Status

This is still an early research codebase, not a packaged product.

The live viewer is already usable and is the main way to inspect the synthetic market visually while the simulator is running.
