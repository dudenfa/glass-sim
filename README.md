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
- `gymnasium` and `stable-baselines3` are included for the first PPO experiment
- the first install may take longer now because `stable-baselines3` pulls in CPU `torch`
- for optional test/development tools:

  ```bash
  python3 -m pip install -e ".[dev]"
  ```

- for editable RL work:

  ```bash
  python3 -m pip install -e ".[rl,dev]"
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

## RL Groundwork

The repo now includes the first single-agent RL boundary in:
- `src/marl_trading/rl/boundary.py`
- `src/marl_trading/rl/env.py`

Current status:
- one learning-controlled agent can already be inserted into the scripted market
- the environment exposes a compact vector observation and a small discrete action set
- PPO dependencies are installed with `requirements.txt` or `.[rl]`
- reward shaping supports both a linear absolute-inventory penalty and a separate quadratic inventory-risk penalty
- train/eval entry points live in `scripts/train_rl_agent.py` and `scripts/eval_rl_agent.py`

Quick smoke check:

```bash
PYTHONPATH=src python3 -c "from marl_trading.rl import RLAction, RLActionType, SingleAgentMarketEnv; env = SingleAgentMarketEnv(); obs = env.reset(seed=7, horizon=32); print('obs_dim=', len(obs)); _, reward, done, info = env.step(RLAction(RLActionType.HOLD)); print('reward=', round(reward, 4), 'done=', done, 'step=', info['step_index'])"
```

Example train command:

```bash
PYTHONPATH=src python3 scripts/train_rl_agent.py \
  --preset baseline \
  --learning-agent-id trend_01 \
  --total-timesteps 50000 \
  --inv-penalty 0.0 \
  --inv-risk-penalty 0.0005 \
  --checkpoint checkpoints/ppo_baseline_trend_01_zeroinv_risk5e-4_50k.zip
```

Example eval command:

```bash
PYTHONPATH=src python3 scripts/eval_rl_agent.py \
  --checkpoint checkpoints/ppo_baseline_trend_01_zeroinv_risk5e-4_50k.zip \
  --preset baseline \
  --learning-agent-id trend_01 \
  --inv-penalty 0.0 \
  --inv-risk-penalty 0.0005 \
  --output artifacts/rl_eval_baseline_trend_01_zeroinv_risk5e-4_50k.json
```

Both scripts preserve the existing long-form reward flags, accept the shorter aliases above, and record reward metadata including the base reward term, formula, and the two inventory-shaping coefficients.

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
  train_rl_agent.py
  eval_rl_agent.py
```

## Status

This is still an early research codebase, not a packaged product.

The live viewer is already usable and is the main way to inspect the synthetic market visually while the simulator is running.
