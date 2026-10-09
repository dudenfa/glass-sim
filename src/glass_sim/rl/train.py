"""Train a single learning agent with MaskablePPO.

    python -m glass_sim.rl.train --name ppo_full_v1 --timesteps 200000

Writes to ``checkpoints/glass/<name>/``:

- ``model.zip``: the final MaskablePPO policy;
- ``checkpoints/``: intermediate policies;
- ``config.json``: the full environment and PPO configuration (needed to run
  the policy later with the same observation and action spaces);
- ``training_log.csv``: one row per finished episode (return, PnL, actions).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .actions import ACTION_NAMES, ActionConfig
from .env import EnvConfig, LEARNER_ID, PerpTradingEnv, make_env
from .observation import ObservationConfig
from .reward import RewardConfig


def _parse_seeds(text: str) -> tuple:
    seeds: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        elif part.strip():
            seeds.append(int(part))
    return tuple(seeds)


def _episode_logger(path: Path):
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeLogger(BaseCallback):
        """Logs every finished episode: return, PnL, liquidations and the action mix."""

        def __init__(self) -> None:
            super().__init__()
            self.handle = path.open("w", newline="")
            self.writer = csv.writer(self.handle)
            self.writer.writerow(["timesteps", "episode", "return", "pnl_pct", "liquidations", "length", "seconds"] + [f"act_{n}" for n in ACTION_NAMES])
            self.episode = 0
            self.started = time.perf_counter()
            self.returns: Dict[int, float] = {}
            self.lengths: Dict[int, int] = {}
            self.liqs: Dict[int, float] = {}
            self.actions: Dict[int, np.ndarray] = {}
            self.recent: List[float] = []

        def _on_step(self) -> bool:
            infos = self.locals["infos"]
            dones = self.locals["dones"]
            rewards = self.locals["rewards"]
            for i, info in enumerate(infos):
                self.returns[i] = self.returns.get(i, 0.0) + float(rewards[i])
                self.lengths[i] = self.lengths.get(i, 0) + 1
                self.liqs[i] = self.liqs.get(i, 0.0) + float(info.get("liquidations", 0.0))
                acts = self.actions.setdefault(i, np.zeros(len(ACTION_NAMES)))
                if "action" in info:
                    acts[ACTION_NAMES.index(info["action"])] += 1
                if dones[i]:
                    self.episode += 1
                    equity = info.get("equity", 0.0)
                    pnl_pct = info.get("final_pnl_pct", None)
                    length = self.lengths[i]
                    mix = (acts / max(1, length)).round(4).tolist()
                    self.writer.writerow([self.num_timesteps, self.episode, round(self.returns[i], 4), pnl_pct, self.liqs[i], length, round(time.perf_counter() - self.started, 1)] + mix)
                    self.handle.flush()
                    self.recent.append(self.returns[i])
                    self.recent = self.recent[-20:]
                    self.logger.record("episode/return_mean20", float(np.mean(self.recent)))
                    self.returns[i] = 0.0
                    self.lengths[i] = 0
                    self.liqs[i] = 0.0
                    self.actions[i] = np.zeros(len(ACTION_NAMES))
            return True

        def _on_training_end(self) -> None:
            self.handle.close()

    return EpisodeLogger()


class _PnLInfoEnv(PerpTradingEnv):
    """Adds the final PnL percentage to the info of the last step of an episode."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            account = self.market.accounts[LEARNER_ID]
            info["final_pnl_pct"] = round(100.0 * (account.equity(self.market.mark_price) / account.initial_collateral - 1.0), 4)
        return obs, reward, terminated, truncated, info


def train(
    name: str,
    env_config: EnvConfig,
    timesteps: int = 200_000,
    n_envs: int = 8,
    seed: int = 0,
    out_root: str = "checkpoints/glass",
    ppo_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_every: int = 50_000,
    init_from: Optional[str] = None,
) -> Path:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)

    # Worker k of n plays seeds k, k+n, k+2n, ... so workers never replay each other's worlds.
    def factory(rank: int):
        def _init():
            env = _PnLInfoEnv(env_config)
            env.seed_offset = rank
            env.seed_stride = n_envs
            return env

        return _init

    venv_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = venv_cls([factory(i) for i in range(n_envs)])
    params = dict(
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 128]},
    )
    params.update(ppo_kwargs or {})
    config_record = {
        "name": name,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "algorithm": "MaskablePPO",
        "ppo": params,
        "env": asdict(env_config),
        "observation_features": _PnLInfoEnv(env_config).encoder.feature_names,
        "actions": ACTION_NAMES,
    }
    (out / "config.json").write_text(json.dumps(config_record, indent=2, default=str))

    if init_from:
        # Continue training an existing policy (same observation and action spaces required).
        model = MaskablePPO.load(init_from, env=venv, device="cpu")
        config_record["init_from"] = init_from
        (out / "config.json").write_text(json.dumps(config_record, indent=2, default=str))
    else:
        model = MaskablePPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **params)
    callbacks = CallbackList(
        [
            _episode_logger(out / "training_log.csv"),
            CheckpointCallback(save_freq=max(1, checkpoint_every // n_envs), save_path=str(out / "checkpoints"), name_prefix="model"),
        ]
    )
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    model.save(str(out / "model.zip"))
    venv.close()
    config_record["training_seconds"] = round(time.perf_counter() - started, 1)
    (out / "config.json").write_text(json.dumps(config_record, indent=2, default=str))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train one MaskablePPO agent in the perpetual-futures market")
    parser.add_argument("--name", required=True)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--transparency", default=None, choices=["anonymous", "attributed", "full"])
    parser.add_argument("--episode-duration", default="1h")
    parser.add_argument("--train-seeds", default="1-8")
    parser.add_argument("--collateral", type=float, default=100_000.0)
    parser.add_argument("--wake-interval", type=float, default=5.0)
    parser.add_argument("--private-signals", action="store_true", help="give the learner the informed traders' news signals (pipeline test)")
    parser.add_argument("--news-rate", type=float, default=None, help="override news per hour")
    parser.add_argument("--no-transparency-features", action="store_true")
    parser.add_argument("--trades-table", type=int, default=0, help="rows of recent trades in the observation (raw-table variant)")
    parser.add_argument("--accounts-table", type=int, default=0, help="rows of largest other accounts in the observation (FULL only)")
    parser.add_argument("--net-arch", default="128,128", help="hidden layer sizes of actor and critic")
    parser.add_argument("--liquidation-penalty", type=float, default=0.05)
    parser.add_argument("--inactivity-penalty", type=float, default=0.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--out-root", default="checkpoints/glass")
    parser.add_argument("--init-from", default=None, help="model.zip to continue training from")
    args = parser.parse_args(argv)

    from ..config import parse_duration

    env_config = EnvConfig(
        scenario=args.scenario,
        transparency=args.transparency,
        episode_duration=parse_duration(args.episode_duration),
        train_seeds=_parse_seeds(args.train_seeds),
        collateral=args.collateral,
        wake_interval=args.wake_interval,
        private_signals=args.private_signals,
        news_rate_per_hour=args.news_rate,
        observation=ObservationConfig(
            include_transparency=not args.no_transparency_features,
            trades_table=args.trades_table,
            accounts_table=args.accounts_table,
        ),
        actions=ActionConfig(max_leverage=args.max_leverage),
        reward=RewardConfig(liquidation_penalty=args.liquidation_penalty, inactivity_penalty=args.inactivity_penalty),
    )
    net_arch = [int(x) for x in args.net_arch.split(",") if x]
    out = train(
        args.name,
        env_config,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        out_root=args.out_root,
        ppo_kwargs={"policy_kwargs": {"net_arch": net_arch}},
        init_from=args.init_from,
    )
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
