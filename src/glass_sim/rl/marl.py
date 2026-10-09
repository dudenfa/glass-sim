"""Multi-agent training: several learning agents trading in the same market.

Independent PPO (IPPO): each learner has its own policy network, its own
experience and its own reward (the same exit-value reward as the single-agent
environment). With ``shared_policy=True`` all learners use one network
(parameter sharing) instead.

How experience is collected:

- ``n_workers`` processes each run a market that contains the scripted
  population plus ``n_learners`` learning agents (``marl_01``, ``marl_02``, ...).
- A market pauses whenever any learner must decide. Each worker therefore has
  exactly one pending decision at a time; the main process batches the pending
  decisions of all workers per learner, samples actions, and sends them back.
- A learner's transition runs from one of its decisions to its next one. Its
  reward (change in exit value plus penalties) is computed when it decides
  again, when it is liquidated out, or when the episode ends.

Training uses the clipped PPO objective with generalized advantage estimation,
computed separately for each (worker, learner) stream of transitions, with the
network and optimizer of ``sb3_contrib.MaskablePPO``. Policies are saved as
regular ``MaskablePPO`` checkpoints, so ``PolicyAgent``, the evaluation
scripts and the viewer can use them directly.

    python -m glass_sim.rl.marl --name marl3_full --learners 3 --timesteps 600000
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import scenarios
from ..agents.base import ExternalAgent
from ..config import AgentSpec, parse_duration
from ..market import PerpetualMarket
from ..types import Transparency
from .actions import ACTION_NAMES, ActionConfig, ActionSpace
from .observation import ObservationConfig, ObservationEncoder
from .reward import RewardConfig, RewardTracker


@dataclass(frozen=True)
class MultiAgentConfig:
    scenario: str = "default"
    transparency: Optional[str] = None
    n_learners: int = 3
    episode_duration: float = 4 * 3600.0
    train_seeds: Tuple[int, ...] = tuple(range(2000, 3000))
    collateral: float = 100_000.0
    wake_interval: float = 5.0
    prefix: str = "marl"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def learner_ids(self) -> List[str]:
        return [f"{self.prefix}_{i + 1:02d}" for i in range(self.n_learners)]

    def market_config(self, seed: int, duration: Optional[float] = None):
        config = scenarios.get(self.scenario, seed=seed, duration=float(duration or self.episode_duration))
        if self.transparency:
            config = replace(config, transparency=Transparency(self.transparency))
        learners = AgentSpec(
            ExternalAgent,
            count=self.n_learners,
            collateral=self.collateral,
            params={"wake_interval": self.wake_interval},
            prefix=self.prefix,
            external=True,
        )
        return replace(config, agents=config.agents + (learners,))


# =============================================================================
# Worker: one market with several learners
# =============================================================================


class MultiAgentMarket:
    """A market with several external learners, stepped one decision at a time."""

    def __init__(self, config: MultiAgentConfig) -> None:
        self.config = config
        self.ids = config.learner_ids()
        self.encoder = ObservationEncoder(config.observation)
        self.actions = ActionSpace(config.actions)
        self.market: Optional[PerpetualMarket] = None
        self.trackers: Dict[str, RewardTracker] = {}
        self.open: Dict[str, bool] = {}

    def _advance(self) -> Optional[str]:
        market = self.market
        while market.pending_decision is None and not market.finished:
            market.step()
        return market.pending_decision

    def reset(self, seed: int) -> Dict[str, Any]:
        self.market = PerpetualMarket(self.config.market_config(seed))
        self.trackers = {agent_id: RewardTracker(self.market, agent_id, self.config.reward) for agent_id in self.ids}
        self.open = {agent_id: False for agent_id in self.ids}
        return self._next_request(closed=[])

    def _observe(self, agent_id: str) -> Tuple[np.ndarray, np.ndarray]:
        view = self.market.view(agent_id)
        return self.encoder.encode(view), self.actions.mask(view)

    def _close(self, agent_id: str, terminal: bool) -> Tuple[str, float, bool, Dict[str, float]]:
        reward, info = self.trackers[agent_id].step()
        self.open[agent_id] = False
        return agent_id, float(reward), terminal, info

    def _next_request(self, closed: List[Tuple[str, float, bool, Dict[str, float]]]) -> Dict[str, Any]:
        """Advance to the next learner decision; return it plus any transitions that ended meanwhile."""
        while True:
            agent_id = self._advance()
            # Learners liquidated out since the last request end their episode now.
            for other in self.ids:
                if self.open[other] and not self.market.accounts[other].active:
                    closed.append(self._close(other, terminal=True))
            if agent_id is None:
                for other in self.ids:
                    if self.open[other]:
                        closed.append(self._close(other, terminal=False))
                return {"done": True, "closed": closed, "final": self._episode_stats()}
            if not self.market.accounts[agent_id].active:
                self.market.act(agent_id, [])
                continue
            if self.open[agent_id]:
                closed.append(self._close(agent_id, terminal=False))
            obs, mask = self._observe(agent_id)
            return {"done": False, "agent": agent_id, "obs": obs, "mask": mask, "closed": closed}

    def step(self, agent_id: str, action: int) -> Dict[str, Any]:
        view = self.market.view(agent_id)
        mask = self.actions.mask(view)
        orders = self.actions.decode(action, view) if mask[action] else []
        self.market.act(agent_id, orders)
        self.open[agent_id] = True
        return self._next_request(closed=[])

    def _episode_stats(self) -> Dict[str, float]:
        mark = self.market.mark_price
        return {agent_id: 100.0 * (self.market.accounts[agent_id].equity(mark) / self.market.accounts[agent_id].initial_collateral - 1.0) for agent_id in self.ids}


def _worker(remote, config: MultiAgentConfig, seeds: Sequence[int]) -> None:
    env = MultiAgentMarket(config)
    episode = 0
    try:
        while True:
            command, payload = remote.recv()
            if command == "reset":
                seed = int(seeds[episode % len(seeds)])
                episode += 1
                result = env.reset(seed)
                result["seed"] = seed
                remote.send(result)
            elif command == "step":
                remote.send(env.step(*payload))
            elif command == "close":
                break
    finally:
        remote.close()


# =============================================================================
# Learner bookkeeping and PPO update
# =============================================================================


@dataclass
class _Pending:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float


@dataclass
class _Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    terminal: bool
    next_value: float = 0.0
    has_next: bool = False


class Learner:
    """One policy plus its experience streams (one per worker)."""

    def __init__(self, name: str, model, n_workers: int, gamma: float, gae_lambda: float) -> None:
        self.name = name
        self.model = model
        self.policy = model.policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pending: Dict[int, _Pending] = {}
        self.streams: Dict[int, List[_Transition]] = {w: [] for w in range(n_workers)}
        self.episode_returns: List[float] = []
        self._running: Dict[int, float] = {}
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)

    def size(self) -> int:
        return sum(len(s) for s in self.streams.values())

    def record_close(self, worker: int, reward: float, terminal: bool) -> None:
        pending = self.pending.pop(worker, None)
        if pending is None:
            return
        self.streams[worker].append(_Transition(pending.obs, pending.mask, pending.action, pending.log_prob, pending.value, reward, terminal))
        self._running[worker] = self._running.get(worker, 0.0) + reward

    def record_value_of_next(self, worker: int, value: float) -> None:
        """The value of the new decision state bootstraps the previous transition."""
        stream = self.streams[worker]
        if stream and not stream[-1].terminal and not stream[-1].has_next:
            stream[-1].next_value = value
            stream[-1].has_next = True

    def end_episode(self, worker: int) -> None:
        self.episode_returns.append(self._running.pop(worker, 0.0))

    def batch(self) -> Dict[str, np.ndarray]:
        obs, masks, actions, log_probs, values, advantages, returns = [], [], [], [], [], [], []
        for worker, stream in self.streams.items():
            adv = 0.0
            stream_adv = [0.0] * len(stream)
            for t in reversed(range(len(stream))):
                tr = stream[t]
                not_terminal = 0.0 if tr.terminal else 1.0
                next_value = tr.next_value if tr.has_next else tr.value  # episode cut: bootstrap with own value
                delta = tr.reward + self.gamma * not_terminal * next_value - tr.value
                continues = t + 1 < len(stream) and tr.has_next and not tr.terminal
                adv = delta + (self.gamma * self.gae_lambda * not_terminal * adv if continues else 0.0)
                stream_adv[t] = adv
            for tr, a in zip(stream, stream_adv):
                obs.append(tr.obs)
                masks.append(tr.mask)
                actions.append(tr.action)
                log_probs.append(tr.log_prob)
                values.append(tr.value)
                advantages.append(a)
                returns.append(a + tr.value)
        # Every stored transition is complete (its reward is known and, unless the episode ended,
        # its successor's value was recorded), so streams can be cleared after the update.
        for worker in self.streams:
            self.streams[worker] = []
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "masks": np.asarray(masks, dtype=bool),
            "actions": np.asarray(actions, dtype=np.int64),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
            "returns": np.asarray(returns, dtype=np.float32),
        }


def ppo_update(policy, data: Dict[str, np.ndarray], epochs: int, batch_size: int, clip: float, ent_coef: float, vf_coef: float, max_grad_norm: float) -> Dict[str, float]:
    import torch

    device = policy.device
    n = len(data["actions"])
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0
    data = dict(data)
    # Normalize advantages once over the whole batch. Per-minibatch normalization breaks
    # when the last minibatch holds a single sample (its standard deviation is undefined).
    adv_all = data["advantages"]
    data["advantages"] = ((adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)).astype(np.float32)
    tensors = {k: torch.as_tensor(v, device=device) for k, v in data.items()}
    for _ in range(epochs):
        order = np.random.permutation(n)
        for start in range(0, n, batch_size):
            chunk = order[start:start + batch_size]
            if len(chunk) < 2:
                continue
            idx = torch.as_tensor(chunk, device=device)
            obs = tensors["obs"][idx]
            actions = tensors["actions"][idx]
            masks = tensors["masks"][idx].cpu().numpy()
            values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=masks)
            values = values.flatten()
            adv = tensors["advantages"][idx]
            ratio = torch.exp(log_prob - tensors["log_probs"][idx])
            loss_pi = -torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
            loss_v = torch.nn.functional.mse_loss(tensors["returns"][idx], values)
            loss_ent = -entropy.mean()
            loss = loss_pi + vf_coef * loss_v + ent_coef * loss_ent
            if not torch.isfinite(loss):
                continue  # safety net: never let an invalid loss reach the weights
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            policy.optimizer.step()
            with torch.no_grad():
                stats["approx_kl"] += float(((ratio - 1) - torch.log(ratio)).mean())
            stats["policy_loss"] += loss_pi.item()
            stats["value_loss"] += loss_v.item()
            stats["entropy"] += -loss_ent.item()
            updates += 1
    return {k: v / max(1, updates) for k, v in stats.items()}


def _make_model(obs_size: int, n_actions: int, net_arch: List[int], learning_rate: float, seed: int):
    import gymnasium as gym
    from sb3_contrib import MaskablePPO

    class _Spaces(gym.Env):
        observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(obs_size,), dtype=np.float32)
        action_space = gym.spaces.Discrete(n_actions)

        def reset(self, *, seed=None, options=None):
            return np.zeros(obs_size, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(obs_size, dtype=np.float32), 0.0, True, False, {}

        def action_masks(self):
            return np.ones(n_actions, dtype=bool)

    return MaskablePPO("MlpPolicy", _Spaces(), learning_rate=learning_rate, seed=seed, device="cpu", policy_kwargs={"net_arch": net_arch}, verbose=0)


# =============================================================================
# Training loop
# =============================================================================


def train_marl(
    name: str,
    config: MultiAgentConfig,
    timesteps_per_learner: int = 600_000,
    n_workers: int = 8,
    rollout_per_learner: int = 8192,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    net_arch: Sequence[int] = (128, 128),
    shared_policy: bool = False,
    seed: int = 0,
    out_root: str = "checkpoints/glass",
    init_from: Optional[str] = None,
) -> Path:
    import torch

    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    ids = config.learner_ids()
    encoder = ObservationEncoder(config.observation)
    n_actions = len(ACTION_NAMES)

    models = {}
    if init_from:
        # Continue training saved learners (one <id>.zip per learner in ``init_from``).
        from sb3_contrib import MaskablePPO

        for agent_id in ids:
            models[agent_id] = MaskablePPO.load(str(Path(init_from) / f"{agent_id}.zip"), device="cpu")
    else:
        shared = _make_model(encoder.size, n_actions, list(net_arch), learning_rate, seed) if shared_policy else None
        for i, agent_id in enumerate(ids):
            models[agent_id] = shared if shared_policy else _make_model(encoder.size, n_actions, list(net_arch), learning_rate, seed * 100 + i)
    learners = {agent_id: Learner(agent_id, models[agent_id], n_workers, gamma, gae_lambda) for agent_id in ids}
    # Learners that share one network are updated together.
    policy_groups = [tuple(ids)] if shared_policy else [(agent_id,) for agent_id in ids]

    record = {
        "name": name,
        "algorithm": "IPPO (MaskablePPO networks)" + (" with parameter sharing" if shared_policy else ""),
        "learners": ids,
        "timesteps_per_learner": timesteps_per_learner,
        "n_workers": n_workers,
        "rollout_per_learner": rollout_per_learner,
        "ppo": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "gamma": gamma, "gae_lambda": gae_lambda, "clip": clip, "ent_coef": ent_coef, "vf_coef": vf_coef, "net_arch": list(net_arch)},
        "seed": seed,
        "init_from": init_from,
        "env": asdict(config),
        "observation_features": encoder.feature_names,
        "actions": ACTION_NAMES,
    }
    (out / "config.json").write_text(json.dumps(record, indent=2, default=str))

    ctx = mp.get_context("spawn")
    seeds = list(config.train_seeds)
    remotes, processes = [], []
    for w in range(n_workers):
        parent, child = ctx.Pipe()
        worker_seeds = seeds[w::n_workers]
        proc = ctx.Process(target=_worker, args=(child, config, worker_seeds), daemon=True)
        proc.start()
        child.close()
        remotes.append(parent)
        processes.append(proc)

    log_handle = (out / "training_log.csv").open("w", newline="")
    log = csv.writer(log_handle)
    log.writerow(["seconds", "decisions", "learner", "update", "mean_episode_return_last10", "policy_loss", "value_loss", "entropy", "approx_kl"] + [f"act_{n}" for n in ACTION_NAMES])
    episodes_csv = (out / "episodes.csv").open("w", newline="")
    episodes_log = csv.writer(episodes_csv)
    episodes_log.writerow(["seconds", "worker", "seed"] + [f"{agent_id}_pnl_pct" for agent_id in ids])

    started = time.perf_counter()
    decisions = {agent_id: 0 for agent_id in ids}
    updates = {agent_id: 0 for agent_id in ids}
    for remote in remotes:
        remote.send(("reset", None))
    requests = [remote.recv() for remote in remotes]
    worker_seed = [r.get("seed") for r in requests]

    def handle_closed(w: int, closed) -> None:
        for agent_id, reward, terminal, _info in closed:
            learners[agent_id].record_close(w, reward, terminal)

    try:
        while min(decisions.values()) < timesteps_per_learner:
            # 1. Record transitions that ended and episode ends.
            for w, req in enumerate(requests):
                handle_closed(w, req["closed"])
                if req["done"]:
                    for agent_id in ids:
                        learners[agent_id].end_episode(w)
                    episodes_log.writerow([round(time.perf_counter() - started, 1), w, worker_seed[w]] + [round(req["final"][a], 4) for a in ids])
                    episodes_csv.flush()
            # 2. Reset finished workers.
            for w, req in enumerate(requests):
                if req["done"]:
                    remotes[w].send(("reset", None))
            for w, req in enumerate(requests):
                if req["done"]:
                    requests[w] = remotes[w].recv()
                    worker_seed[w] = requests[w].get("seed")
                    handle_closed(w, requests[w]["closed"])
            # 3. Batch the pending decisions by learner.
            by_learner: Dict[str, List[int]] = {}
            for w, req in enumerate(requests):
                by_learner.setdefault(req["agent"], []).append(w)
            chosen: Dict[int, int] = {}
            for agent_id, workers in by_learner.items():
                learner = learners[agent_id]
                obs = np.stack([requests[w]["obs"] for w in workers])
                masks = np.stack([requests[w]["mask"] for w in workers])
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=learner.policy.device)
                    dist = learner.policy.get_distribution(obs_t, action_masks=masks)
                    actions = dist.get_actions(deterministic=False)
                    log_probs = dist.log_prob(actions).cpu().numpy()
                    values = learner.policy.predict_values(obs_t).cpu().numpy().flatten()
                    actions = actions.cpu().numpy()
                for k, w in enumerate(workers):
                    learner.record_value_of_next(w, float(values[k]))
                    learner.pending[w] = _Pending(obs[k], masks[k], int(actions[k]), float(log_probs[k]), float(values[k]))
                    learner.action_counts[int(actions[k])] += 1
                    chosen[w] = int(actions[k])
                    decisions[agent_id] += 1
            # 4. Step all workers in parallel.
            for w, req in enumerate(requests):
                remotes[w].send(("step", (req["agent"], chosen[w])))
            requests = [remote.recv() for remote in remotes]
            # 5. Update learners whose buffers are full.
            for members in policy_groups:
                total = sum(learners[a].size() for a in members)
                if total < rollout_per_learner * len(members):
                    continue
                data_parts = [learners[a].batch() for a in members]
                data = {k: np.concatenate([d[k] for d in data_parts]) for k in data_parts[0]}
                stats = ppo_update(models[members[0]].policy, data, epochs, batch_size, clip, ent_coef, vf_coef, max_grad_norm)
                for a in members:
                    updates[a] += 1
                    learner = learners[a]
                    mix = learner.action_counts / max(1, learner.action_counts.sum())
                    recent = learner.episode_returns[-10:]
                    log.writerow(
                        [round(time.perf_counter() - started, 1), decisions[a], a, updates[a], round(float(np.mean(recent)), 4) if recent else ""]
                        + [round(stats[k], 5) for k in ("policy_loss", "value_loss", "entropy", "approx_kl")]
                        + [round(float(x), 4) for x in mix]
                    )
                    learner.action_counts[:] = 0
                log_handle.flush()
    finally:
        for remote in remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in processes:
            proc.join(timeout=5)
        log_handle.close()
        episodes_csv.close()

    for agent_id in ids:
        models[agent_id].save(str(out / f"{agent_id}.zip"))
        # Per-learner config so PolicyAgent can rebuild the spaces (same layout as single-agent runs).
        learner_dir = out / agent_id
        learner_dir.mkdir(exist_ok=True)
        models[agent_id].save(str(learner_dir / "model.zip"))
        (learner_dir / "config.json").write_text(json.dumps({"env": {"observation": asdict(config.observation), "actions": asdict(config.actions), "collateral": config.collateral, "wake_interval": config.wake_interval}}, indent=2))
    record["training_seconds"] = round(time.perf_counter() - started, 1)
    record["decisions"] = decisions
    (out / "config.json").write_text(json.dumps(record, indent=2, default=str))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train several learning agents together (IPPO) in the futures market")
    parser.add_argument("--name", required=True)
    parser.add_argument("--learners", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=600_000, help="decisions per learner")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--transparency", default=None, choices=["anonymous", "attributed", "full"])
    parser.add_argument("--episode-duration", default="4h")
    parser.add_argument("--train-seeds", default="2000-2999")
    parser.add_argument("--collateral", type=float, default=100_000.0)
    parser.add_argument("--trades-table", type=int, default=0)
    parser.add_argument("--accounts-table", type=int, default=0)
    parser.add_argument("--net-arch", default="128,128")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--rollout", type=int, default=8192, help="transitions per learner between updates")
    parser.add_argument("--out-root", default="checkpoints/glass")
    parser.add_argument("--init-from", default=None, help="run directory whose marl_XX.zip learners are trained further")
    args = parser.parse_args(argv)
    from .train import _parse_seeds

    config = MultiAgentConfig(
        scenario=args.scenario,
        transparency=args.transparency,
        n_learners=args.learners,
        episode_duration=parse_duration(args.episode_duration),
        train_seeds=_parse_seeds(args.train_seeds),
        collateral=args.collateral,
        observation=ObservationConfig(trades_table=args.trades_table, accounts_table=args.accounts_table),
    )
    out = train_marl(
        args.name,
        config,
        timesteps_per_learner=args.timesteps,
        n_workers=args.workers,
        rollout_per_learner=args.rollout,
        net_arch=[int(x) for x in args.net_arch.split(",") if x],
        shared_policy=args.shared_policy,
        seed=args.seed,
        out_root=args.out_root,
        init_from=args.init_from,
    )
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
