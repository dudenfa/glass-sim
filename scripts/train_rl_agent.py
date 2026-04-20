from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.configs import available_preset_names, build_preset_config, get_preset
from marl_trading.core.config import SimulationConfig

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first PPO agent inside the scripted synthetic market.",
    )
    parser.add_argument(
        "--preset",
        choices=available_preset_names(),
        default="baseline",
        help="Named preset used as the scripted market ecology.",
    )
    parser.add_argument(
        "--learning-agent-id",
        default="trend_01",
        help="Agent slot to replace at runtime with the RL controller.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base seed for training episodes.")
    parser.add_argument("--horizon", type=int, default=None, help="Override the preset event horizon.")
    parser.add_argument("--total-timesteps", type=int, default=50_000, help="Total PPO training timesteps.")
    parser.add_argument(
        "--learning-agent-starting-inventory",
        type=float,
        default=0.0,
        help="Starting inventory for the runtime-replaced RL slot only.",
    )
    parser.add_argument("--reward-inventory-penalty", type=float, default=0.0, help="Optional absolute-inventory penalty coefficient.")
    parser.add_argument("--max-quantity", type=int, default=3, help="Maximum discrete order quantity exposed to PPO.")
    parser.add_argument("--max-price-offset-ticks", type=int, default=3, help="Maximum discrete price-offset ticks exposed to PPO.")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor.")
    parser.add_argument("--verbose", type=int, default=1, help="stable-baselines3 verbosity level.")
    parser.add_argument("--device", default="auto", help="Stable-Baselines device string.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional output path for the trained PPO checkpoint.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional JSON sidecar for training metadata.")
    parser.add_argument("--force-overwrite", action="store_true", help="Allow overwriting an existing checkpoint path.")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    return parser.parse_args(argv)


def import_ppo_stack() -> tuple[Any, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError(
            "PPO training requires optional dependencies `stable-baselines3` and `gymnasium`. "
            "Install them before running scripts/train_rl_agent.py."
        ) from exc
    return PPO, Monitor


def _preset_overview() -> str:
    lines = ["Available presets:"]
    for name in available_preset_names():
        preset = get_preset(name)
        lines.append(f"- {preset.name}: {preset.description}")
    return "\n".join(lines)


def build_training_config(
    preset_name: str,
    *,
    seed: int | None = None,
    horizon: int | None = None,
) -> tuple[SimulationConfig, int]:
    config = build_preset_config(preset_name)
    if seed is not None:
        config = replace(config, seed=int(seed))
    effective_horizon = int(horizon if horizon is not None else config.market.event_horizon)
    return config, effective_horizon


def default_checkpoint_path(preset_name: str, learning_agent_id: str) -> Path:
    return DEFAULT_CHECKPOINT_DIR / f"ppo_{preset_name}_{learning_agent_id}.zip"


def metadata_path_for_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".json")


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    path = args.checkpoint if args.checkpoint is not None else default_checkpoint_path(args.preset, args.learning_agent_id)
    return Path(path).resolve()


def validate_checkpoint_target(path: Path, *, force_overwrite: bool) -> None:
    if path.exists() and not force_overwrite:
        raise FileExistsError(f"Checkpoint already exists: {path}. Use --force-overwrite to replace it.")


def build_training_metadata(
    *,
    args: argparse.Namespace,
    config: SimulationConfig,
    effective_horizon: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "preset": str(args.preset),
        "description": get_preset(str(args.preset)).description,
        "learning_agent_id": str(args.learning_agent_id),
        "seed": int(config.seed),
        "horizon": int(effective_horizon),
        "total_timesteps": int(args.total_timesteps),
        "learning_agent_starting_inventory": float(args.learning_agent_starting_inventory),
        "reward_inventory_penalty": float(args.reward_inventory_penalty),
        "max_quantity": int(args.max_quantity),
        "max_price_offset_ticks": int(args.max_price_offset_ticks),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "device": str(args.device),
        "checkpoint": str(checkpoint_path),
        "runtime_slot_replacement": True,
    }


def train_ppo_agent(args: argparse.Namespace) -> dict[str, Any]:
    PPO, Monitor = import_ppo_stack()
    from marl_trading.rl import GymSingleAgentMarketEnv, SingleAgentEnvConfig, SingleAgentMarketEnv

    config, effective_horizon = build_training_config(args.preset, seed=args.seed, horizon=args.horizon)
    checkpoint_path = resolve_checkpoint_path(args)
    validate_checkpoint_target(checkpoint_path, force_overwrite=bool(args.force_overwrite))

    env_config = SingleAgentEnvConfig(
        learning_agent_id=str(args.learning_agent_id),
        learning_agent_starting_inventory=float(args.learning_agent_starting_inventory),
        reward_inventory_penalty=float(args.reward_inventory_penalty),
        auto_increment_seed_on_reset=True,
    )
    core_env = SingleAgentMarketEnv(config=config, env_config=env_config, horizon=effective_horizon)
    gym_env = GymSingleAgentMarketEnv(
        core_env,
        max_quantity=int(args.max_quantity),
        max_price_offset_ticks=int(args.max_price_offset_ticks),
    )
    monitored_env = Monitor(gym_env)

    model = PPO(
        "MlpPolicy",
        monitored_env,
        verbose=int(args.verbose),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        device=str(args.device),
    )
    model.learn(total_timesteps=int(args.total_timesteps), progress_bar=False)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_path))

    metadata = build_training_metadata(
        args=args,
        config=config,
        effective_horizon=effective_horizon,
        checkpoint_path=checkpoint_path,
    )
    metadata_output = Path(args.metadata_output).resolve() if args.metadata_output is not None else metadata_path_for_checkpoint(checkpoint_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["metadata_output"] = str(metadata_output)
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_presets:
        print(_preset_overview())
        return

    try:
        metadata = train_ppo_agent(args)
    except (RuntimeError, FileExistsError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Saved PPO checkpoint to {metadata['checkpoint']}")
    print(f"Saved training metadata to {metadata['metadata_output']}")


if __name__ == "__main__":
    main()
