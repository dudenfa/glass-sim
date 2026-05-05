from __future__ import annotations

import argparse
import importlib
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
from marl_trading.rl.scenario import prepare_frozen_agent_config, prepare_learning_agent_config

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
ALGORITHM_CHOICES = ("ppo", "maskable_ppo")
REWARD_BASE_TERM = "realized_pnl_delta"
REWARD_FORMULA = (
    "realized_pnl_delta - inactivity_penalty(if no trade) - abs(inventory) * reward_inventory_penalty - "
    "inventory^2 * reward_inventory_risk_penalty"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the first PPO agent inside the scripted synthetic market with explicit "
            "reward-shaping controls for the RL slot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Per-step reward follows the environment base term with optional inventory shaping: "
            f"{REWARD_FORMULA}."
        ),
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        default="ppo",
        help="RL algorithm backend used for training.",
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
    parser.add_argument(
        "--add-learning-agent",
        action="store_true",
        help="Add the RL agent as a new participant instead of replacing an existing scripted slot.",
    )
    parser.add_argument(
        "--learning-agent-template-id",
        default=None,
        help="Existing scripted agent id to clone when --add-learning-agent is enabled.",
    )
    parser.add_argument(
        "--frozen-agent-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO checkpoint for a frozen runtime opponent inserted into the market during training.",
    )
    parser.add_argument(
        "--frozen-agent-id",
        default=None,
        help="Agent id controlled by the frozen PPO checkpoint when --frozen-agent-checkpoint is provided.",
    )
    parser.add_argument(
        "--add-frozen-agent",
        action="store_true",
        help="Add the frozen PPO opponent as a new participant instead of replacing an existing scripted slot.",
    )
    parser.add_argument(
        "--frozen-agent-template-id",
        default=None,
        help="Existing scripted agent id to clone when --add-frozen-agent is enabled.",
    )
    parser.add_argument(
        "--frozen-agent-starting-inventory",
        type=float,
        default=None,
        help="Optional starting inventory override for the frozen PPO slot.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base seed for training episodes.")
    parser.add_argument(
        "--train-seeds",
        default=None,
        help="Comma-separated seed schedule for multi-seed training episodes. Overrides auto-increment base-seed resets when provided.",
    )
    parser.add_argument("--horizon", type=int, default=None, help="Override the preset event horizon.")
    parser.add_argument("--total-timesteps", type=int, default=50_000, help="Total PPO training timesteps.")
    parser.add_argument(
        "--learning-agent-starting-inventory",
        type=float,
        default=0.0,
        help="Starting inventory for the runtime-replaced RL slot only.",
    )
    action_group = parser.add_argument_group(
        "Phase A action space",
        "Optional simplified action-space controls for the first RL trading experiments.",
    )
    action_group.add_argument(
        "--phase-a-action-space",
        dest="phase_a_action_space",
        action="store_true",
        default=True,
        help="Use the simplified discrete Phase A action space.",
    )
    action_group.add_argument(
        "--full-action-space",
        dest="phase_a_action_space",
        action="store_false",
        help="Use the full MultiDiscrete action space instead of the simplified Phase A action space.",
    )
    action_group.add_argument(
        "--include-cancel-action",
        action="store_true",
        help="Include cancel_oldest in the simplified Phase A action set.",
    )
    action_group.add_argument(
        "--fixed-order-quantity",
        type=int,
        default=1,
        help="Fixed quantity used by the simplified Phase A action space.",
    )
    action_group.add_argument(
        "--fixed-price-offset-ticks",
        type=int,
        default=1,
        help="Fixed limit-price offset used by the simplified Phase A action space.",
    )
    reward_group = parser.add_argument_group(
        "Reward shaping",
        "Optional shaping terms applied on top of realized PnL delta.",
    )
    reward_group.add_argument(
        "--reward-inactivity-penalty",
        type=float,
        default=0.0,
        help="Flat penalty applied on RL steps where the learning agent records no trade.",
    )
    reward_group.add_argument(
        "--reward-inventory-penalty",
        "--inv-penalty",
        dest="reward_inventory_penalty",
        type=float,
        default=0.0,
        help="Linear abs(inventory) coefficient subtracted from the reward each RL step.",
    )
    reward_group.add_argument(
        "--reward-inventory-risk-penalty",
        "--inv-risk-penalty",
        dest="reward_inventory_risk_penalty",
        type=float,
        default=0.0,
        help="Quadratic inventory^2 risk coefficient subtracted from the reward each RL step.",
    )
    parser.add_argument("--max-quantity", type=int, default=3, help="Maximum discrete order quantity exposed to PPO in the full action space.")
    parser.add_argument("--max-price-offset-ticks", type=int, default=3, help="Maximum discrete price-offset ticks exposed to PPO in the full action space.")
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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def import_ppo_stack(algorithm: str) -> tuple[Any, Any]:
    try:
        monitor_module = importlib.import_module("stable_baselines3.common.monitor")
        Monitor = getattr(monitor_module, "Monitor")
        if algorithm == "maskable_ppo":
            algo_module = importlib.import_module("sb3_contrib.ppo_mask")
            model_class = getattr(algo_module, "MaskablePPO")
        else:
            algo_module = importlib.import_module("stable_baselines3")
            model_class = getattr(algo_module, "PPO")
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        if algorithm == "maskable_ppo":
            raise RuntimeError(
                "MaskablePPO training requires optional dependencies `stable-baselines3`, "
                "`gymnasium`, and `sb3-contrib`. Install them before running scripts/train_rl_agent.py."
            ) from exc
        raise RuntimeError(
            "PPO training requires optional dependencies `stable-baselines3` and `gymnasium`. "
            "Install them before running scripts/train_rl_agent.py."
        ) from exc
    return model_class, Monitor


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
    learning_agent_id: str = "trend_01",
    add_learning_agent: bool = False,
    learning_agent_template_id: str | None = None,
    frozen_agent_id: str | None = None,
    add_frozen_agent: bool = False,
    frozen_agent_template_id: str | None = None,
) -> tuple[SimulationConfig, int]:
    config = build_preset_config(preset_name)
    if seed is not None:
        config = replace(config, seed=int(seed))
    if frozen_agent_id is not None:
        config = prepare_frozen_agent_config(
            config,
            frozen_agent_id=frozen_agent_id,
            add_frozen_agent=add_frozen_agent,
            frozen_agent_template_id=frozen_agent_template_id,
        )
    config = prepare_learning_agent_config(
        config,
        learning_agent_id=learning_agent_id,
        add_learning_agent=add_learning_agent,
        learning_agent_template_id=learning_agent_template_id,
    )
    effective_horizon = int(horizon if horizon is not None else config.market.event_horizon)
    return config, effective_horizon


def parse_seed_schedule(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    values = [chunk.strip() for chunk in str(raw).split(",")]
    cleaned = tuple(int(value) for value in values if value)
    if raw is not None and not cleaned:
        raise ValueError("Expected at least one integer in --train-seeds.")
    return cleaned


def validate_runtime_agent_args(args: argparse.Namespace) -> None:
    frozen_agent_checkpoint = args.frozen_agent_checkpoint
    frozen_agent_id = str(args.frozen_agent_id or "").strip()
    if frozen_agent_checkpoint is None:
        if args.frozen_agent_id is not None or bool(args.add_frozen_agent) or args.frozen_agent_template_id is not None:
            raise ValueError("--frozen-agent-checkpoint is required when configuring a frozen agent.")
        return
    if not frozen_agent_id:
        raise ValueError("--frozen-agent-id is required when --frozen-agent-checkpoint is provided.")
    if frozen_agent_id == str(args.learning_agent_id):
        raise ValueError("frozen-agent-id must be different from learning-agent-id.")


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


def build_reward_metadata(
    *,
    reward_inactivity_penalty: float,
    reward_inventory_penalty: float,
    reward_inventory_risk_penalty: float,
) -> dict[str, Any]:
    return {
        "reward_signal": REWARD_BASE_TERM,
        "reward_base_term": REWARD_BASE_TERM,
        "reward_formula": REWARD_FORMULA,
        "reward_summary": (
            f"{REWARD_BASE_TERM} - {float(reward_inactivity_penalty):g} * inactivity(if no trade) - "
            f"{float(reward_inventory_penalty):g} * abs(inventory) - "
            f"{float(reward_inventory_risk_penalty):g} * inventory^2"
        ),
        "reward_shaping": {
            "inactivity_penalty": {
                "coefficient": float(reward_inactivity_penalty),
                "target": "no_trade_step",
            },
            "linear_inventory_penalty": {
                "coefficient": float(reward_inventory_penalty),
                "target": "abs_inventory",
            },
            "quadratic_inventory_risk_penalty": {
                "coefficient": float(reward_inventory_risk_penalty),
                "target": "inventory_squared",
            },
        },
    }


def build_training_metadata(
    *,
    args: argparse.Namespace,
    config: SimulationConfig,
    effective_horizon: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    reward_metadata = build_reward_metadata(
        reward_inactivity_penalty=float(args.reward_inactivity_penalty),
        reward_inventory_penalty=float(args.reward_inventory_penalty),
        reward_inventory_risk_penalty=float(args.reward_inventory_risk_penalty),
    )
    return {
        "preset": str(args.preset),
        "description": get_preset(str(args.preset)).description,
        "algorithm": str(args.algorithm),
        "learning_agent_id": str(args.learning_agent_id),
        "add_learning_agent": bool(args.add_learning_agent),
        "learning_agent_template_id": None if args.learning_agent_template_id is None else str(args.learning_agent_template_id),
        "frozen_agent_checkpoint": None if args.frozen_agent_checkpoint is None else str(Path(args.frozen_agent_checkpoint).resolve()),
        "frozen_agent_id": None if args.frozen_agent_id is None else str(args.frozen_agent_id),
        "add_frozen_agent": bool(args.add_frozen_agent),
        "frozen_agent_template_id": None if args.frozen_agent_template_id is None else str(args.frozen_agent_template_id),
        "frozen_agent_starting_inventory": None if args.frozen_agent_starting_inventory is None else float(args.frozen_agent_starting_inventory),
        "seed": int(config.seed),
        "train_seeds": list(parse_seed_schedule(args.train_seeds)),
        "horizon": int(effective_horizon),
        "total_timesteps": int(args.total_timesteps),
        "learning_agent_starting_inventory": float(args.learning_agent_starting_inventory),
        "phase_a_action_space": bool(args.phase_a_action_space),
        "include_cancel_action": bool(args.include_cancel_action),
        "fixed_order_quantity": int(args.fixed_order_quantity),
        "fixed_price_offset_ticks": int(args.fixed_price_offset_ticks),
        "reward_inactivity_penalty": float(args.reward_inactivity_penalty),
        "reward_inventory_penalty": float(args.reward_inventory_penalty),
        "reward_inventory_risk_penalty": float(args.reward_inventory_risk_penalty),
        "max_quantity": int(args.max_quantity),
        "max_price_offset_ticks": int(args.max_price_offset_ticks),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "device": str(args.device),
        "checkpoint": str(checkpoint_path),
        "runtime_learning_agent_mode": "add" if bool(args.add_learning_agent) else "replace",
        "runtime_frozen_agent_mode": (
            None
            if args.frozen_agent_id is None
            else ("add" if bool(args.add_frozen_agent) else "replace")
        ),
        **reward_metadata,
    }


def train_ppo_agent(args: argparse.Namespace) -> dict[str, Any]:
    PPO, Monitor = import_ppo_stack(str(args.algorithm))
    from marl_trading.rl import GymSingleAgentMarketEnv, SingleAgentEnvConfig, SingleAgentMarketEnv

    if str(args.algorithm) == "maskable_ppo" and not bool(args.phase_a_action_space):
        raise ValueError("MaskablePPO is currently supported only with the simplified Phase A action space.")
    validate_runtime_agent_args(args)

    config, effective_horizon = build_training_config(
        args.preset,
        seed=args.seed,
        horizon=args.horizon,
        learning_agent_id=str(args.learning_agent_id),
        add_learning_agent=bool(args.add_learning_agent),
        learning_agent_template_id=None if args.learning_agent_template_id is None else str(args.learning_agent_template_id),
        frozen_agent_id=None if args.frozen_agent_id is None else str(args.frozen_agent_id),
        add_frozen_agent=bool(args.add_frozen_agent),
        frozen_agent_template_id=None if args.frozen_agent_template_id is None else str(args.frozen_agent_template_id),
    )
    train_seeds = parse_seed_schedule(args.train_seeds)
    checkpoint_path = resolve_checkpoint_path(args)
    validate_checkpoint_target(checkpoint_path, force_overwrite=bool(args.force_overwrite))

    env_config = SingleAgentEnvConfig(
        learning_agent_id=str(args.learning_agent_id),
        learning_agent_starting_inventory=float(args.learning_agent_starting_inventory),
        frozen_agent_id=None if args.frozen_agent_id is None else str(args.frozen_agent_id),
        frozen_agent_checkpoint_path=None if args.frozen_agent_checkpoint is None else str(Path(args.frozen_agent_checkpoint).resolve()),
        frozen_agent_starting_inventory=None if args.frozen_agent_starting_inventory is None else float(args.frozen_agent_starting_inventory),
        train_seeds=train_seeds,
        phase_a_action_space=bool(args.phase_a_action_space),
        include_cancel_action=bool(args.include_cancel_action),
        fixed_order_quantity=int(args.fixed_order_quantity),
        fixed_price_offset_ticks=int(args.fixed_price_offset_ticks),
        reward_realized_pnl_delta_coefficient=1.0,
        reward_inventory_penalty=float(args.reward_inventory_penalty),
        reward_inventory_risk_penalty=float(args.reward_inventory_risk_penalty),
        reward_inactivity_penalty=float(args.reward_inactivity_penalty),
        auto_increment_seed_on_reset=not bool(train_seeds),
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
