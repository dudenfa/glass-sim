from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.configs import available_preset_names, get_preset
from marl_trading.marl import IPPOTrainingConfig, JointIPPORunner, build_joint_training_config
from marl_trading.rl.scenario import FrozenAgentSpec, build_frozen_agent_specs


DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints" / "marl_ippo_run"


def parse_ids(raw: str) -> tuple[str, str, str]:
    ids = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("--learner-ids must contain exactly three unique comma-separated ids.")
    return ids  # type: ignore[return-value]


def parse_seed_schedule(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("--train-seeds must include at least one seed.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train three concurrent self-interested MaskablePPO agents in one shared synthetic market.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=available_preset_names(), default="baseline")
    parser.add_argument("--seed", type=int, default=1, help="Base training seed.")
    parser.add_argument("--train-seeds", default="1,2,3,4,5,6,7,8", help="Comma-separated episode seed schedule.")
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="PPO decisions collected and trained per MARL agent.")
    parser.add_argument("--learner-ids", default="marl_01,marl_02,marl_03")
    parser.add_argument("--learning-agent-starting-inventory", type=float, default=16.0)
    parser.add_argument(
        "--frozen-agent",
        action="append",
        default=[],
        metavar="AGENT_ID:CHECKPOINT[:TEMPLATE_ID[:STARTING_INVENTORY]]",
        help="Frozen PPO opponent. Repeat to train MARL against the existing RL trio.",
    )
    parser.add_argument("--exclude-agent", action="append", default=[], metavar="AGENT_ID")
    parser.add_argument("--preserve-excluded-turn-slots", action="store_true")
    parser.add_argument("--n-steps", type=int, default=512, help="Shared rollout decisions per MARL agent before PPO updates.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-equity-delta-coefficient", type=float, default=0.1)
    parser.add_argument("--reward-inactivity-penalty", type=float, default=0.01)
    parser.add_argument("--reward-inventory-penalty", type=float, default=0.002)
    parser.add_argument("--reward-inventory-risk-penalty", type=float, default=0.00005)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-overwrite", action="store_true")
    return parser


def resolve_frozen_agent_specs(args: argparse.Namespace) -> tuple[FrozenAgentSpec, ...]:
    return build_frozen_agent_specs(frozen_agents=tuple(args.frozen_agent or ()))


def build_metadata(
    *,
    args: argparse.Namespace,
    learner_id: str,
    learner_ids: tuple[str, str, str],
    frozen_agents: tuple[FrozenAgentSpec, ...],
    checkpoint_path: Path,
    training_result: dict[str, object],
) -> dict[str, object]:
    return {
        "algorithm": "maskable_ppo",
        "marl_algorithm": "ippo",
        "marl_training_mode": "joint_self_interested",
        "marl_learner_id": learner_id,
        "marl_learner_ids": list(learner_ids),
        "preset": str(args.preset),
        "description": get_preset(str(args.preset)).description,
        "seed": int(args.seed),
        "train_seeds": list(parse_seed_schedule(args.train_seeds)),
        "horizon": int(args.horizon),
        "total_timesteps": int(args.total_timesteps),
        "training_result": training_result,
        "learning_agent_id": learner_id,
        "add_learning_agent": True,
        "learning_agent_template_id": "trend_01",
        "learning_agent_starting_inventory": float(args.learning_agent_starting_inventory),
        "frozen_agents": [spec.to_metadata() for spec in frozen_agents],
        "excluded_agent_ids": list(args.exclude_agent or ()),
        "preserve_excluded_turn_slots": bool(args.preserve_excluded_turn_slots),
        "phase_a_action_space": True,
        "include_cancel_action": False,
        "fixed_order_quantity": 1,
        "fixed_price_offset_ticks": 1,
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "reward_equity_delta_coefficient": float(args.reward_equity_delta_coefficient),
        "reward_inactivity_penalty": float(args.reward_inactivity_penalty),
        "reward_inventory_penalty": float(args.reward_inventory_penalty),
        "reward_inventory_risk_penalty": float(args.reward_inventory_risk_penalty),
        "reward_signal": "realized_pnl_delta + reward_equity_delta_coefficient * equity_delta",
        "checkpoint": str(checkpoint_path),
    }


def train_joint_ippo(args: argparse.Namespace) -> dict[str, object]:
    learner_ids = parse_ids(args.learner_ids)
    frozen_agents = resolve_frozen_agent_specs(args)
    collisions = set(learner_ids) & {spec.agent_id for spec in frozen_agents}
    if collisions:
        raise ValueError(f"MARL learner ids collide with frozen opponents: {', '.join(sorted(collisions))}")

    output_dir = args.output_dir.resolve()
    checkpoint_paths = {learner_id: output_dir / f"{learner_id}.zip" for learner_id in learner_ids}
    existing = [path for path in checkpoint_paths.values() if path.exists()]
    if existing and not args.force_overwrite:
        raise FileExistsError("Checkpoint already exists: " + ", ".join(str(path) for path in existing))

    training_config = IPPOTrainingConfig(
        learner_ids=learner_ids,
        learner_starting_inventory=float(args.learning_agent_starting_inventory),
        frozen_agents=frozen_agents,
        horizon=int(args.horizon),
        train_seeds=parse_seed_schedule(args.train_seeds),
        total_timesteps=int(args.total_timesteps),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        reward_equity_delta_coefficient=float(args.reward_equity_delta_coefficient),
        reward_inactivity_penalty=float(args.reward_inactivity_penalty),
        reward_inventory_penalty=float(args.reward_inventory_penalty),
        reward_inventory_risk_penalty=float(args.reward_inventory_risk_penalty),
        device=str(args.device),
        seed=int(args.seed),
    )
    simulation_config = build_joint_training_config(
        preset_name=str(args.preset),
        learner_ids=learner_ids,
        frozen_agents=frozen_agents,
        seed=int(args.seed),
        excluded_agent_ids=tuple(args.exclude_agent or ()),
        preserve_excluded_turn_slots=bool(args.preserve_excluded_turn_slots),
    )
    runner = JointIPPORunner(simulation_config, training_config)

    def report_progress(completed: int, total: int) -> None:
        print(f"Joint IPPO progress: {completed}/{total} decisions per MARL agent")

    training_result = runner.train(progress_callback=report_progress)
    runner.save(checkpoint_paths)
    metadata_paths: dict[str, Path] = {}
    for learner_id, checkpoint_path in checkpoint_paths.items():
        metadata = build_metadata(
            args=args,
            learner_id=learner_id,
            learner_ids=learner_ids,
            frozen_agents=frozen_agents,
            checkpoint_path=checkpoint_path,
            training_result=training_result,
        )
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metadata_paths[learner_id] = metadata_path
    return {
        "checkpoint_paths": {learner_id: str(path) for learner_id, path in checkpoint_paths.items()},
        "metadata_paths": {learner_id: str(path) for learner_id, path in metadata_paths.items()},
        "training_result": training_result,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = train_joint_ippo(args)
    except (FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    for learner_id, checkpoint_path in result["checkpoint_paths"].items():
        print(f"Saved MARL checkpoint for {learner_id} to {checkpoint_path}")
        print(f"Saved MARL metadata for {learner_id} to {result['metadata_paths'][learner_id]}")


if __name__ == "__main__":
    main()
