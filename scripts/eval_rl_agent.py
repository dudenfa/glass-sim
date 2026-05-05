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

REWARD_BASE_TERM = "realized_pnl_delta"
ALGORITHM_CHOICES = ("auto", "ppo", "maskable_ppo")
REWARD_FORMULA = (
    "realized_pnl_delta - inactivity_penalty(if no trade) - abs(inventory) * reward_inventory_penalty - "
    "inventory^2 * reward_inventory_risk_penalty"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained PPO agent in the scripted market and emit a comparison-ready report "
            "with explicit reward-shaping metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Per-step reward follows the environment base term with optional inventory shaping: "
            f"{REWARD_FORMULA}."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved PPO checkpoint.")
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        default="auto",
        help="Algorithm backend used when loading the checkpoint. `auto` reads the checkpoint sidecar when available.",
    )
    parser.add_argument(
        "--preset",
        choices=available_preset_names(),
        default="baseline",
        help="Named preset used for evaluation.",
    )
    parser.add_argument(
        "--learning-agent-id",
        default="trend_01",
        help="Agent slot replaced by the learned policy at runtime.",
    )
    parser.add_argument(
        "--add-learning-agent",
        action="store_true",
        help="Add the PPO agent as a new participant instead of replacing an existing scripted slot.",
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
        help="Optional PPO checkpoint for a frozen runtime opponent inserted into the market during evaluation.",
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
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed override.")
    parser.add_argument("--horizon", type=int, default=None, help="Evaluation horizon override.")
    parser.add_argument(
        "--learning-agent-starting-inventory",
        type=float,
        default=0.0,
        help="Starting inventory for the runtime-replaced RL slot only.",
    )
    action_group = parser.add_argument_group(
        "Phase A action space",
        "Optional simplified action-space controls for evaluation parity with training.",
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
    parser.add_argument("--max-quantity", type=int, default=3, help="Maximum discrete order quantity exposed in the full action space.")
    parser.add_argument("--max-price-offset-ticks", type=int, default=3, help="Maximum discrete price-offset ticks exposed in the full action space.")
    parser.add_argument("--device", default="auto", help="Stable-Baselines device string.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions during evaluation.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the evaluation report.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of compact text.")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def import_ppo(algorithm: str) -> tuple[Any, Any | None]:
    try:
        if algorithm == "maskable_ppo":
            algo_module = importlib.import_module("sb3_contrib.ppo_mask")
            utils_module = importlib.import_module("sb3_contrib.common.maskable.utils")
            model_class = getattr(algo_module, "MaskablePPO")
            get_action_masks = getattr(utils_module, "get_action_masks")
            return model_class, get_action_masks
        algo_module = importlib.import_module("stable_baselines3")
        model_class = getattr(algo_module, "PPO")
        return model_class, None
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        if algorithm == "maskable_ppo":
            raise RuntimeError(
                "MaskablePPO evaluation requires optional dependencies `stable-baselines3`, "
                "`gymnasium`, and `sb3-contrib`. Install them before running scripts/eval_rl_agent.py."
            ) from exc
        raise RuntimeError(
            "RL evaluation requires the optional dependency `stable-baselines3`. "
            "Install it before running scripts/eval_rl_agent.py."
        ) from exc


def resolve_algorithm(checkpoint_path: Path, requested_algorithm: str) -> str:
    if requested_algorithm != "auto":
        return requested_algorithm
    metadata_path = checkpoint_path.with_suffix(".json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            algorithm = metadata.get("algorithm")
            if algorithm in {"ppo", "maskable_ppo"}:
                return str(algorithm)
        except (OSError, ValueError, TypeError):
            pass
    return "ppo"


def _normalize_checkpoint_load_path(checkpoint_path: Path) -> str:
    resolved = checkpoint_path.resolve()
    if resolved.suffix == ".zip":
        return str(resolved.with_suffix(""))
    return str(resolved)


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


def _preset_overview() -> str:
    lines = ["Available presets:"]
    for name in available_preset_names():
        preset = get_preset(name)
        lines.append(f"- {preset.name}: {preset.description}")
    return "\n".join(lines)


def build_eval_config(
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


def build_rl_evaluation_payload(
    *,
    checkpoint_path: Path,
    algorithm: str,
    preset_name: str,
    learning_agent_id: str,
    add_learning_agent: bool,
    learning_agent_template_id: str | None,
    learning_agent_starting_inventory: float,
    frozen_agent_checkpoint: Path | None,
    frozen_agent_id: str | None,
    add_frozen_agent: bool,
    frozen_agent_template_id: str | None,
    frozen_agent_starting_inventory: float | None,
    phase_a_action_space: bool,
    include_cancel_action: bool,
    fixed_order_quantity: int,
    fixed_price_offset_ticks: int,
    reward_inactivity_penalty: float,
    reward_inventory_penalty: float,
    reward_inventory_risk_penalty: float,
    result: Any,
    config: SimulationConfig,
    horizon: int,
    deterministic: bool,
    open_orders_by_agent: dict[str, int] | None = None,
) -> dict[str, Any]:
    from marl_trading.analysis import (
        build_agent_health_metrics,
        build_portfolio_health_rows,
        format_market_health_summary,
        format_portfolio_health_breakdown,
        summarize_market_health,
    )

    summary = summarize_market_health(result)
    final_mark_price = float(
        summary.final_midpoint
        if summary.final_midpoint is not None
        else summary.final_fundamental or config.market.starting_mid_price
    )
    order_counts = {str(agent_id): int(count) for agent_id, count in (open_orders_by_agent or {}).items()}
    agent_metrics = build_agent_health_metrics(
        list(result.event_log.events),
        config.agents,
        starting_midpoint=float(config.market.starting_mid_price),
        final_mark_price=final_mark_price,
        open_orders_by_agent=order_counts,
    )
    portfolio_rows = build_portfolio_health_rows(
        result.final_portfolios,
        config.agents,
        starting_midpoint=float(config.market.starting_mid_price),
        agent_metrics=agent_metrics,
    )
    adjusted_rows = []
    for row in portfolio_rows:
        if row.agent_id != learning_agent_id:
            adjusted_rows.append(row)
            continue
        adjusted_starting_inventory = float(learning_agent_starting_inventory)
        adjusted_starting_equity = float(row.starting_cash + adjusted_starting_inventory * float(config.market.starting_mid_price))
        adjusted_starting_free_equity = adjusted_starting_equity if row.starting_free_equity is not None else None
        adjusted_rows.append(
            replace(
                row,
                starting_inventory=adjusted_starting_inventory,
                starting_equity=adjusted_starting_equity,
                starting_free_equity=adjusted_starting_free_equity,
                inventory_delta=float(row.ending_inventory - adjusted_starting_inventory),
                equity_delta=float(row.ending_equity - adjusted_starting_equity),
                total_pnl=float(row.ending_equity - adjusted_starting_equity),
            )
        )
    portfolio_rows = adjusted_rows
    report = "\n\n".join(
        [
            format_market_health_summary(summary, preset_name=preset_name, seed=config.seed, horizon=horizon),
            format_portfolio_health_breakdown(portfolio_rows),
        ]
    )
    serialized_rows = [row.to_dict() for row in portfolio_rows]
    reward_metadata = build_reward_metadata(
        reward_inactivity_penalty=reward_inactivity_penalty,
        reward_inventory_penalty=reward_inventory_penalty,
        reward_inventory_risk_penalty=reward_inventory_risk_penalty,
    )
    return {
        "preset": get_preset(preset_name).name,
        "description": get_preset(preset_name).description,
        "label": f"{preset_name}_rl",
        "seed": int(config.seed),
        "horizon": int(horizon),
        "report": report,
        "summary": summary.to_dict(),
        "portfolio_breakdown": serialized_rows,
        "agents": serialized_rows,
        "metadata": {
            "mode": "rl_evaluation",
            "checkpoint": str(checkpoint_path.resolve()),
            "algorithm": str(algorithm),
            "learning_agent_id": str(learning_agent_id),
            "add_learning_agent": bool(add_learning_agent),
            "learning_agent_template_id": None if learning_agent_template_id is None else str(learning_agent_template_id),
            "learning_agent_starting_inventory": float(learning_agent_starting_inventory),
            "frozen_agent_checkpoint": None if frozen_agent_checkpoint is None else str(frozen_agent_checkpoint.resolve()),
            "frozen_agent_id": None if frozen_agent_id is None else str(frozen_agent_id),
            "add_frozen_agent": bool(add_frozen_agent),
            "frozen_agent_template_id": None if frozen_agent_template_id is None else str(frozen_agent_template_id),
            "frozen_agent_starting_inventory": None if frozen_agent_starting_inventory is None else float(frozen_agent_starting_inventory),
            "phase_a_action_space": bool(phase_a_action_space),
            "include_cancel_action": bool(include_cancel_action),
            "fixed_order_quantity": int(fixed_order_quantity),
            "fixed_price_offset_ticks": int(fixed_price_offset_ticks),
            "reward_inactivity_penalty": float(reward_inactivity_penalty),
            "reward_inventory_penalty": float(reward_inventory_penalty),
            "reward_inventory_risk_penalty": float(reward_inventory_risk_penalty),
            "deterministic": bool(deterministic),
            "runtime_learning_agent_mode": "add" if bool(add_learning_agent) else "replace",
            "runtime_frozen_agent_mode": None if frozen_agent_id is None else ("add" if bool(add_frozen_agent) else "replace"),
            **reward_metadata,
        },
    }


def _serialize_report(payload: dict[str, Any]) -> dict[str, Any]:
    portfolio_breakdown = payload.get("portfolio_breakdown")
    if isinstance(portfolio_breakdown, list):
        portfolio_breakdown = [
            row.to_dict() if hasattr(row, "to_dict") else dict(row)
            for row in portfolio_breakdown
        ]
    summary = payload.get("summary")
    if hasattr(summary, "to_dict"):
        summary = summary.to_dict()
    return {
        "preset": payload["preset"],
        "description": payload["description"],
        "label": payload.get("label"),
        "seed": payload["seed"],
        "horizon": payload["horizon"],
        "report": payload["report"],
        "summary": summary,
        "portfolio_breakdown": portfolio_breakdown,
        "agents": portfolio_breakdown,
        "metadata": payload.get("metadata", {}),
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    algorithm = resolve_algorithm(checkpoint, str(args.algorithm))
    PPO, get_action_masks = import_ppo(algorithm)
    from marl_trading.rl import GymSingleAgentMarketEnv, SingleAgentEnvConfig, SingleAgentMarketEnv

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if algorithm == "maskable_ppo" and not bool(args.phase_a_action_space):
        raise ValueError("MaskablePPO evaluation is currently supported only with the simplified Phase A action space.")
    validate_runtime_agent_args(args)

    config, effective_horizon = build_eval_config(
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
    env_config = SingleAgentEnvConfig(
        learning_agent_id=str(args.learning_agent_id),
        learning_agent_starting_inventory=float(args.learning_agent_starting_inventory),
        frozen_agent_id=None if args.frozen_agent_id is None else str(args.frozen_agent_id),
        frozen_agent_checkpoint_path=None if args.frozen_agent_checkpoint is None else str(Path(args.frozen_agent_checkpoint).resolve()),
        frozen_agent_starting_inventory=None if args.frozen_agent_starting_inventory is None else float(args.frozen_agent_starting_inventory),
        phase_a_action_space=bool(args.phase_a_action_space),
        include_cancel_action=bool(args.include_cancel_action),
        fixed_order_quantity=int(args.fixed_order_quantity),
        fixed_price_offset_ticks=int(args.fixed_price_offset_ticks),
        reward_realized_pnl_delta_coefficient=1.0,
        reward_inventory_penalty=float(args.reward_inventory_penalty),
        reward_inventory_risk_penalty=float(args.reward_inventory_risk_penalty),
        reward_inactivity_penalty=float(args.reward_inactivity_penalty),
        auto_increment_seed_on_reset=False,
    )
    core_env = SingleAgentMarketEnv(config=config, env_config=env_config, horizon=effective_horizon)
    gym_env = GymSingleAgentMarketEnv(
        core_env,
        max_quantity=int(args.max_quantity),
        max_price_offset_ticks=int(args.max_price_offset_ticks),
    )
    model = PPO.load(_normalize_checkpoint_load_path(checkpoint), device=str(args.device))

    observation, _ = gym_env.reset(seed=config.seed, options={"horizon": effective_horizon})
    terminated = False
    truncated = False
    while not (terminated or truncated):
        predict_kwargs: dict[str, Any] = {"deterministic": not bool(args.stochastic)}
        if get_action_masks is not None:
            predict_kwargs["action_masks"] = get_action_masks(gym_env)
        action, _state = model.predict(observation, **predict_kwargs)
        observation, _reward, terminated, truncated, _info = gym_env.step(action)

    result = gym_env.build_run_result()
    open_orders_by_agent = {
        agent_id: len(queue)
        for agent_id, queue in (core_env.simulator.open_orders.items() if core_env.simulator is not None else [])
    }
    return build_rl_evaluation_payload(
        checkpoint_path=checkpoint,
        algorithm=algorithm,
        preset_name=str(args.preset),
        learning_agent_id=str(args.learning_agent_id),
        add_learning_agent=bool(args.add_learning_agent),
        learning_agent_template_id=None if args.learning_agent_template_id is None else str(args.learning_agent_template_id),
        learning_agent_starting_inventory=float(args.learning_agent_starting_inventory),
        frozen_agent_checkpoint=None if args.frozen_agent_checkpoint is None else Path(args.frozen_agent_checkpoint),
        frozen_agent_id=None if args.frozen_agent_id is None else str(args.frozen_agent_id),
        add_frozen_agent=bool(args.add_frozen_agent),
        frozen_agent_template_id=None if args.frozen_agent_template_id is None else str(args.frozen_agent_template_id),
        frozen_agent_starting_inventory=None if args.frozen_agent_starting_inventory is None else float(args.frozen_agent_starting_inventory),
        phase_a_action_space=bool(args.phase_a_action_space),
        include_cancel_action=bool(args.include_cancel_action),
        fixed_order_quantity=int(args.fixed_order_quantity),
        fixed_price_offset_ticks=int(args.fixed_price_offset_ticks),
        reward_inactivity_penalty=float(args.reward_inactivity_penalty),
        reward_inventory_penalty=float(args.reward_inventory_penalty),
        reward_inventory_risk_penalty=float(args.reward_inventory_risk_penalty),
        result=result,
        config=config,
        horizon=effective_horizon,
        deterministic=not bool(args.stochastic),
        open_orders_by_agent=open_orders_by_agent,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_presets:
        print(_preset_overview())
        return

    try:
        payload = evaluate_checkpoint(args)
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    output = json.dumps(_serialize_report(payload), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + ("\n" if not output.endswith("\n") else ""), encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
