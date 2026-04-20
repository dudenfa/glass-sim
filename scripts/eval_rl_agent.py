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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent in the scripted market and emit a comparison-ready report.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved PPO checkpoint.")
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
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed override.")
    parser.add_argument("--horizon", type=int, default=None, help="Evaluation horizon override.")
    parser.add_argument(
        "--learning-agent-starting-inventory",
        type=float,
        default=0.0,
        help="Starting inventory for the runtime-replaced RL slot only.",
    )
    parser.add_argument("--reward-inventory-penalty", type=float, default=0.0, help="Optional absolute-inventory penalty coefficient.")
    parser.add_argument("--max-quantity", type=int, default=3, help="Maximum discrete order quantity exposed to PPO.")
    parser.add_argument("--max-price-offset-ticks", type=int, default=3, help="Maximum discrete price-offset ticks exposed to PPO.")
    parser.add_argument("--device", default="auto", help="Stable-Baselines device string.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions during evaluation.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the evaluation report.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of compact text.")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    return parser.parse_args(argv)


def import_ppo() -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError(
            "RL evaluation requires the optional dependency `stable-baselines3`. "
            "Install it before running scripts/eval_rl_agent.py."
        ) from exc
    return PPO


def _normalize_checkpoint_load_path(checkpoint_path: Path) -> str:
    resolved = checkpoint_path.resolve()
    if resolved.suffix == ".zip":
        return str(resolved.with_suffix(""))
    return str(resolved)


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
) -> tuple[SimulationConfig, int]:
    config = build_preset_config(preset_name)
    if seed is not None:
        config = replace(config, seed=int(seed))
    effective_horizon = int(horizon if horizon is not None else config.market.event_horizon)
    return config, effective_horizon


def build_rl_evaluation_payload(
    *,
    checkpoint_path: Path,
    preset_name: str,
    learning_agent_id: str,
    learning_agent_starting_inventory: float,
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
    report = "\n\n".join(
        [
            format_market_health_summary(summary, preset_name=preset_name, seed=config.seed, horizon=horizon),
            format_portfolio_health_breakdown(portfolio_rows),
        ]
    )
    serialized_rows = [row.to_dict() for row in portfolio_rows]
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
            "learning_agent_id": str(learning_agent_id),
            "learning_agent_starting_inventory": float(learning_agent_starting_inventory),
            "deterministic": bool(deterministic),
            "runtime_slot_replacement": True,
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
    PPO = import_ppo()
    from marl_trading.rl import GymSingleAgentMarketEnv, SingleAgentEnvConfig, SingleAgentMarketEnv

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config, effective_horizon = build_eval_config(args.preset, seed=args.seed, horizon=args.horizon)
    env_config = SingleAgentEnvConfig(
        learning_agent_id=str(args.learning_agent_id),
        learning_agent_starting_inventory=float(args.learning_agent_starting_inventory),
        reward_inventory_penalty=float(args.reward_inventory_penalty),
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
        action, _state = model.predict(observation, deterministic=not bool(args.stochastic))
        observation, _reward, terminated, truncated, _info = gym_env.step(action)

    result = gym_env.build_run_result()
    open_orders_by_agent = {
        agent_id: len(queue)
        for agent_id, queue in (core_env.simulator.open_orders.items() if core_env.simulator is not None else [])
    }
    return build_rl_evaluation_payload(
        checkpoint_path=checkpoint,
        preset_name=str(args.preset),
        learning_agent_id=str(args.learning_agent_id),
        learning_agent_starting_inventory=float(args.learning_agent_starting_inventory),
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
