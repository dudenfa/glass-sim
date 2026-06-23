from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_rl_agent
from scripts.run_market_health import _serialize_report as serialize_market_health_report
from scripts.run_market_health import build_market_health_report


DEFAULT_SEEDS = tuple(range(31, 41))
DEFAULT_CHECKPOINTS = {
    "rl_01": REPO_ROOT / "checkpoints/maskableppo_baseline_trend_01_multiseed8_sellpressurev3_100k.zip",
    "rl_02": REPO_ROOT / "checkpoints/maskableppo_baseline_rl_02_vs_frozen_rl_01_v1_softinv_b_6agent_100k.zip",
    "rl_03": REPO_ROOT / "checkpoints/maskableppo_baseline_rl_03_vs_frozen_rl_01_v1_rl_02_softinv_b_7agent_100k.zip",
}


@dataclass(frozen=True)
class StructureSpec:
    structure_id: str
    label: str
    learning_agent_id: str | None = None
    frozen_agent_ids: tuple[str, ...] = ()


STRUCTURES = (
    StructureSpec("s0_scripted", "S0: 4 scripted"),
    StructureSpec("s1_rl01", "S1: 4 scripted + rl_01", learning_agent_id="rl_01"),
    StructureSpec("s1_rl02", "S1: 4 scripted + rl_02", learning_agent_id="rl_02"),
    StructureSpec("s1_rl03", "S1: 4 scripted + rl_03", learning_agent_id="rl_03"),
    StructureSpec("s2_rl01_rl02", "S2: 4 scripted + rl_01 + rl_02", learning_agent_id="rl_02", frozen_agent_ids=("rl_01",)),
    StructureSpec(
        "s3_rl01_rl02_rl03",
        "S3: 4 scripted + rl_01 + rl_02 + rl_03",
        learning_agent_id="rl_03",
        frozen_agent_ids=("rl_01", "rl_02"),
    ),
)


def parse_seeds(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("Seed ranges must be ascending.")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("At least one seed is required.")
    return tuple(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed-16 agent-ranking evaluation panel.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/agent_ranking_31_40")
    parser.add_argument("--seeds", default="31-40", help="Comma-separated seeds and inclusive ranges.")
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--rl-inventory", type=float, default=16.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def _eval_argv(
    spec: StructureSpec,
    *,
    seed: int,
    horizon: int,
    preset: str,
    rl_inventory: float,
) -> list[str]:
    if spec.learning_agent_id is None:
        raise ValueError("Scripted-only structure has no RL evaluation command.")
    learning_agent_id = spec.learning_agent_id
    argv = [
        "--checkpoint",
        str(DEFAULT_CHECKPOINTS[learning_agent_id]),
        "--preset",
        preset,
        "--seed",
        str(seed),
        "--horizon",
        str(horizon),
        "--learning-agent-id",
        learning_agent_id,
        "--add-learning-agent",
        "--learning-agent-template-id",
        "trend_01",
        "--learning-agent-starting-inventory",
        str(rl_inventory),
    ]
    for frozen_agent_id in spec.frozen_agent_ids:
        argv.extend(
            [
                "--frozen-agent",
                f"{frozen_agent_id}:{DEFAULT_CHECKPOINTS[frozen_agent_id]}:trend_01:{rl_inventory}",
            ]
        )
    return argv


def build_structure_report(
    spec: StructureSpec,
    *,
    seed: int,
    horizon: int,
    preset: str,
    rl_inventory: float,
) -> dict[str, Any]:
    if spec.learning_agent_id is None:
        payload = serialize_market_health_report(
            build_market_health_report(preset, seed=seed, horizon=horizon, portfolio_breakdown=True)
        )
    else:
        args = eval_rl_agent.parse_args(
            _eval_argv(spec, seed=seed, horizon=horizon, preset=preset, rl_inventory=rl_inventory)
        )
        payload = eval_rl_agent._serialize_report(eval_rl_agent.evaluate_checkpoint(args))

    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "ranking_structure_id": spec.structure_id,
            "ranking_structure_label": spec.label,
            "ranking_rl_starting_inventory": float(rl_inventory),
            "ranking_learning_agent_id": spec.learning_agent_id,
            "ranking_frozen_agent_ids": list(spec.frozen_agent_ids),
        }
    )
    payload["metadata"] = metadata
    return payload


def run_panel(
    *,
    output_dir: Path,
    seeds: tuple[int, ...],
    horizon: int,
    preset: str,
    rl_inventory: float,
    skip_existing: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = len(STRUCTURES) * len(seeds)
    completed = 0
    for spec in STRUCTURES:
        for seed in seeds:
            completed += 1
            path = output_dir / f"{spec.structure_id}_seed_{seed}.json"
            if skip_existing and path.exists():
                print(f"[{completed}/{total}] Keeping {path.name}")
                written.append(path)
                continue
            print(f"[{completed}/{total}] Running {spec.label}, seed {seed}")
            report = build_structure_report(
                spec,
                seed=seed,
                horizon=horizon,
                preset=preset,
                rl_inventory=rl_inventory,
            )
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            written.append(path)
    manifest = {
        "preset": preset,
        "horizon": int(horizon),
        "seeds": list(seeds),
        "rl_starting_inventory": float(rl_inventory),
        "structures": [
            {
                "structure_id": spec.structure_id,
                "label": spec.label,
                "learning_agent_id": spec.learning_agent_id,
                "frozen_agent_ids": list(spec.frozen_agent_ids),
            }
            for spec in STRUCTURES
        ],
        "report_count": len(written),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for checkpoint in DEFAULT_CHECKPOINTS.values():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {checkpoint}")
    written = run_panel(
        output_dir=args.output_dir.resolve(),
        seeds=parse_seeds(args.seeds),
        horizon=int(args.horizon),
        preset=str(args.preset),
        rl_inventory=float(args.rl_inventory),
        skip_existing=bool(args.skip_existing),
    )
    print(f"Wrote {len(written)} ranking reports to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
