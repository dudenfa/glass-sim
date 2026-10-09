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
from scripts.run_agent_ranking_panel import parse_seeds


RL01 = REPO_ROOT / "checkpoints/maskableppo_baseline_trend_01_multiseed8_sellpressurev3_100k.zip"
RL02 = REPO_ROOT / "checkpoints/maskableppo_baseline_rl_02_vs_frozen_rl_01_v1_softinv_b_6agent_100k.zip"
RL03_V2 = REPO_ROOT / "checkpoints/maskableppo_baseline_rl_03_v2_vs_frozen_rl_01_v1_rl_02_softinv_b_endowed16_7agent_100k.zip"
MARL7 = {
    "marl_01": REPO_ROOT / "checkpoints/marl_ippo_scripted_7agent_100k/marl_01.zip",
    "marl_02": REPO_ROOT / "checkpoints/marl_ippo_scripted_7agent_100k/marl_02.zip",
    "marl_03": REPO_ROOT / "checkpoints/marl_ippo_scripted_7agent_100k/marl_03.zip",
}
MARL10 = {
    "marl_vs_rl_01": REPO_ROOT / "checkpoints/marl_ippo_vs_rl_10agent_100k/marl_vs_rl_01.zip",
    "marl_vs_rl_02": REPO_ROOT / "checkpoints/marl_ippo_vs_rl_10agent_100k/marl_vs_rl_02.zip",
    "marl_vs_rl_03": REPO_ROOT / "checkpoints/marl_ippo_vs_rl_10agent_100k/marl_vs_rl_03.zip",
}


@dataclass(frozen=True)
class PopulationSpec:
    structure_id: str
    label: str
    trained_population: str
    deployment_population: str
    primary_id: str
    primary_checkpoint: Path
    frozen_agents: tuple[tuple[str, Path], ...]


def _frozen(checkpoints: dict[str, Path], primary_id: str) -> tuple[tuple[str, Path], ...]:
    return tuple((agent_id, checkpoint) for agent_id, checkpoint in checkpoints.items() if agent_id != primary_id)


STAGE_A_STRUCTURES = (
    PopulationSpec(
        "rl_trio_7",
        "Sequential RL trio in 7-agent market",
        "sequential_rl",
        "4 scripted + 3 RL",
        "rl_03",
        RL03_V2,
        (("rl_01", RL01), ("rl_02", RL02)),
    ),
    PopulationSpec(
        "marl7_in_7",
        "7-agent-trained MARL trio in 7-agent market",
        "marl7_scripted_only",
        "4 scripted + 3 MARL",
        "marl_01",
        MARL7["marl_01"],
        _frozen(MARL7, "marl_01"),
    ),
    PopulationSpec(
        "marl10_in_7",
        "10-agent-trained MARL trio in 7-agent market",
        "marl10_vs_rl",
        "4 scripted + 3 MARL",
        "marl_vs_rl_01",
        MARL10["marl_vs_rl_01"],
        _frozen(MARL10, "marl_vs_rl_01"),
    ),
    PopulationSpec(
        "marl7_in_10",
        "7-agent-trained MARL trio in 10-agent market",
        "marl7_scripted_only",
        "4 scripted + 3 RL + 3 MARL",
        "marl_01",
        MARL7["marl_01"],
        (("rl_01", RL01), ("rl_02", RL02), ("rl_03", RL03_V2), *_frozen(MARL7, "marl_01")),
    ),
    PopulationSpec(
        "marl10_in_10",
        "10-agent-trained MARL trio in 10-agent market",
        "marl10_vs_rl",
        "4 scripted + 3 RL + 3 MARL",
        "marl_vs_rl_01",
        MARL10["marl_vs_rl_01"],
        (("rl_01", RL01), ("rl_02", RL02), ("rl_03", RL03_V2), *_frozen(MARL10, "marl_vs_rl_01")),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the Stage A IPPO MARL populations on held-out seeds.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/marl_stage_a_31_40")
    parser.add_argument("--seeds", default="31-40")
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--starting-inventory", type=float, default=16.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def _argv(spec: PopulationSpec, *, seed: int, horizon: int, preset: str, starting_inventory: float) -> list[str]:
    argv = [
        "--checkpoint", str(spec.primary_checkpoint),
        "--preset", preset,
        "--seed", str(seed),
        "--horizon", str(horizon),
        "--learning-agent-id", spec.primary_id,
        "--add-learning-agent",
        "--learning-agent-template-id", "trend_01",
        "--learning-agent-starting-inventory", str(starting_inventory),
        "--phase-a-action-space",
        "--fixed-order-quantity", "1",
        "--fixed-price-offset-ticks", "1",
        "--reward-equity-delta-coefficient", "0.1",
        "--reward-inactivity-penalty", "0.01",
        "--reward-inventory-penalty", "0.002",
        "--reward-inventory-risk-penalty", "0.00005",
    ]
    for agent_id, checkpoint in spec.frozen_agents:
        argv.extend(["--frozen-agent", f"{agent_id}:{checkpoint}:trend_01:{starting_inventory}"])
    return argv


def build_structure_report(
    spec: PopulationSpec,
    *,
    seed: int,
    horizon: int,
    preset: str,
    starting_inventory: float,
) -> dict[str, Any]:
    args = eval_rl_agent.parse_args(
        _argv(
            spec,
            seed=seed,
            horizon=horizon,
            preset=preset,
            starting_inventory=starting_inventory,
        )
    )
    payload = eval_rl_agent._serialize_report(eval_rl_agent.evaluate_checkpoint(args))
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "marl_stage_a_structure_id": spec.structure_id,
            "marl_stage_a_structure_label": spec.label,
            "marl_stage_a_trained_population": spec.trained_population,
            "marl_stage_a_deployment_population": spec.deployment_population,
            "marl_stage_a_primary_id": spec.primary_id,
            "marl_stage_a_frozen_ids": [agent_id for agent_id, _checkpoint in spec.frozen_agents],
            "marl_stage_a_starting_inventory": float(starting_inventory),
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
    starting_inventory: float,
    skip_existing: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in (RL01, RL02, RL03_V2, *MARL7.values(), *MARL10.values()):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {checkpoint}")

    written: list[Path] = []
    total = len(STAGE_A_STRUCTURES) * len(seeds)
    completed = 0
    for spec in STAGE_A_STRUCTURES:
        for seed in seeds:
            completed += 1
            path = output_dir / f"{spec.structure_id}_seed_{seed}.json"
            if skip_existing and path.exists():
                print(f"[{completed}/{total}] Keeping {path.name}")
                continue
            payload = build_structure_report(
                spec,
                seed=seed,
                horizon=horizon,
                preset=preset,
                starting_inventory=starting_inventory,
            )
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            written.append(path)
            print(f"[{completed}/{total}] Wrote {path.name}")

    manifest = {
        "analysis": "stage_a_ippo_marl",
        "preset": preset,
        "seeds": list(seeds),
        "horizon": int(horizon),
        "starting_inventory": float(starting_inventory),
        "structures": [
            {
                "structure_id": spec.structure_id,
                "label": spec.label,
                "trained_population": spec.trained_population,
                "deployment_population": spec.deployment_population,
                "primary_id": spec.primary_id,
                "primary_checkpoint": str(spec.primary_checkpoint),
                "frozen_agents": [{"agent_id": agent_id, "checkpoint": str(path)} for agent_id, path in spec.frozen_agents],
            }
            for spec in STAGE_A_STRUCTURES
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    written = run_panel(
        output_dir=args.output_dir.resolve(),
        seeds=parse_seeds(args.seeds),
        horizon=int(args.horizon),
        preset=str(args.preset),
        starting_inventory=float(args.starting_inventory),
        skip_existing=bool(args.skip_existing),
    )
    print(f"Wrote {len(written)} Stage A reports to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
