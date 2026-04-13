from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.analysis import compare_market_runs, format_market_run_comparison, load_market_run
from scripts.run_market_health import build_market_health_report


def parse_run_spec(spec: str) -> dict[str, str]:
    text = spec.strip()
    if not text:
        raise ValueError("Run spec cannot be empty.")

    tokens = [token for token in re.split(r"[,\s]+", text) if token]
    if not tokens:
        raise ValueError("Run spec cannot be empty.")

    values: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if "=" not in token:
            if index == 0 and "preset" not in values:
                values["preset"] = token
                continue
            raise ValueError(f"Invalid run spec token: {token!r}")

        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in {"preset", "seed", "horizon"}:
            raise ValueError(f"Unsupported run spec key: {key!r}")
        if not value:
            raise ValueError(f"Missing value for run spec key: {key!r}")
        values[key] = value

    if "preset" not in values:
        raise ValueError("Run spec must include a preset.")
    return values


def _resolve_source(source: str) -> dict[str, object]:
    path = Path(source)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"JSON report at {path} must decode to an object.")
        return data

    spec = parse_run_spec(source)
    preset = spec["preset"]
    seed = int(spec["seed"]) if "seed" in spec else None
    horizon = int(spec["horizon"]) if "horizon" in spec else None
    payload = build_market_health_report(
        preset,
        seed=seed,
        horizon=horizon,
        portfolio_breakdown=True,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two synthetic market runs from JSON reports or run specs.",
    )
    parser.add_argument("left", help="Left report path or run spec, for example preset=baseline seed=7 horizon=10000.")
    parser.add_argument("right", help="Right report path or run spec, for example preset=high_news seed=7 horizon=10000.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown text.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    left_payload = _resolve_source(args.left)
    right_payload = _resolve_source(args.right)

    comparison = compare_market_runs(left_payload, right_payload)
    output = json.dumps(comparison.to_dict(), indent=2, sort_keys=True) if args.json else format_market_run_comparison(comparison)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + ("\n" if not output.endswith("\n") else ""), encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
