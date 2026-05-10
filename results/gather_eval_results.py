#!/usr/bin/env python3
"""Gather LIBERO evaluation JSON files into a tab-separated summary table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT = Path("results_csv/eval_summary.csv")
FIELDNAMES = [
    "model",
    "Action horizon",
    "seed",
    "libero10",
    "libero10 avg",
    "libero10_object",
    "libero10_object avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather LIBERO eval JSON files into a tab-separated CSV table."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing eval JSON files. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output TSV/CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def model_from_path(model_path: str | None) -> str:
    if not model_path:
        return ""

    path = Path(model_path)
    if path.name.startswith("checkpoint-") and path.parent.name:
        return path.parent.name
    return path.name or str(path)


def fallback_from_filename(path: Path) -> tuple[str, int | None, int | None, str | None]:
    name = path.stem
    model = ""
    horizon = None
    seed = None
    perturbation = None

    model_match = re.search(r"model(.+?)_task", name)
    if model_match:
        model = model_match.group(1)

    horizon_match = re.search(r"_h(\d+)$", name)
    if horizon_match:
        horizon = int(horizon_match.group(1))

    seed_match = re.search(r"_seed(\d+)_", name)
    if seed_match:
        seed = int(seed_match.group(1))

    perturb_match = re.search(r"_pert([^_]+)_seed", name)
    if perturb_match:
        perturbation = perturb_match.group(1)

    return model, horizon, seed, perturbation


def read_result(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping unreadable JSON: {path} ({exc})")
        return None

    if not isinstance(result, dict):
        print(f"Skipping non-object JSON: {path}")
        return None

    return result


def metric_kind(path: Path, config: dict) -> str | None:
    perturbation = config.get("perturbation_type")
    if perturbation is None:
        _, _, _, perturbation = fallback_from_filename(path)

    if perturbation is None:
        return "libero10"
    if perturbation == "object":
        return "libero10_object"
    return None


def format_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def gather(results_dir: Path) -> list[dict[str, str]]:
    rows_by_key: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)

    for path in sorted(results_dir.glob("*.json")):
        result = read_result(path)
        if result is None:
            continue

        config = result.get("config") or {}
        if not isinstance(config, dict):
            config = {}

        fallback_model, fallback_horizon, fallback_seed, _ = fallback_from_filename(path)
        model = model_from_path(config.get("model_path")) or fallback_model or "unknown"
        horizon = config.get("action_horizon", fallback_horizon)
        seed = config.get("random_seed", fallback_seed)
        score = result.get("overall_success_rate")
        kind = metric_kind(path, config)

        if kind is None:
            continue
        if horizon is None or seed is None or score is None:
            print(f"Skipping incomplete result: {path}")
            continue

        try:
            key = (model, int(horizon), int(seed))
            rows_by_key[key][kind] = float(score)
        except (TypeError, ValueError):
            print(f"Skipping result with invalid values: {path}")

    standard_avgs: dict[tuple[str, int], float | None] = {}
    object_avgs: dict[tuple[str, int], float | None] = {}
    model_horizon_keys = {(model, horizon) for model, horizon, _ in rows_by_key}

    for model_horizon in model_horizon_keys:
        values = [
            metrics["libero10"]
            for (model, horizon, _), metrics in rows_by_key.items()
            if (model, horizon) == model_horizon and "libero10" in metrics
        ]
        standard_avgs[model_horizon] = mean(values)

        values = [
            metrics["libero10_object"]
            for (model, horizon, _), metrics in rows_by_key.items()
            if (model, horizon) == model_horizon and "libero10_object" in metrics
        ]
        object_avgs[model_horizon] = mean(values)

    rows = []
    for model, horizon, seed in sorted(rows_by_key, key=lambda x: (x[0], x[1], x[2])):
        metrics = rows_by_key[(model, horizon, seed)]
        model_horizon = (model, horizon)
        rows.append(
            {
                "model": model,
                "Action horizon": str(horizon),
                "seed": str(seed),
                "libero10": format_score(metrics.get("libero10")),
                "libero10 avg": format_score(standard_avgs[model_horizon]),
                "libero10_object": format_score(metrics.get("libero10_object")),
                "libero10_object avg": format_score(object_avgs[model_horizon]),
            }
        )

    return rows


def write_rows(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = gather(args.results_dir)
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
