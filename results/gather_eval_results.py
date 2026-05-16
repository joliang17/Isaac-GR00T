#!/usr/bin/env python3
"""Gather LIBERO evaluation JSON files into a tab-separated summary table.

Output format mirrors openpi's results_csv/eval_summary.csv: one row per
(model, action_horizon), aggregating success rates (as percentages) across
seeds, with a column per LIBERO suite / perturbation type.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT = Path("results_csv/eval_summary.csv")

# Vanilla libero10 plus libero_pro perturbations. Order here is the column order.
SUMMARY_FIELDS = [
    "model",
    "action_horizon",
    "seeds",
    "libero10_avg",
    "libero_pro_object_avg",
    "libero_pro_semantic_avg",
    "libero_pro_task_avg",
    "libero_pro_position_avg",
    "libero_pro_environment_avg",
]

# perturbation_type value -> summary column.
PERTURBATION_COLUMNS = {
    "object": "libero_pro_object_avg",
    "semantic": "libero_pro_semantic_avg",
    "task": "libero_pro_task_avg",
    "position": "libero_pro_position_avg",
    "environment": "libero_pro_environment_avg",
}


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
    """Return the checkpoint's run name (the directory holding checkpoint-*)."""
    if not model_path:
        return ""

    path = Path(model_path)
    if path.name.startswith("checkpoint-") and path.parent.name:
        return path.parent.name
    return path.name or str(path)


def fallback_from_filename(path: Path) -> tuple[str, int | None, int | None]:
    """Best-effort (model, horizon, seed) recovery when config fields are missing."""
    name = path.stem
    model = ""
    horizon = None
    seed = None

    model_match = re.search(r"model(.+?)_task", name)
    if model_match:
        model = model_match.group(1)

    horizon_match = re.search(r"_h(\d+)$", name)
    if horizon_match:
        horizon = int(horizon_match.group(1))

    seed_match = re.search(r"_seed(\d+)_", name)
    if seed_match:
        seed = int(seed_match.group(1))

    return model, horizon, seed


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


def result_column(config: dict) -> str | None:
    """Map a result's task suite / perturbation to a summary column."""
    suite = str(config.get("task_suite_name", "")).replace("_", "").lower()
    if suite and not suite.startswith("libero10"):
        return None

    perturbation = config.get("perturbation_type")
    perturbation = str(perturbation or "").lower()
    if not perturbation or perturbation == "none":
        return "libero10_avg"
    return PERTURBATION_COLUMNS.get(perturbation)


def format_average(value: float | None) -> str:
    if value is None:
        return ""
    rounded = round(float(value), 2)
    text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def gather(results_dir: Path) -> list[dict[str, str]]:
    # (model, horizon) -> {"seeds": set, "values": column -> [percent, ...]}
    grouped: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"seeds": set(), "values": defaultdict(list)}
    )

    paths = sorted(results_dir.glob("libero_eval_*.json"))
    paths += sorted(results_dir.glob("libero_pro_*.json"))

    for path in paths:
        result = read_result(path)
        if result is None:
            continue

        config = result.get("config") or {}
        if not isinstance(config, dict):
            config = {}

        fb_model, fb_horizon, fb_seed = fallback_from_filename(path)
        model = model_from_path(config.get("model_path")) or fb_model or "unknown"
        horizon = config.get("action_horizon", fb_horizon)
        seed = config.get("random_seed", fb_seed)
        score = result.get("overall_success_rate")
        column = result_column(config)

        if column is None:
            continue
        if horizon is None or seed is None or score is None:
            print(f"Skipping incomplete result: {path}")
            continue

        try:
            key = (model, int(horizon))
            group = grouped[key]
            group["seeds"].add(int(seed))
            group["values"][column].append(float(score) * 100.0)
        except (TypeError, ValueError):
            print(f"Skipping result with invalid values: {path}")

    rows = []
    for model, horizon in sorted(grouped, key=lambda x: (x[0], x[1])):
        group = grouped[(model, horizon)]
        row = {field: "" for field in SUMMARY_FIELDS}
        row["model"] = model
        row["action_horizon"] = str(horizon)
        row["seeds"] = ",".join(str(s) for s in sorted(group["seeds"]))
        for column, percents in group["values"].items():
            row[column] = format_average(mean(percents)) if percents else ""
        rows.append(row)

    return rows


def write_rows(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SUMMARY_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = gather(args.results_dir)
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
