#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print metric tables from NSFW-SoK evaluation_summary_all.csv."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
NATIVE_SOK_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SUMMARY = NATIVE_SOK_ROOT / "results" / "generated" / "evaluation_summary_all.csv"
MODEL_ORDER = ["SD14", "SD15", "SD21", "SDXL", "SD3", "FLUX"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NSFW-SoK evaluation summary tables.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Path to evaluation_summary_all.csv.")
    parser.add_argument("--metric", action="append", default=None, help="Metric name filter. Can be used multiple times.")
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name filter. Can be used multiple times.")
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    rows = read_rows(summary_path)
    rows = filter_rows(rows, metrics=args.metric, datasets=args.dataset)

    print(f"[Analyze] summary: {summary_path}")
    print(f"[Analyze] rows: {len(rows)}")
    if args.metric:
        print(f"[Analyze] metric filters: {args.metric}")
    if args.dataset:
        print(f"[Analyze] dataset filters: {args.dataset}")
    print()

    if not rows:
        print("No rows matched.")
        return

    print_state_counts(rows)
    print()

    for metric_name in sorted({row["metric_name"] for row in rows}):
        metric_rows = [row for row in rows if row["metric_name"] == metric_name]
        print_metric_table(metric_name, metric_rows)
        print()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filter_rows(
    rows: list[dict[str, str]],
    *,
    metrics: list[str] | None,
    datasets: list[str] | None,
) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if datasets and row.get("dataset_name") not in datasets:
            continue
        if metrics and not any(metric_matches(row.get("metric_name", ""), metric) for metric in metrics):
            continue
        output.append(row)
    return output


def metric_matches(metric_name: str, query: str) -> bool:
    return metric_name == query or metric_name == f"mean_{query}" or query in metric_name


def print_state_counts(rows: list[dict[str, str]]) -> None:
    states = Counter(row.get("metric_state", "") for row in rows)
    datasets = Counter(row.get("dataset_name", "") for row in rows)
    print("[Analyze] metric states:", format_counter(states))
    print("[Analyze] datasets:", format_counter(datasets))


def format_counter(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def print_metric_table(metric_name: str, rows: list[dict[str, str]]) -> None:
    row_keys = sorted({method_key(row) for row in rows})
    models = ordered_models({row.get("model_name", "") for row in rows})
    values = build_value_map(rows)

    print(f"## {metric_name}")
    print()
    print("| method | " + " | ".join(models) + " |")
    print("|---|" + "|".join("---:" for _ in models) + "|")
    for row_key in row_keys:
        cells = [values.get((row_key, model), "-") for model in models]
        print(f"| {row_key} | " + " | ".join(cells) + " |")


def method_key(row: dict[str, str]) -> str:
    return "/".join(
        [
            row.get("method_name", ""),
            row.get("checkpoint_name", ""),
            row.get("attack_name", ""),
            row.get("dataset_name", ""),
        ]
    )


def ordered_models(models: set[str]) -> list[str]:
    ordered = [model for model in MODEL_ORDER if model in models]
    ordered.extend(sorted(model for model in models if model and model not in MODEL_ORDER))
    return ordered


def build_value_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(method_key(row), row.get("model_name", ""))].append(row)

    for key, group_rows in grouped.items():
        values[key] = format_group_value(group_rows)
    return values


def format_group_value(rows: list[dict[str, str]]) -> str:
    state = metric_state(rows)
    if state == "failed":
        return "ERR"
    if state == "pending":
        return "PENDING"
    if state == "skipped":
        return "SKIP"

    value = first_non_empty(row.get("metric_value", "") for row in rows)
    if value == "":
        return "-"
    return format_value(value)


def metric_state(rows: list[dict[str, str]]) -> str:
    states = {row.get("metric_state", "") for row in rows}
    statuses = {row.get("status", "") for row in rows}
    if "failed" in states or "failed" in statuses:
        return "failed"
    if "pending" in states or "pending" in statuses:
        return "pending"
    if states == {"skipped"} or statuses == {"skipped"}:
        return "skipped"
    return "completed"


def first_non_empty(values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def format_value(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value

    if number.is_integer():
        return str(int(number))
    if abs(number) >= 100:
        return f"{number:.2f}"
    return f"{number:.4f}"


if __name__ == "__main__":
    main()
