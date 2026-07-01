from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


POLICIES = [
    ("dive", "DIVE"),
    ("value_rank", "ValueRank"),
    ("uncertainty", "Uncertainty"),
    ("random", "Random"),
    ("oracle", "Oracle"),
]

SELECTION_POLICIES = [
    ("dive", "DIVE"),
    ("value_rank", "ValueRank"),
    ("uncertainty", "Uncertainty"),
]


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.5")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = json.loads(Path(args.input).read_text())
    budgets = _parse_budgets(args.budgets)
    markdown = render_markdown_summary(report, budgets=budgets)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    return markdown


def render_markdown_summary(report: dict, budgets: list[float]) -> str:
    lines = [
        "# Main Curves Summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in report.get('seeds', []))}",
        "",
    ]
    for environment in report.get("environments", []):
        if environment not in report:
            continue
        lines.extend(_environment_summary(environment, report[environment], budgets))
    return "\n".join(lines).rstrip() + "\n"


def _environment_summary(environment: str, data: dict, budgets: list[float]) -> list[str]:
    aggregate = {float(row["budget_fraction"]): row for row in data.get("aggregate_budget_curve", [])}
    seed_curves = [run.get("budget_curve", []) for run in data.get("seed_runs", [])]
    lines = [
        f"## {environment}",
        "",
        "### return",
        "",
        "| budget | " + " | ".join(label for _, label in POLICIES) + " |",
        "| --- | " + " | ".join("---" for _ in POLICIES) + " |",
    ]
    for budget in budgets:
        row = aggregate.get(float(budget))
        if row is None:
            continue
        values = [_format_mean_std(row, f"{policy}_return") for policy, _ in POLICIES]
        lines.append(f"| {_format_float(budget)} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### selected delta",
            "",
            "| budget | " + " | ".join(label for _, label in SELECTION_POLICIES) + " |",
            "| --- | " + " | ".join("---" for _ in SELECTION_POLICIES) + " |",
        ]
    )
    for budget in budgets:
        values = [
            _format_seed_stat(seed_curves, budget, lambda row, policy=policy: row[f"{policy}_selection"]["mean_delta_r"])
            for policy, _ in SELECTION_POLICIES
        ]
        lines.append(f"| {_format_float(budget)} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### harmful / wasted",
            "",
            "| budget | " + " | ".join(label for _, label in SELECTION_POLICIES) + " |",
            "| --- | " + " | ".join("---" for _ in SELECTION_POLICIES) + " |",
        ]
    )
    for budget in budgets:
        values = [
            _format_harm_waste(seed_curves, budget, policy)
            for policy, _ in SELECTION_POLICIES
        ]
        lines.append(f"| {_format_float(budget)} | " + " | ".join(values) + " |")

    lines.extend(["", ""])
    return lines


def _format_mean_std(row: dict, metric: str) -> str:
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    if mean_key not in row:
        return "-"
    return f"{_format_float(row[mean_key])} +/- {_format_float(row.get(std_key, 0.0))}"


def _format_seed_stat(seed_curves: list[list[dict]], budget: float, extractor: Callable[[dict], float]) -> str:
    values = []
    for curve in seed_curves:
        row = _find_budget_row(curve, budget)
        if row is not None:
            values.append(float(extractor(row)))
    if not values:
        return "-"
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return f"{_format_float(mean)} +/- {_format_float(variance ** 0.5)}"


def _format_harm_waste(seed_curves: list[list[dict]], budget: float, policy: str) -> str:
    harmful = _format_seed_stat(
        seed_curves,
        budget,
        lambda row: row[f"{policy}_selection"]["harmful_selected"],
    )
    wasted = _format_seed_stat(
        seed_curves,
        budget,
        lambda row: row[f"{policy}_selection"]["wasted_selected"],
    )
    return f"H {harmful}, W {wasted}"


def _find_budget_row(curve: list[dict], budget: float) -> dict | None:
    for row in curve:
        if abs(float(row["budget_fraction"]) - float(budget)) < 1e-9:
            return row
    return None


def _parse_budgets(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("at least one budget must be provided")
    return budgets


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
