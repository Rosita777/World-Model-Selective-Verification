from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_boxoban_learned_pilot import _available_features, _report, _split_rows
from wmsv.analysis.budget_curves import aggregate_budget_curves, evaluate_budget_curve
from wmsv.pilots.boxoban import build_boxoban_pilot_rows
from wmsv.pilots.maze import build_maze_pilot_rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxoban-limit", type=int, default=200)
    parser.add_argument("--boxoban-levels-path", default="data/external/boxoban-sample/medium/train")
    parser.add_argument("--maze-variants", type=int, default=3)
    parser.add_argument("--maze-episodes-per-variant", type=int, default=100)
    parser.add_argument("--budgets", default="0,0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    budgets = _parse_budgets(args.budgets)
    seeds = _parse_seeds(args.seeds) if args.seeds else [int(args.seed)]
    report = {
        "environments": ["boxoban", "maze"],
        "seeds": seeds,
        "boxoban": _multi_seed_environment_report(
            rows_builder=lambda seed: build_boxoban_pilot_rows(
                limit=args.boxoban_limit,
                seed=seed,
                levels_path=args.boxoban_levels_path,
            ),
            environment="boxoban",
            budgets=budgets,
            seeds=seeds,
            min_always_gain=0.12,
            min_auroc=0.62,
        ),
        "maze": _multi_seed_environment_report(
            rows_builder=lambda seed: build_maze_pilot_rows(
                variants=args.maze_variants,
                episodes_per_variant=args.maze_episodes_per_variant,
                seed=seed,
            ),
            environment="maze",
            budgets=budgets,
            seeds=seeds,
            min_always_gain=0.10,
            min_auroc=0.60,
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _multi_seed_environment_report(
    rows_builder,
    environment: str,
    budgets: list[float],
    seeds: list[int],
    min_always_gain: float,
    min_auroc: float,
) -> dict:
    seed_runs = [
        {
            "seed": seed,
            **_environment_report(
                rows_builder(seed),
                environment=environment,
                budgets=budgets,
                random_seed=seed,
                min_always_gain=min_always_gain,
                min_auroc=min_auroc,
            ),
        }
        for seed in seeds
    ]
    first = seed_runs[0]
    return {
        "summary": first["summary"],
        "budget_curve": first["budget_curve"],
        "seed_runs": seed_runs,
        "aggregate_budget_curve": aggregate_budget_curves(
            [run["budget_curve"] for run in seed_runs],
            metrics=[
                "dive_return",
                "value_rank_return",
                "risk_aware_value_return",
                "random_return",
                "uncertainty_return",
                "oracle_return",
                "think_longer_return",
                "uniform_true_return",
                "dive_nodes",
                "value_rank_nodes",
                "risk_aware_value_nodes",
                "think_longer_nodes",
                "uniform_true_nodes",
            ],
        ),
    }


def _environment_report(
    rows: list[dict],
    environment: str,
    budgets: list[float],
    random_seed: int,
    min_always_gain: float,
    min_auroc: float,
) -> dict:
    train_rows, eval_rows = _split_rows(rows)
    return {
        "summary": _report(
            rows,
            environment=environment,
            min_always_gain=min_always_gain,
            min_auroc=min_auroc,
        ),
        "budget_curve": evaluate_budget_curve(
            train_rows=train_rows,
            eval_rows=eval_rows,
            feature_names=_available_features(rows),
            budgets=budgets,
            random_seed=random_seed,
        ),
    }


def _parse_budgets(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("at least one budget must be provided")
    for budget in budgets:
        if not 0.0 <= budget <= 1.0:
            raise ValueError("budgets must be in [0, 1]")
    return budgets


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed must be provided")
    return seeds


if __name__ == "__main__":
    main()
