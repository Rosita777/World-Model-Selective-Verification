from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_stage_a_smoke import parse_budget_list, policy_return_vectors
from wmsv.analysis.bootstrap import bootstrap_mean_ci, bootstrap_paired_delta_ci


def split_item_rows(item: dict) -> tuple[list[dict], list[dict]]:
    if item.get("budget_sweep"):
        train_count = int(item["budget_sweep"][0]["train_rows"])
    else:
        train_count = int(item["ranker"]["train_rows"])
    rows = item["rows"]
    return rows[:train_count], rows[train_count:]


def summarize_ci_for_item(
    item: dict,
    budget_fraction: float,
    samples: int = 1000,
    seed: int = 0,
    random_seed: int = 0,
) -> dict:
    train_rows, eval_rows = split_item_rows(item)
    vectors = policy_return_vectors(
        train_rows,
        eval_rows,
        budget_fraction=budget_fraction,
        random_seed=random_seed,
    )
    policy_ci = {
        name: bootstrap_mean_ci(values, samples=samples, seed=seed)
        for name, values in vectors.items()
    }
    delta_ci = {}
    if "decision" in vectors:
        for baseline in ["random", "uncertainty", "think_longer", "uniform_true", "cheap"]:
            delta_ci[f"decision_minus_{baseline}"] = bootstrap_paired_delta_ci(
                vectors["decision"],
                vectors[baseline],
                samples=samples,
                seed=seed,
            )
    return {
        "push_error_rate": item["push_error_rate"],
        "budget_fraction": budget_fraction,
        "eval_rows": len(eval_rows),
        "policy_ci": policy_ci,
        "delta_ci": delta_ci,
    }


def summarize_file(
    path: str | Path,
    budgets: list[float],
    rates: list[float] | None = None,
    samples: int = 1000,
    seed: int = 0,
    random_seed: int = 0,
) -> list[dict]:
    data = json.loads(Path(path).read_text())
    summaries = []
    for item in data:
        if rates is not None and float(item["push_error_rate"]) not in rates:
            continue
        for budget in budgets:
            summaries.append(
                summarize_ci_for_item(
                    item,
                    budget_fraction=budget,
                    samples=samples,
                    seed=seed,
                    random_seed=random_seed,
                )
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--budgets", default="0.25")
    parser.add_argument("--rates", default=None)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    rates = parse_budget_list(args.rates) if args.rates else None
    summaries = summarize_file(
        args.path,
        budgets=parse_budget_list(args.budgets),
        rates=rates,
        samples=args.samples,
        seed=args.seed,
        random_seed=args.random_seed,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
