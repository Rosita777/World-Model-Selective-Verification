from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_boxoban_learned_pilot import _split_rows, _success_rate
from wmsv.analysis.budget_curves import evaluate_budget_curve
from wmsv.analysis.budgeting import mean_return_for_mask, threshold_budget_mask
from wmsv.analysis.verification_value import auroc, positive_label_rate
from wmsv.gating.dive import DIVEFeatureSchema, fit_dive_v0, fit_value_rank_gate
from wmsv.gating.feature_sets import available_features, parse_feature_set_list
from wmsv.pilots.boxoban import build_boxoban_pilot_rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--levels-path", default="data/external/boxoban-sample/medium/train")
    parser.add_argument("--feature-sets", default="score,uncertainty,plan,trajectory,impact,all")
    parser.add_argument("--budgets", default="0,0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = build_boxoban_pilot_rows(
        limit=args.limit,
        seed=args.seed,
        levels_path=args.levels_path,
    )
    feature_sets = parse_feature_set_list(args.feature_sets)
    budgets = _parse_budgets(args.budgets)
    train_rows, eval_rows = _split_rows(rows)
    report = {
        "environment": "boxoban",
        "num_rows": len(rows),
        "feature_set_results": [
            _feature_set_report(
                feature_set=name,
                rows=rows,
                train_rows=train_rows,
                eval_rows=eval_rows,
                budgets=budgets,
                random_seed=args.seed,
            )
            for name in feature_sets
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _feature_set_report(
    feature_set: str,
    rows: list[dict],
    train_rows: list[dict],
    eval_rows: list[dict],
    budgets: list[float],
    random_seed: int,
) -> dict:
    feature_names = available_features(rows, feature_set)
    curve = evaluate_budget_curve(
        train_rows=train_rows,
        eval_rows=eval_rows,
        feature_names=feature_names,
        budgets=budgets,
        random_seed=random_seed,
    )
    return {
        "feature_set": feature_set,
        "feature_names": feature_names,
        "summary": _summary_for_features(train_rows, eval_rows, feature_names),
        "budget_curve": curve,
    }


def _summary_for_features(train_rows: list[dict], eval_rows: list[dict], feature_names: list[str]) -> dict:
    schema = DIVEFeatureSchema(feature_names)
    gate = fit_dive_v0(train_rows, schema)
    value_rank_gate = fit_value_rank_gate(train_rows, schema)
    scores = [gate.score(row) for row in eval_rows]
    value_rank_scores = [value_rank_gate.score(row) for row in eval_rows]
    mask20 = threshold_budget_mask(scores, 0.20)
    value_rank_mask20 = threshold_budget_mask(value_rank_scores, 0.20)
    random_mask20 = [(idx % 5) == 0 for idx in range(len(eval_rows))]
    return {
        "cheap_return": sum(float(row["r_c"]) for row in eval_rows) / len(eval_rows),
        "always_verify_return": sum(float(row["r_v"]) for row in eval_rows) / len(eval_rows),
        "cheap_success": _success_rate(eval_rows, "r_c", "boxoban"),
        "always_verify_success": _success_rate(eval_rows, "r_v", "boxoban"),
        "dive_budget20_return": mean_return_for_mask(eval_rows, mask20),
        "value_rank_budget20_return": mean_return_for_mask(eval_rows, value_rank_mask20),
        "random_budget20_return": mean_return_for_mask(eval_rows, random_mask20),
        "positive_label_rate": positive_label_rate(eval_rows),
        "helpful_auroc": auroc(scores, [int(row["y_helpful"]) for row in eval_rows]),
    }


def _parse_budgets(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("at least one budget must be provided")
    for budget in budgets:
        if not 0.0 <= budget <= 1.0:
            raise ValueError("budgets must be in [0, 1]")
    return budgets


if __name__ == "__main__":
    main()
