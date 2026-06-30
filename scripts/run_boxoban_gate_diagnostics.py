from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_boxoban_learned_pilot import _split_rows
from wmsv.analysis.gate_diagnostics import compare_ranked_selections, feature_correlations
from wmsv.gating.dive import DIVEFeatureSchema, fit_dive_v0
from wmsv.gating.feature_sets import available_features
from wmsv.pilots.boxoban import build_boxoban_pilot_rows


DIAGNOSTIC_FEATURES = [
    "cheap_plan_uncertainty_mean",
    "cheap_plan_uncertainty_max",
    "cheap_plan_early_uncertainty",
    "cheap_plan_deadlock_fraction",
    "decision_instability",
    "impact_uncertainty",
    "irreversibility_risk",
    "counterfactual_action_gap",
    "counterfactual_gap_confidence",
    "temporal_consistency",
    "temporal_inconsistency",
]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--levels-path", default="data/external/boxoban-sample/medium/train")
    parser.add_argument("--budgets", default="0.1,0.2,0.5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = build_boxoban_pilot_rows(
        limit=args.limit,
        seed=args.seed,
        levels_path=args.levels_path,
    )
    train_rows, eval_rows = _split_rows(rows)
    feature_names = available_features(rows, "all")
    gate = fit_dive_v0(train_rows, DIVEFeatureSchema(feature_names))
    dive_scores = [gate.score(row) for row in eval_rows]
    uncertainty_scores = [float(row["ensemble_uncertainty"]) for row in eval_rows]
    budgets = _parse_budgets(args.budgets)
    report = {
        "environment": "boxoban",
        "num_rows": len(rows),
        "feature_names": feature_names,
        "correlations": feature_correlations(
            eval_rows,
            reference="ensemble_uncertainty",
            features=[name for name in DIAGNOSTIC_FEATURES if all(name in row for row in eval_rows)],
        ),
        "selection_comparisons": [
            compare_ranked_selections(
                eval_rows,
                left_scores=dive_scores,
                right_scores=uncertainty_scores,
                budget_fraction=budget,
            )
            for budget in budgets
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


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
