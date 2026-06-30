from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmsv.analysis.budgeting import mean_return_for_mask, threshold_budget_mask
from wmsv.analysis.verification_value import auroc, go_no_go_status, positive_label_rate
from wmsv.gating.dive import DIVEFeatureSchema, fit_dive_v0
from wmsv.gating.feature_sets import available_features
from wmsv.pilots.boxoban import build_boxoban_pilot_rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--levels-path", default="data/external/boxoban-sample/medium/train")
    parser.add_argument("--train-level-count", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = build_boxoban_pilot_rows(
        limit=args.limit,
        levels_path=args.levels_path,
        train_level_count=args.train_level_count,
    )
    report = _report(rows, environment="boxoban", min_always_gain=0.12, min_auroc=0.62)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _report(rows: list[dict], environment: str, min_always_gain: float, min_auroc: float) -> dict:
    train, eval_rows = _split_rows(rows)
    schema = DIVEFeatureSchema(_available_features(rows))
    gate = fit_dive_v0(train, schema)
    scores = [gate.score(row) for row in eval_rows]
    mask20 = threshold_budget_mask(scores, 0.20)
    random_mask20 = [(idx % 5) == 0 for idx in range(len(eval_rows))]
    cheap_return = sum(float(row["r_c"]) for row in eval_rows) / len(eval_rows)
    always_return = sum(float(row["r_v"]) for row in eval_rows) / len(eval_rows)
    cheap_success = _success_rate(eval_rows, "r_c", environment)
    always_success = _success_rate(eval_rows, "r_v", environment)
    dive_return = mean_return_for_mask(eval_rows, mask20)
    random_return = mean_return_for_mask(eval_rows, random_mask20)
    helpful_auroc = auroc(scores, [int(row["y_helpful"]) for row in eval_rows])
    status = go_no_go_status(
        cheap_success=cheap_success,
        always_verify_success=always_success,
        positive_label_rate=positive_label_rate(eval_rows),
        helpful_auroc=helpful_auroc,
        budget20_gain=dive_return - random_return,
        min_always_gain=min_always_gain,
        min_auroc=min_auroc,
    )
    return {
        "environment": environment,
        "num_rows": len(rows),
        "cheap_return": cheap_return,
        "always_verify_return": always_return,
        "cheap_success": cheap_success,
        "always_verify_success": always_success,
        "dive_budget20_return": dive_return,
        "random_budget20_return": random_return,
        "positive_label_rate": positive_label_rate(eval_rows),
        "helpful_auroc": helpful_auroc,
        "go_no_go": status,
    }


def _available_features(rows: list[dict]) -> list[str]:
    return available_features(rows, "all")


def _split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [row for idx, row in enumerate(rows) if idx % 2 == 0]
    eval_rows = [row for idx, row in enumerate(rows) if idx % 2 == 1]
    if not train or not eval_rows:
        split = max(1, len(rows) // 2)
        train = rows[:split]
        eval_rows = rows[split:] or rows
    return train, eval_rows


def _success_rate(rows: list[dict], key: str, environment: str) -> float:
    if not rows:
        return 0.0
    if environment == "maze":
        return sum(float(row[key]) >= 0.0 for row in rows) / len(rows)
    return sum(float(row[key]) > 0.0 for row in rows) / len(rows)


if __name__ == "__main__":
    main()
