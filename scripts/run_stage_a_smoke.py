from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmsv.analysis.feasibility import summarize_labels
from wmsv.analysis.stage_a import make_label_row
from wmsv.data.boxoban import iter_boxoban_levels
from wmsv.envs.sokoban import parse_level
from wmsv.gating.simple import fit_centroid_gate, mean_policy_return, top_fraction_mask
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import DegradedPushEvaluator, PotentialEvaluator, TrueEvaluator


TINY_LEVELS = [
    [
        "#####",
        "#@$.#",
        "#   #",
        "#####",
    ],
    [
        "######",
        "#@ $.#",
        "#    #",
        "######",
    ],
    [
        "######",
        "#@ $ #",
        "#   .#",
        "######",
    ],
    [
        "#######",
        "#@ $ .#",
        "#     #",
        "#######",
    ],
]


def load_levels(levels_folder: str | Path | None, limit: int | None) -> list[tuple[str, list[str]]]:
    if levels_folder is None:
        return [(f"tiny-{idx}", level) for idx, level in enumerate(TINY_LEVELS)]
    return [
        (level.level_id, level.lines)
        for level in iter_boxoban_levels(levels_folder, limit=limit)
    ]


def build_rows(
    push_error_rate: float,
    corrupt_push_penalty: float,
    levels_folder: str | Path | None = None,
    limit: int | None = None,
    cheap_depth: int = 3,
    cheap_width: int = 8,
    verifier_depth: int = 5,
    verifier_width: int = 16,
    eval_depth: int = 5,
    eval_width: int = 16,
) -> list[dict]:
    cheap = BeamPlanner(
        PotentialEvaluator(DegradedPushEvaluator(
            push_error_rate=push_error_rate,
            corrupt_push_penalty=corrupt_push_penalty,
        )),
        depth=cheap_depth,
        width=cheap_width,
    )
    verifier = BeamPlanner(PotentialEvaluator(TrueEvaluator()), depth=verifier_depth, width=verifier_width)
    evaluator = BeamPlanner(PotentialEvaluator(TrueEvaluator()), depth=eval_depth, width=eval_width)
    rows = []
    for level_id, level in load_levels(levels_folder, limit):
        rows.append(make_label_row(level_id, parse_level(level), cheap, verifier, evaluator))
    return rows


def evaluate_rankers(rows: list[dict], budget_fraction: float) -> dict:
    cheap_return = mean_policy_return(rows, [False] * len(rows))
    always_return = mean_policy_return(rows, [True] * len(rows))
    uncertainty_scores = [float(row["uncertainty_proxy"]) for row in rows]
    uncertainty_mask = top_fraction_mask(uncertainty_scores, budget_fraction)
    uncertainty_return = mean_policy_return(rows, uncertainty_mask)

    result = {
        "cheap_return": cheap_return,
        "always_return": always_return,
        "uncertainty_return": uncertainty_return,
        "decision_return": None,
    }
    if any(row["label"] == 1 for row in rows) and any(row["label"] == 0 for row in rows):
        gate = fit_centroid_gate(rows, ["score_margin", "uncertainty_proxy", "cheap_score"])
        decision_scores = [gate.score(row) for row in rows]
        decision_mask = top_fraction_mask(decision_scores, budget_fraction)
        result["decision_return"] = mean_policy_return(rows, decision_mask)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/boxoban_kill_or_continue_v0/stage_a_smoke.json")
    parser.add_argument("--levels", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--cheap-depth", type=int, default=3)
    parser.add_argument("--cheap-width", type=int, default=8)
    parser.add_argument("--verifier-depth", type=int, default=5)
    parser.add_argument("--verifier-width", type=int, default=16)
    parser.add_argument("--eval-depth", type=int, default=5)
    parser.add_argument("--eval-width", type=int, default=16)
    args = parser.parse_args()

    summary = []
    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rows = build_rows(
            rate,
            args.penalty,
            levels_folder=args.levels,
            limit=args.limit,
            cheap_depth=args.cheap_depth,
            cheap_width=args.cheap_width,
            verifier_depth=args.verifier_depth,
            verifier_width=args.verifier_width,
            eval_depth=args.eval_depth,
            eval_width=args.eval_width,
        )
        counts = summarize_labels(rows, epsilon=0.01)
        summary.append(
            {
                "push_error_rate": rate,
                "counts": {
                    "total": counts.total,
                    "helpful": counts.helpful,
                    "harmful": counts.harmful,
                    "wasted": counts.wasted,
                    "spurious": counts.spurious,
                    "helpful_rate": counts.helpful_rate,
                    "harmful_rate": counts.harmful_rate,
                    "wasted_rate": counts.wasted_rate,
                },
                "ranker": evaluate_rankers(rows, args.budget),
                "rows": rows,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
