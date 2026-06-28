from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from wmsv.analysis.feasibility import summarize_labels
from wmsv.analysis.stage_a import evaluate_first_action, make_label_row
from wmsv.analysis.uncertainty import ensemble_uncertainty
from wmsv.data.boxoban import iter_boxoban_levels
from wmsv.envs.sokoban import parse_level
from wmsv.gating.simple import fit_centroid_gate, mean_policy_return, selection_summary, top_fraction_mask
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
    uncertainty_seeds: int = 1,
    think_longer_depth: int | None = None,
    think_longer_width: int | None = None,
) -> list[dict]:
    think_longer_depth = think_longer_depth or cheap_depth * 2
    think_longer_width = think_longer_width or cheap_width * 2
    cheap = BeamPlanner(
        PotentialEvaluator(DegradedPushEvaluator(
            push_error_rate=push_error_rate,
            corrupt_push_penalty=corrupt_push_penalty,
        )),
        depth=cheap_depth,
        width=cheap_width,
    )
    think_longer = BeamPlanner(
        PotentialEvaluator(DegradedPushEvaluator(
            push_error_rate=push_error_rate,
            corrupt_push_penalty=corrupt_push_penalty,
        )),
        depth=think_longer_depth,
        width=think_longer_width,
    )
    verifier = BeamPlanner(PotentialEvaluator(TrueEvaluator()), depth=verifier_depth, width=verifier_width)
    evaluator = BeamPlanner(PotentialEvaluator(TrueEvaluator()), depth=eval_depth, width=eval_width)
    uncertainty_planners = [
        BeamPlanner(
            PotentialEvaluator(DegradedPushEvaluator(
                push_error_rate=push_error_rate,
                seed=seed,
                corrupt_push_penalty=corrupt_push_penalty,
            )),
            depth=cheap_depth,
            width=cheap_width,
        )
        for seed in range(max(1, uncertainty_seeds))
    ]
    rows = []
    for level_id, level in load_levels(levels_folder, limit):
        state = parse_level(level)
        row = make_label_row(level_id, state, cheap, verifier, evaluator)
        think_longer_result = think_longer.plan(state)
        row["a_t"] = think_longer_result.action
        row["r_t"] = evaluate_first_action(state, think_longer_result.action, evaluator)
        row["think_longer_score"] = think_longer_result.score
        row["think_longer_nodes"] = think_longer_result.nodes_expanded
        uncertainty = ensemble_uncertainty(state, uncertainty_planners)
        row["ensemble_action_disagreement"] = uncertainty["action_disagreement"]
        row["ensemble_score_variance"] = uncertainty["score_variance"]
        row["ensemble_num_planners"] = uncertainty["num_planners"]
        row["ensemble_uncertainty"] = (
            row["ensemble_action_disagreement"] + row["ensemble_score_variance"]
        )
        rows.append(row)
    return rows


def parse_budget_list(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("at least one budget must be provided")
    for budget in budgets:
        if not 0.0 <= budget <= 1.0:
            raise ValueError("budgets must be in [0, 1]")
    return budgets


def split_rows(rows: list[dict], train_fraction: float) -> tuple[list[dict], list[dict]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    split = int(round(len(rows) * train_fraction))
    split = min(max(1, split), max(1, len(rows) - 1))
    return rows[:split], rows[split:]


def mean_policy_nodes(rows: list[dict], verify_mask: list[bool]) -> float:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have the same length")
    if not rows:
        return 0.0
    total = 0.0
    for row, verify in zip(rows, verify_mask):
        nodes = float(row["cheap_nodes"])
        if verify:
            nodes += float(row["verifier_nodes"])
        total += nodes
    return total / len(rows)


def mean_think_longer_nodes(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get("think_longer_nodes", row["cheap_nodes"])) for row in rows) / len(rows)


def masked_return_values(rows: list[dict], verify_mask: list[bool]) -> list[float]:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have the same length")
    return [
        float(row["r_v"] if verify else row["r_c"])
        for row, verify in zip(rows, verify_mask)
    ]


def policy_return_vectors(
    train_rows: list[dict],
    eval_rows: list[dict],
    budget_fraction: float,
    random_seed: int = 0,
) -> dict[str, list[float]]:
    uncertainty_scores = [
        float(row.get("ensemble_uncertainty", row["uncertainty_proxy"]))
        for row in eval_rows
    ]
    uncertainty_mask = top_fraction_mask(uncertainty_scores, budget_fraction)
    rng = random.Random(random_seed)
    random_mask = top_fraction_mask([rng.random() for _ in eval_rows], budget_fraction)
    oracle_scores = [
        float(row["r_v"]) - float(row["r_c"])
        for row in eval_rows
    ]
    oracle_mask = top_fraction_mask(oracle_scores, budget_fraction)
    vectors = {
        "cheap": [float(row["r_c"]) for row in eval_rows],
        "always": [float(row["r_v"]) for row in eval_rows],
        "think_longer": [float(row.get("r_t", row["r_c"])) for row in eval_rows],
        "uncertainty": masked_return_values(eval_rows, uncertainty_mask),
        "random": masked_return_values(eval_rows, random_mask),
        "oracle": masked_return_values(eval_rows, oracle_mask),
    }
    if any(row["label"] == 1 for row in train_rows) and any(row["label"] == 0 for row in train_rows):
        gate = fit_centroid_gate(
            train_rows,
            [
                "score_margin",
                "uncertainty_proxy",
                "cheap_score",
                "ensemble_action_disagreement",
                "ensemble_score_variance",
            ],
        )
        decision_scores = [gate.score(row) for row in eval_rows]
        decision_mask = top_fraction_mask(decision_scores, budget_fraction)
        vectors["decision"] = masked_return_values(eval_rows, decision_mask)
    return vectors


def evaluate_rankers(
    train_rows: list[dict],
    eval_rows: list[dict],
    budget_fraction: float,
    random_seed: int = 0,
) -> dict:
    cheap_return = mean_policy_return(eval_rows, [False] * len(eval_rows))
    always_return = mean_policy_return(eval_rows, [True] * len(eval_rows))
    think_longer_return = (
        sum(float(row.get("r_t", row["r_c"])) for row in eval_rows) / len(eval_rows)
        if eval_rows else 0.0
    )
    uncertainty_scores = [
        float(row.get("ensemble_uncertainty", row["uncertainty_proxy"]))
        for row in eval_rows
    ]
    uncertainty_mask = top_fraction_mask(uncertainty_scores, budget_fraction)
    uncertainty_return = mean_policy_return(eval_rows, uncertainty_mask)
    rng = random.Random(random_seed)
    random_mask = top_fraction_mask([rng.random() for _ in eval_rows], budget_fraction)
    random_return = mean_policy_return(eval_rows, random_mask)
    oracle_scores = [
        float(row["r_v"]) - float(row["r_c"])
        for row in eval_rows
    ]
    oracle_mask = top_fraction_mask(oracle_scores, budget_fraction)
    oracle_return = mean_policy_return(eval_rows, oracle_mask)

    result = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "cheap_return": cheap_return,
        "always_return": always_return,
        "think_longer_return": think_longer_return,
        "uncertainty_return": uncertainty_return,
        "random_return": random_return,
        "oracle_return": oracle_return,
        "decision_return": None,
        "uncertainty_selection": selection_summary(eval_rows, uncertainty_mask),
        "random_selection": selection_summary(eval_rows, random_mask),
        "oracle_selection": selection_summary(eval_rows, oracle_mask),
        "decision_selection": None,
        "cheap_nodes": mean_policy_nodes(eval_rows, [False] * len(eval_rows)),
        "always_nodes": mean_policy_nodes(eval_rows, [True] * len(eval_rows)),
        "think_longer_nodes": mean_think_longer_nodes(eval_rows),
        "uncertainty_nodes": mean_policy_nodes(eval_rows, uncertainty_mask),
        "random_nodes": mean_policy_nodes(eval_rows, random_mask),
        "oracle_nodes": mean_policy_nodes(eval_rows, oracle_mask),
        "decision_nodes": None,
    }
    if any(row["label"] == 1 for row in train_rows) and any(row["label"] == 0 for row in train_rows):
        gate = fit_centroid_gate(
            train_rows,
            [
                "score_margin",
                "uncertainty_proxy",
                "cheap_score",
                "ensemble_action_disagreement",
                "ensemble_score_variance",
            ],
        )
        decision_scores = [gate.score(row) for row in eval_rows]
        decision_mask = top_fraction_mask(decision_scores, budget_fraction)
        result["decision_return"] = mean_policy_return(eval_rows, decision_mask)
        result["decision_selection"] = selection_summary(eval_rows, decision_mask)
        result["decision_nodes"] = mean_policy_nodes(eval_rows, decision_mask)
    return result


def run_budget_sweep(
    train_rows: list[dict],
    eval_rows: list[dict],
    budgets: list[float],
    random_seed: int = 0,
) -> list[dict]:
    sweep = []
    for budget in budgets:
        result = evaluate_rankers(
            train_rows,
            eval_rows,
            budget,
            random_seed=random_seed,
        )
        result["budget_fraction"] = budget
        sweep.append(result)
    return sweep


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
    parser.add_argument("--uncertainty-seeds", type=int, default=5)
    parser.add_argument("--think-longer-depth", type=int, default=None)
    parser.add_argument("--think-longer-width", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--budgets", default=None)
    args = parser.parse_args()
    budgets = parse_budget_list(args.budgets) if args.budgets else [args.budget]

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
            uncertainty_seeds=args.uncertainty_seeds,
            think_longer_depth=args.think_longer_depth,
            think_longer_width=args.think_longer_width,
        )
        train_rows, eval_rows = split_rows(rows, args.train_fraction)
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
                "ranker": evaluate_rankers(
                    train_rows,
                    eval_rows,
                    args.budget,
                    random_seed=args.random_seed,
                ),
                "budget_sweep": run_budget_sweep(
                    train_rows,
                    eval_rows,
                    budgets,
                    random_seed=args.random_seed,
                ),
                "rows": rows,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
