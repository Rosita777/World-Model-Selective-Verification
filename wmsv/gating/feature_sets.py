from __future__ import annotations


FEATURE_SETS = {
    "score": [
        "score_margin",
        "cheap_score",
    ],
    "uncertainty": [
        "uncertainty_proxy",
        "ensemble_action_disagreement",
        "ensemble_score_variance",
        "ensemble_plan_disagreement",
    ],
    "plan": [
        "cheap_plan_length",
        "cheap_plan_turns",
        "cheap_plan_unique_actions",
        "cheap_plan_score_per_step",
    ],
    "trajectory": [
        "cheap_plan_final_progress",
        "cheap_plan_state_change_fraction",
        "cheap_plan_box_change_fraction",
    ],
    "impact": [
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
    ],
}

FEATURE_SETS["all"] = [
    *FEATURE_SETS["score"],
    *FEATURE_SETS["uncertainty"],
    *FEATURE_SETS["plan"],
    *FEATURE_SETS["trajectory"],
    *FEATURE_SETS["impact"],
]


def available_features(rows: list[dict], feature_set: str = "all") -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}")
    return [name for name in FEATURE_SETS[feature_set] if all(name in row for row in rows)]


def parse_feature_set_list(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one feature set must be provided")
    for name in names:
        if name not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {name!r}")
    return names
