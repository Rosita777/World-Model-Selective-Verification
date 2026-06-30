from wmsv.gating.feature_sets import FEATURE_SETS, available_features, parse_feature_set_list


def test_feature_sets_expose_boxoban_gate_ablation_groups():
    assert "score_margin" in FEATURE_SETS["score"]
    assert "ensemble_action_disagreement" in FEATURE_SETS["uncertainty"]
    assert "cheap_plan_turns" in FEATURE_SETS["plan"]
    assert "cheap_plan_box_change_fraction" in FEATURE_SETS["trajectory"]
    assert "impact_uncertainty" in FEATURE_SETS["impact"]
    assert "counterfactual_action_gap" in FEATURE_SETS["impact"]
    assert "temporal_inconsistency" in FEATURE_SETS["impact"]
    assert set(FEATURE_SETS["score"]).issubset(FEATURE_SETS["all"])
    assert set(FEATURE_SETS["impact"]).issubset(FEATURE_SETS["all"])


def test_impact_feature_set_does_not_use_verifier_outputs():
    assert all("verifier" not in name for name in FEATURE_SETS["impact"])
    assert all(name not in {"a_v", "r_v", "delta_r", "label"} for name in FEATURE_SETS["impact"])


def test_available_features_filters_missing_columns_in_order():
    rows = [
        {"score_margin": 1.0, "cheap_score": 0.0},
        {"score_margin": 0.5, "cheap_score": 0.1},
    ]

    assert available_features(rows, "all") == ["score_margin", "cheap_score"]


def test_parse_feature_set_list_accepts_comma_separated_names():
    assert parse_feature_set_list("score, trajectory") == ["score", "trajectory"]
