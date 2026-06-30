from wmsv.analysis.gate_diagnostics import feature_correlations, compare_ranked_selections


ROWS = [
    {"r_c": 0.0, "r_v": 1.0, "y_helpful": 1, "uncertainty": 0.1, "impact": 0.2},
    {"r_c": 0.0, "r_v": 0.8, "y_helpful": 1, "uncertainty": 0.2, "impact": 0.4},
    {"r_c": 0.5, "r_v": 0.5, "y_helpful": 0, "uncertainty": 0.9, "impact": 0.6},
    {"r_c": 0.5, "r_v": 0.0, "y_helpful": 0, "uncertainty": 1.0, "impact": 0.8},
]


def test_feature_correlations_reports_reference_correlations():
    result = feature_correlations(
        ROWS,
        reference="uncertainty",
        features=["impact"],
    )

    assert result["reference"] == "uncertainty"
    assert result["correlations"]["impact"] > 0.0


def test_compare_ranked_selections_reports_overlap_and_improvement():
    result = compare_ranked_selections(
        ROWS,
        left_scores=[1.0, 0.9, 0.1, 0.0],
        right_scores=[0.0, 0.1, 0.9, 1.0],
        budget_fraction=0.5,
    )

    assert result["budget_fraction"] == 0.5
    assert result["left_selected"] == 2
    assert result["right_selected"] == 2
    assert result["overlap"] == 0
    assert result["left_mean_improvement"] > result["right_mean_improvement"]
    assert result["left_helpful_precision"] == 1.0
