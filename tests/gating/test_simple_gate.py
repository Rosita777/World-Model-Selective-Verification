from wmsv.gating.simple import (
    CentroidGate,
    fit_centroid_gate,
    mean_policy_return,
    selection_summary,
    top_fraction_mask,
)


ROWS = [
    {"score_margin": 0.1, "uncertainty_proxy": 0.2, "r_c": 0.0, "r_v": 1.0, "label": 1},
    {"score_margin": 0.2, "uncertainty_proxy": 0.1, "r_c": 0.0, "r_v": 1.0, "label": 1},
    {"score_margin": 1.2, "uncertainty_proxy": 0.9, "r_c": 1.0, "r_v": 1.0, "label": 0},
    {"score_margin": 1.1, "uncertainty_proxy": 0.8, "r_c": 1.0, "r_v": 0.0, "label": 0},
]


def test_centroid_gate_scores_positive_like_rows_higher():
    gate = fit_centroid_gate(ROWS, feature_names=["score_margin"])

    positive_score = gate.score({"score_margin": 0.15})
    negative_score = gate.score({"score_margin": 1.15})

    assert positive_score > negative_score


def test_top_fraction_mask_selects_highest_scores():
    mask = top_fraction_mask([0.1, 0.9, 0.4, 0.2], fraction=0.5)

    assert mask == [False, True, True, False]


def test_mean_policy_return_uses_verified_return_when_masked():
    value = mean_policy_return(ROWS, [True, False, False, True])

    assert value == 0.5


def test_selection_summary_counts_selected_verification_outcomes():
    rows = [
        {"a_c": 0, "a_v": 1, "r_c": 0.0, "r_v": 1.0},
        {"a_c": 0, "a_v": 1, "r_c": 1.0, "r_v": 0.0},
        {"a_c": 0, "a_v": 1, "r_c": 0.5, "r_v": 0.5},
        {"a_c": 0, "a_v": 0, "r_c": 0.5, "r_v": 0.5},
    ]

    summary = selection_summary(rows, [True, True, False, True], epsilon=0.01)

    assert summary == {
        "selected": 3,
        "helpful_selected": 1,
        "harmful_selected": 1,
        "spurious_selected": 0,
        "wasted_selected": 1,
        "helpful_precision": 1 / 3,
    }


def test_centroid_gate_requires_positive_and_negative_examples():
    try:
        fit_centroid_gate([{"score_margin": 0.1, "label": 1}], feature_names=["score_margin"])
    except ValueError as exc:
        assert "positive and negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
