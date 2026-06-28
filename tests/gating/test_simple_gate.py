from wmsv.gating.simple import (
    CentroidGate,
    fit_centroid_gate,
    mean_policy_return,
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


def test_centroid_gate_requires_positive_and_negative_examples():
    try:
        fit_centroid_gate([{"score_margin": 0.1, "label": 1}], feature_names=["score_margin"])
    except ValueError as exc:
        assert "positive and negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

