import math

from wmsv.gating.dive import DIVEFeatureSchema, fit_dive_v0, fit_uncertainty_label_gate, fit_value_rank_gate


ROWS = [
    {
        "score_margin": 0.1,
        "ensemble_uncertainty": 0.8,
        "cheap_score": 0.1,
        "y_change": 1,
        "y_helpful": 1,
        "y_harm": 0,
        "y_waste": 0,
        "delta_r": 0.8,
        "uncertainty_label": 1,
    },
    {
        "score_margin": 0.2,
        "ensemble_uncertainty": 0.7,
        "cheap_score": 0.2,
        "y_change": 1,
        "y_helpful": 1,
        "y_harm": 0,
        "y_waste": 0,
        "delta_r": 0.7,
        "uncertainty_label": 1,
    },
    {
        "score_margin": 1.0,
        "ensemble_uncertainty": 0.1,
        "cheap_score": 0.9,
        "y_change": 0,
        "y_helpful": 0,
        "y_harm": 0,
        "y_waste": 1,
        "delta_r": 0.0,
        "uncertainty_label": 0,
    },
    {
        "score_margin": 1.1,
        "ensemble_uncertainty": 0.2,
        "cheap_score": 0.8,
        "y_change": 1,
        "y_helpful": 0,
        "y_harm": 1,
        "y_waste": 0,
        "delta_r": -0.5,
        "uncertainty_label": 0,
    },
]


def test_dive_scores_helpful_rows_above_wasted_rows():
    schema = DIVEFeatureSchema(["score_margin", "ensemble_uncertainty", "cheap_score"])
    gate = fit_dive_v0(ROWS, schema)

    helpful = gate.score({"score_margin": 0.15, "ensemble_uncertainty": 0.75, "cheap_score": 0.1})
    wasted = gate.score({"score_margin": 1.05, "ensemble_uncertainty": 0.15, "cheap_score": 0.9})

    assert helpful > wasted


def test_dive_predict_heads_returns_named_outputs():
    schema = DIVEFeatureSchema(["score_margin", "ensemble_uncertainty", "cheap_score"])
    gate = fit_dive_v0(ROWS, schema)

    heads = gate.predict_heads({"score_margin": 0.15, "ensemble_uncertainty": 0.75, "cheap_score": 0.1})

    assert set(heads) == {"p_change", "delta_hat", "p_harm", "p_waste"}
    assert 0.0 <= heads["p_change"] <= 1.0
    assert 0.0 <= heads["p_harm"] <= 1.0
    assert 0.0 <= heads["p_waste"] <= 1.0
    assert math.isfinite(heads["delta_hat"])


def test_uncertainty_label_gate_uses_same_schema():
    schema = DIVEFeatureSchema(["score_margin", "ensemble_uncertainty", "cheap_score"])
    gate = fit_uncertainty_label_gate(ROWS, schema)

    high = gate.score({"score_margin": 0.15, "ensemble_uncertainty": 0.75, "cheap_score": 0.1})
    low = gate.score({"score_margin": 1.05, "ensemble_uncertainty": 0.15, "cheap_score": 0.9})

    assert high > low


def test_value_rank_gate_prioritizes_larger_predicted_improvements():
    rows = [
        {
            **row,
            "value_signal": value_signal,
            "delta_r": delta_r,
        }
        for row, value_signal, delta_r in [
            (ROWS[0], 1.0, 1.0),
            (ROWS[1], 0.8, 0.7),
            (ROWS[2], 0.1, 0.0),
            (ROWS[3], -0.3, -0.5),
        ]
    ]
    schema = DIVEFeatureSchema(["value_signal"])
    gate = fit_value_rank_gate(rows, schema)

    high = gate.score({"value_signal": 1.0})
    mid = gate.score({"value_signal": 0.8})
    waste = gate.score({"value_signal": 0.1})
    harmful = gate.score({"value_signal": -0.3})

    assert high > mid > waste
    assert harmful == 0.0
