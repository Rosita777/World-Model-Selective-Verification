import pytest

from wmsv.analysis.budget_curves import aggregate_budget_curves, evaluate_budget_curve


def _row(label: int, r_c: float, r_v: float, feature: float, uncertainty: float) -> dict:
    return {
        "a_c": 0,
        "a_v": 1 if label else 0,
        "r_c": r_c,
        "r_v": r_v,
        "label": label,
        "y_change": label,
        "y_helpful": label,
        "y_harm": 0,
        "y_waste": 1 - label,
        "delta_r": r_v - r_c,
        "feature": feature,
        "ensemble_uncertainty": uncertainty,
        "uncertainty_proxy": uncertainty,
        "cheap_nodes": 10,
        "verifier_nodes": 100,
        "r_t": 0.25,
        "think_longer_nodes": 30,
        "r_u": 0.75,
        "uniform_true_nodes": 35,
    }


def test_evaluate_budget_curve_reports_core_budget_policies():
    rows = [
        _row(1, 0.0, 1.0, 1.0, 0.1),
        _row(1, 0.0, 0.8, 0.9, 0.2),
        _row(0, 0.5, 0.5, 0.1, 1.0),
        _row(0, 0.5, 0.0, 0.0, 0.9),
    ]

    curve = evaluate_budget_curve(
        train_rows=rows,
        eval_rows=rows,
        feature_names=["feature"],
        budgets=[0.0, 0.5, 1.0],
        random_seed=0,
    )

    assert [item["budget_fraction"] for item in curve] == [0.0, 0.5, 1.0]
    assert curve[0]["dive_return"] == curve[0]["cheap_return"]
    assert curve[-1]["dive_return"] == curve[-1]["always_verify_return"]
    assert {
        "random_return",
        "uncertainty_return",
        "oracle_return",
        "value_rank_return",
        "think_longer_return",
        "uniform_true_return",
    }.issubset(curve[1])
    assert {
        "dive_nodes",
        "random_nodes",
        "uncertainty_nodes",
        "oracle_nodes",
        "value_rank_nodes",
        "think_longer_nodes",
        "uniform_true_nodes",
    }.issubset(curve[1])


def test_evaluate_budget_curve_selects_helpful_rows_for_dive_on_toy_data():
    rows = [
        _row(1, 0.0, 1.0, 1.0, 0.1),
        _row(1, 0.0, 0.8, 0.9, 0.2),
        _row(0, 0.5, 0.5, 0.1, 1.0),
        _row(0, 0.5, 0.0, 0.0, 0.9),
    ]

    result = evaluate_budget_curve(
        train_rows=rows,
        eval_rows=rows,
        feature_names=["feature"],
        budgets=[0.5],
        random_seed=0,
    )[0]

    assert result["dive_selection"]["helpful_selected"] == 2
    assert result["dive_return"] > result["uncertainty_return"]


def test_evaluate_budget_curve_reports_value_rank_policy_on_toy_data():
    rows = [
        _row(1, 0.0, 1.0, 1.0, 0.1),
        _row(1, 0.0, 0.8, 0.8, 0.2),
        _row(0, 0.5, 0.5, 0.1, 1.0),
        _row(0, 0.5, 0.0, -0.3, 0.9),
    ]

    result = evaluate_budget_curve(
        train_rows=rows,
        eval_rows=rows,
        feature_names=["feature"],
        budgets=[0.5],
        random_seed=0,
    )[0]

    assert result["value_rank_selection"]["helpful_selected"] == 2
    assert result["value_rank_return"] > result["uncertainty_return"]
    assert result["value_rank_selection"]["mean_delta_r"] == pytest.approx(0.9)
    assert result["value_rank_selection"]["total_delta_r"] == pytest.approx(1.8)


def test_aggregate_budget_curves_reports_mean_and_std_by_budget():
    curves = [
        [
            {"budget_fraction": 0.0, "dive_return": 1.0, "random_return": 0.5},
            {"budget_fraction": 0.5, "dive_return": 2.0, "random_return": 1.0},
        ],
        [
            {"budget_fraction": 0.0, "dive_return": 3.0, "random_return": 1.5},
            {"budget_fraction": 0.5, "dive_return": 4.0, "random_return": 2.0},
        ],
    ]

    aggregate = aggregate_budget_curves(curves, metrics=["dive_return", "random_return"])

    assert aggregate[0]["budget_fraction"] == 0.0
    assert aggregate[0]["dive_return_mean"] == 2.0
    assert aggregate[0]["dive_return_std"] == 1.0
    assert aggregate[1]["random_return_mean"] == 1.5
    assert aggregate[1]["random_return_std"] == 0.5
