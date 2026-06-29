from scripts.summarize_stage_a_ci import summarize_ci_for_item


def _row(label, r_c, r_v, r_t, r_u, score_margin, ensemble_uncertainty):
    return {
        "a_c": 0,
        "a_v": 1 if label else 0,
        "r_c": r_c,
        "r_v": r_v,
        "r_t": r_t,
        "r_u": r_u,
        "label": label,
        "score_margin": score_margin,
        "uncertainty_proxy": ensemble_uncertainty,
        "cheap_score": 0.0,
        "ensemble_action_disagreement": ensemble_uncertainty,
        "ensemble_score_variance": 0.0,
        "ensemble_uncertainty": ensemble_uncertainty,
        "cheap_plan_length": 3.0,
        "cheap_plan_turns": 1.0,
        "cheap_plan_unique_actions": 2.0,
        "cheap_plan_score_per_step": 0.0,
    }


def test_summarize_ci_for_item_reports_policy_and_delta_intervals():
    item = {
        "push_error_rate": 0.5,
        "budget_sweep": [{"train_rows": 2, "budget_fraction": 0.5}],
        "rows": [
            _row(1, 0.0, 1.0, 0.5, 0.75, 0.1, 1.0),
            _row(0, 0.5, 0.5, 0.5, 0.50, 1.0, 0.0),
            _row(1, 0.0, 1.0, 0.5, 0.75, 0.2, 1.0),
            _row(0, 0.5, 0.5, 0.5, 0.50, 1.1, 0.0),
        ],
    }

    summary = summarize_ci_for_item(item, budget_fraction=0.5, samples=20, seed=0)

    assert summary["push_error_rate"] == 0.5
    assert summary["budget_fraction"] == 0.5
    assert "decision" in summary["policy_ci"]
    assert "decision_minus_uncertainty" in summary["delta_ci"]
    assert "decision_minus_uniform_true" in summary["delta_ci"]


def test_summarize_ci_for_item_accepts_plan_gate_feature_set():
    item = {
        "push_error_rate": 0.75,
        "budget_sweep": [{"train_rows": 2, "budget_fraction": 0.5}],
        "rows": [
            _row(1, 0.0, 1.0, 0.5, 0.75, 0.1, 1.0),
            _row(0, 0.5, 0.5, 0.5, 0.50, 1.0, 0.0),
            _row(1, 0.0, 1.0, 0.5, 0.75, 0.2, 1.0),
            _row(0, 0.5, 0.5, 0.5, 0.50, 1.1, 0.0),
        ],
    }

    summary = summarize_ci_for_item(
        item,
        budget_fraction=0.5,
        samples=20,
        seed=0,
        gate_feature_set="plan",
    )

    assert summary["gate_feature_set"] == "plan"
    assert "decision" in summary["policy_ci"]


def test_summarize_ci_for_item_accepts_logistic_gate_model():
    item = {
        "push_error_rate": 0.75,
        "budget_sweep": [{"train_rows": 2, "budget_fraction": 0.5}],
        "rows": [
            _row(1, 0.0, 1.0, 0.5, 0.75, 0.1, 1.0),
            _row(0, 0.5, 0.5, 0.5, 0.50, 1.0, 0.0),
            _row(1, 0.0, 1.0, 0.5, 0.75, 0.2, 1.0),
            _row(0, 0.5, 0.5, 0.5, 0.50, 1.1, 0.0),
        ],
    }

    summary = summarize_ci_for_item(
        item,
        budget_fraction=0.5,
        samples=20,
        seed=0,
        gate_model="logistic",
    )

    assert summary["gate_model"] == "logistic"
    assert "decision" in summary["policy_ci"]
