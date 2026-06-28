from scripts.summarize_stage_a_ci import summarize_ci_for_item


def _row(label, r_c, r_v, r_t, score_margin, ensemble_uncertainty):
    return {
        "a_c": 0,
        "a_v": 1 if label else 0,
        "r_c": r_c,
        "r_v": r_v,
        "r_t": r_t,
        "label": label,
        "score_margin": score_margin,
        "uncertainty_proxy": ensemble_uncertainty,
        "cheap_score": 0.0,
        "ensemble_action_disagreement": ensemble_uncertainty,
        "ensemble_score_variance": 0.0,
        "ensemble_uncertainty": ensemble_uncertainty,
    }


def test_summarize_ci_for_item_reports_policy_and_delta_intervals():
    item = {
        "push_error_rate": 0.5,
        "budget_sweep": [{"train_rows": 2, "budget_fraction": 0.5}],
        "rows": [
            _row(1, 0.0, 1.0, 0.5, 0.1, 1.0),
            _row(0, 0.5, 0.5, 0.5, 1.0, 0.0),
            _row(1, 0.0, 1.0, 0.5, 0.2, 1.0),
            _row(0, 0.5, 0.5, 0.5, 1.1, 0.0),
        ],
    }

    summary = summarize_ci_for_item(item, budget_fraction=0.5, samples=20, seed=0)

    assert summary["push_error_rate"] == 0.5
    assert summary["budget_fraction"] == 0.5
    assert "decision" in summary["policy_ci"]
    assert "decision_minus_uncertainty" in summary["delta_ci"]
