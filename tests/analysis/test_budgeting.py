from wmsv.analysis.budgeting import (
    compute_summary,
    mean_return_for_mask,
    threshold_budget_mask,
)


ROWS = [
    {"r_c": 0.0, "r_v": 1.0, "cheap_nodes": 10, "verifier_nodes": 100},
    {"r_c": 0.5, "r_v": 0.5, "cheap_nodes": 10, "verifier_nodes": 100},
    {"r_c": 0.2, "r_v": 0.8, "cheap_nodes": 10, "verifier_nodes": 100},
    {"r_c": 1.0, "r_v": 0.0, "cheap_nodes": 10, "verifier_nodes": 100},
]


def test_threshold_budget_mask_selects_highest_scores():
    mask = threshold_budget_mask([0.2, 0.9, 0.4, 0.1], budget_fraction=0.5)

    assert mask == [False, True, True, False]


def test_mean_return_for_mask_uses_verified_rows():
    value = mean_return_for_mask(ROWS, [True, False, True, False])

    assert value == (1.0 + 0.5 + 0.8 + 1.0) / 4.0


def test_compute_summary_counts_verifier_calls_and_nodes():
    summary = compute_summary(ROWS, [True, False, True, False])

    assert summary["verifier_call_fraction"] == 0.5
    assert summary["cheap_nodes"] == 40
    assert summary["verifier_nodes"] == 200
    assert summary["total_nodes"] == 240
