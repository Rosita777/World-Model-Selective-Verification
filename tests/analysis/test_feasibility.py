from wmsv.analysis.feasibility import (
    FeasibilityCounts,
    label_from_decisions,
    summarize_labels,
)


def test_label_from_decisions_requires_action_change_and_gain():
    assert label_from_decisions(a_c=0, a_v=1, r_c=0.0, r_v=1.0, epsilon=0.01) == 1
    assert label_from_decisions(a_c=0, a_v=0, r_c=0.0, r_v=1.0, epsilon=0.01) == 0
    assert label_from_decisions(a_c=0, a_v=1, r_c=1.0, r_v=0.0, epsilon=0.01) == 0


def test_summarize_labels_counts_fix_flip_waste_and_spurious():
    labels = [
        {"a_c": 0, "a_v": 1, "r_c": 0.0, "r_v": 1.0},
        {"a_c": 0, "a_v": 1, "r_c": 1.0, "r_v": 0.0},
        {"a_c": 0, "a_v": 0, "r_c": 0.0, "r_v": 1.0},
        {"a_c": 0, "a_v": 1, "r_c": 0.5, "r_v": 0.5},
    ]

    counts = summarize_labels(labels, epsilon=0.01)

    assert counts == FeasibilityCounts(total=4, helpful=1, harmful=1, wasted=1, spurious=1)
    assert counts.helpful_rate == 0.25
    assert counts.harmful_rate == 0.25
    assert counts.wasted_rate == 0.25

