import math

from wmsv.analysis.verification_value import (
    VerificationLabels,
    auroc,
    classify_verification,
    go_no_go_status,
)


def test_classify_verification_marks_helpful_change():
    labels = classify_verification(a_c=0, a_v=1, r_c=0.2, r_v=0.8, epsilon=0.05)

    assert labels == VerificationLabels(
        y_change=1,
        y_helpful=1,
        y_harm=0,
        y_waste=0,
        delta_r=0.6,
    )


def test_classify_verification_marks_harmful_change():
    labels = classify_verification(a_c=0, a_v=2, r_c=0.8, r_v=0.1, epsilon=0.05)

    assert labels.y_change == 1
    assert labels.y_helpful == 0
    assert labels.y_harm == 1
    assert labels.y_waste == 0
    assert labels.delta_r == -0.7000000000000001


def test_classify_verification_marks_waste_when_action_same():
    labels = classify_verification(a_c=0, a_v=0, r_c=0.4, r_v=0.9, epsilon=0.05)

    assert labels.y_change == 0
    assert labels.y_helpful == 0
    assert labels.y_harm == 0
    assert labels.y_waste == 1


def test_classify_verification_supports_continuous_controls():
    labels = classify_verification(
        a_c=[0.0, 0.0],
        a_v=[0.2, 0.0],
        r_c=0.1,
        r_v=0.4,
        epsilon=0.05,
        action_delta=0.1,
    )

    assert labels.y_change == 1
    assert labels.y_helpful == 1


def test_auroc_returns_pairwise_probability():
    value = auroc(scores=[0.1, 0.9, 0.8, 0.2], labels=[0, 1, 1, 0])

    assert value == 1.0


def test_auroc_returns_nan_for_single_class():
    value = auroc(scores=[0.1, 0.2], labels=[1, 1])

    assert math.isnan(value)


def test_go_no_go_status_checks_phase1_thresholds():
    status = go_no_go_status(
        cheap_success=0.40,
        always_verify_success=0.55,
        positive_label_rate=0.25,
        helpful_auroc=0.66,
        budget20_gain=0.04,
        min_always_gain=0.10,
        min_auroc=0.60,
    )

    assert status["passed"] is True
    assert status["failures"] == []
