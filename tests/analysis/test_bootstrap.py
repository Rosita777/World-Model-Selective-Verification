from wmsv.analysis.bootstrap import bootstrap_mean_ci, bootstrap_paired_delta_ci


def test_bootstrap_mean_ci_is_exact_for_constant_values():
    result = bootstrap_mean_ci([1.0, 1.0, 1.0], samples=20, seed=0)

    assert result == {"mean": 1.0, "low": 1.0, "high": 1.0}


def test_bootstrap_paired_delta_ci_is_exact_for_constant_deltas():
    result = bootstrap_paired_delta_ci([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], samples=20, seed=0)

    assert result == {"mean": 1.0, "low": 1.0, "high": 1.0}
