from wmsv.pilots.maze import build_maze_pilot_rows


def test_build_maze_pilot_rows_returns_continuous_labels_and_features():
    rows = build_maze_pilot_rows(variants=2, episodes_per_variant=3, seed=0)

    assert len(rows) == 6
    row = rows[0]
    assert {"a_c", "a_v", "r_c", "r_v", "y_helpful", "delta_r"}.issubset(row)
    assert {"score_margin", "ensemble_uncertainty", "cheap_nodes", "verifier_nodes"}.issubset(row)
    assert {"a_t", "r_t", "think_longer_nodes", "a_u", "r_u", "uniform_true_nodes"}.issubset(row)


def test_maze_pilot_verifier_is_not_systematically_worse_than_cheap():
    rows = build_maze_pilot_rows(variants=2, episodes_per_variant=5, seed=0)

    cheap = sum(float(row["r_c"]) for row in rows) / len(rows)
    verifier = sum(float(row["r_v"]) for row in rows) / len(rows)

    assert verifier >= cheap


def test_maze_pilot_rows_have_episode_diversity():
    rows = build_maze_pilot_rows(variants=1, episodes_per_variant=8, seed=0)

    deltas = {round(float(row["delta_r"]), 6) for row in rows}

    assert len(deltas) > 1


def test_maze_pilot_has_mixed_verification_outcomes():
    rows = build_maze_pilot_rows(variants=2, episodes_per_variant=12, seed=0)

    helpful = sum(int(row["y_helpful"]) for row in rows)
    non_helpful = len(rows) - helpful

    assert helpful >= 4
    assert non_helpful >= 4


def test_maze_pilot_includes_reachable_success_cases():
    rows = build_maze_pilot_rows(variants=2, episodes_per_variant=12, seed=0)

    cheap_successes = sum(float(row["r_c"]) >= 0.0 for row in rows)
    verifier_successes = sum(float(row["r_v"]) >= 0.0 for row in rows)

    assert cheap_successes > 0
    assert verifier_successes > cheap_successes
