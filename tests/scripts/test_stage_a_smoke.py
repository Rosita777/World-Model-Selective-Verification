from scripts.run_stage_a_smoke import (
    build_rows,
    evaluate_rankers,
    gate_features_for_set,
    parse_rate_list,
    parse_budget_list,
    policy_return_vectors,
    run_budget_sweep,
    split_rows,
)


def _gate_row(label: int, r_c: float, r_v: float, score_margin: float, ensemble_uncertainty: float) -> dict:
    return {
        "a_c": 0,
        "a_v": 1 if label else 0,
        "r_c": r_c,
        "r_v": r_v,
        "r_t": r_c,
        "r_u": r_c,
        "label": label,
        "score_margin": score_margin,
        "uncertainty_proxy": ensemble_uncertainty,
        "cheap_score": 0.0,
        "ensemble_action_disagreement": ensemble_uncertainty,
        "ensemble_score_variance": 0.0,
        "ensemble_uncertainty": ensemble_uncertainty,
        "cheap_nodes": 10,
        "verifier_nodes": 100,
        "think_longer_nodes": 30,
        "uniform_true_nodes": 35,
    }


def test_build_rows_returns_label_rows_for_tiny_levels():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        uncertainty_seeds=3,
        cheap_depth=1,
        cheap_width=2,
        think_longer_depth=2,
        think_longer_width=4,
        uniform_true_depth=2,
        uniform_true_width=4,
    )

    assert len(rows) >= 2
    assert {"level_id", "a_c", "a_v", "a_t", "a_u", "r_c", "r_v", "r_t", "r_u", "label"}.issubset(rows[0])
    assert {"ensemble_action_disagreement", "ensemble_score_variance", "ensemble_uncertainty"}.issubset(rows[0])
    assert rows[0]["ensemble_num_planners"] == 3
    assert rows[0]["think_longer_nodes"] > rows[0]["cheap_nodes"]
    assert rows[0]["uniform_true_nodes"] > rows[0]["cheap_nodes"]


def test_build_rows_can_sample_multiple_planning_states_per_level():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        cheap_depth=1,
        cheap_width=2,
        verifier_depth=2,
        verifier_width=4,
        eval_depth=2,
        eval_width=4,
        states_per_level=3,
        state_sampler_depth=1,
        state_sampler_width=4,
    )

    assert len(rows) > len({row["base_level_id"] for row in rows})
    assert {"base_level_id", "state_index"}.issubset(rows[0])
    assert any(":s1" in row["level_id"] for row in rows)


def test_build_rows_supports_deadlock_evaluator_mode():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        cheap_depth=1,
        cheap_width=2,
        verifier_depth=2,
        verifier_width=4,
        eval_depth=2,
        eval_width=4,
        evaluator_mode="deadlock",
    )

    assert len(rows) >= 2
    assert rows[0]["evaluator_mode"] == "deadlock"


def test_build_rows_supports_plan_decision_unit():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        cheap_depth=2,
        cheap_width=4,
        verifier_depth=2,
        verifier_width=4,
        eval_depth=2,
        eval_width=4,
        decision_unit="plan",
    )

    assert len(rows) >= 2
    assert rows[0]["decision_unit"] == "plan"
    assert {"plan_c", "plan_v", "plan_t", "plan_u"}.issubset(rows[0])


def test_build_rows_adds_plan_gate_features():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        cheap_depth=3,
        cheap_width=4,
        verifier_depth=3,
        verifier_width=4,
        eval_depth=3,
        eval_width=4,
        decision_unit="plan",
    )

    assert {
        "cheap_plan_length",
        "cheap_plan_turns",
        "cheap_plan_unique_actions",
        "cheap_plan_score_per_step",
    }.issubset(rows[0])


def test_plan_gate_features_do_not_use_verifier_outputs():
    feature_names = gate_features_for_set("plan")

    assert all("verifier" not in name for name in feature_names)


def test_build_rows_adds_trajectory_gate_features():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        cheap_depth=3,
        cheap_width=4,
        verifier_depth=3,
        verifier_width=4,
        eval_depth=3,
        eval_width=4,
        uncertainty_seeds=3,
        decision_unit="plan",
    )

    assert {
        "cheap_plan_final_progress",
        "cheap_plan_state_change_fraction",
        "cheap_plan_box_change_fraction",
        "ensemble_plan_disagreement",
    }.issubset(rows[0])


def test_trajectory_gate_features_do_not_use_verifier_outputs():
    feature_names = gate_features_for_set("trajectory")

    assert all("verifier" not in name for name in feature_names)


def test_evaluate_rankers_accepts_plan_gate_feature_set():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        cheap_depth=3,
        cheap_width=4,
        verifier_depth=3,
        verifier_width=4,
        eval_depth=3,
        eval_width=4,
        decision_unit="plan",
    )

    result = evaluate_rankers(rows, rows, budget_fraction=0.5, gate_feature_set="plan")

    assert result["gate_feature_set"] == "plan"
    assert isinstance(result["decision_return"], float)


def test_evaluate_rankers_accepts_trajectory_gate_feature_set():
    rows = build_rows(
        push_error_rate=1.0,
        corrupt_push_penalty=1.0,
        cheap_depth=3,
        cheap_width=4,
        verifier_depth=3,
        verifier_width=4,
        eval_depth=3,
        eval_width=4,
        uncertainty_seeds=3,
        decision_unit="plan",
    )

    result = evaluate_rankers(rows, rows, budget_fraction=0.5, gate_feature_set="trajectory")

    assert result["gate_feature_set"] == "trajectory"
    assert isinstance(result["decision_return"], float)


def test_evaluate_rankers_accepts_standardized_centroid_gate_model():
    rows = [
        _gate_row(1, 0.0, 1.0, 0.1, 1.0),
        _gate_row(0, 0.5, 0.5, 1.0, 0.0),
        _gate_row(1, 0.0, 1.0, 0.2, 1.0),
        _gate_row(0, 0.5, 0.5, 1.1, 0.0),
    ]

    result = evaluate_rankers(
        rows,
        rows,
        budget_fraction=0.5,
        gate_model="standardized_centroid",
    )

    assert result["gate_model"] == "standardized_centroid"
    assert isinstance(result["decision_return"], float)


def test_policy_return_vectors_accept_logistic_gate_model():
    rows = [
        _gate_row(1, 0.0, 1.0, 0.1, 1.0),
        _gate_row(0, 0.5, 0.5, 1.0, 0.0),
        _gate_row(1, 0.0, 1.0, 0.2, 1.0),
        _gate_row(0, 0.5, 0.5, 1.1, 0.0),
    ]
    train_rows, eval_rows = split_rows(rows, train_fraction=0.5)

    vectors = policy_return_vectors(
        train_rows,
        eval_rows,
        budget_fraction=0.5,
        gate_model="logistic",
    )

    assert "decision" in vectors
    assert len(vectors["decision"]) == len(eval_rows)


def test_evaluate_rankers_reports_core_returns():
    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0)

    result = evaluate_rankers(rows, rows, budget_fraction=0.5)

    assert {
        "cheap_return",
        "always_return",
        "think_longer_return",
        "uniform_true_return",
        "uncertainty_return",
        "random_return",
        "oracle_return",
        "decision_return",
        "uncertainty_selection",
        "random_selection",
        "oracle_selection",
        "decision_selection",
        "eval_rows",
        "cheap_nodes",
        "always_nodes",
        "think_longer_nodes",
        "uniform_true_nodes",
        "uncertainty_nodes",
        "random_nodes",
        "oracle_nodes",
        "decision_nodes",
    }.issubset(result)
    assert isinstance(result["cheap_return"], float)
    assert isinstance(result["always_return"], float)
    assert isinstance(result["think_longer_return"], float)
    assert isinstance(result["uniform_true_return"], float)
    assert result["gate_model"] == "centroid"
    assert result["eval_rows"] == len(rows)
    assert "helpful_precision" in result["uncertainty_selection"]
    assert result["always_nodes"] > result["cheap_nodes"]
    assert result["think_longer_nodes"] >= result["cheap_nodes"]
    assert result["uniform_true_nodes"] >= result["cheap_nodes"]


def test_evaluate_rankers_reports_random_and_oracle_budget_baselines():
    rows = [
        {
            "a_c": 0,
            "a_v": 1,
            "r_c": 0.0,
            "r_v": 1.0,
            "r_t": 0.5,
            "r_u": 0.75,
            "label": 1,
            "score_margin": 0.1,
            "uncertainty_proxy": 0.0,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 0.0,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 0.0,
            "cheap_nodes": 10,
            "verifier_nodes": 100,
            "think_longer_nodes": 30,
            "uniform_true_nodes": 35,
        },
        {
            "a_c": 0,
            "a_v": 1,
            "r_c": 0.0,
            "r_v": 0.8,
            "r_t": 0.25,
            "r_u": 0.5,
            "label": 1,
            "score_margin": 0.2,
            "uncertainty_proxy": 1.0,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 1.0,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 1.0,
            "cheap_nodes": 10,
            "verifier_nodes": 100,
            "think_longer_nodes": 30,
            "uniform_true_nodes": 35,
        },
        {
            "a_c": 0,
            "a_v": 0,
            "r_c": 0.5,
            "r_v": 0.5,
            "r_t": 0.75,
            "r_u": 0.5,
            "label": 0,
            "score_margin": 1.0,
            "uncertainty_proxy": 0.5,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 0.5,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 0.5,
            "cheap_nodes": 10,
            "verifier_nodes": 100,
            "think_longer_nodes": 30,
            "uniform_true_nodes": 35,
        },
        {
            "a_c": 0,
            "a_v": 1,
            "r_c": 0.5,
            "r_v": 0.0,
            "r_t": 0.25,
            "r_u": 0.25,
            "label": 0,
            "score_margin": 1.1,
            "uncertainty_proxy": 0.4,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 0.4,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 0.4,
            "cheap_nodes": 10,
            "verifier_nodes": 100,
            "think_longer_nodes": 30,
            "uniform_true_nodes": 35,
        },
    ]

    result = evaluate_rankers(rows, rows, budget_fraction=0.25, random_seed=0)

    assert result["oracle_return"] == 0.5
    assert result["oracle_selection"]["helpful_selected"] == 1
    assert isinstance(result["random_return"], float)
    assert result["think_longer_return"] == 0.4375
    assert result["uniform_true_return"] == 0.5
    assert result["cheap_nodes"] == 10
    assert result["always_nodes"] == 110
    assert result["think_longer_nodes"] == 30
    assert result["uniform_true_nodes"] == 35
    assert result["oracle_nodes"] == 35


def test_parse_budget_list_accepts_comma_separated_fractions():
    assert parse_budget_list("0.05,0.10,0.25") == [0.05, 0.1, 0.25]


def test_parse_rate_list_accepts_selected_error_rates():
    assert parse_rate_list("0.50,0.75") == [0.5, 0.75]


def test_run_budget_sweep_reports_one_result_per_budget():
    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0)
    train_rows, eval_rows = split_rows(rows, train_fraction=0.5)

    sweep = run_budget_sweep(train_rows, eval_rows, budgets=[0.25, 0.5], random_seed=0)

    assert [item["budget_fraction"] for item in sweep] == [0.25, 0.5]
    assert all("decision_return" in item for item in sweep)


def test_policy_return_vectors_report_per_level_values_for_ci():
    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0)
    train_rows, eval_rows = split_rows(rows, train_fraction=0.5)

    vectors = policy_return_vectors(train_rows, eval_rows, budget_fraction=0.5, random_seed=0)

    assert {"cheap", "always", "think_longer", "uniform_true", "uncertainty", "random", "oracle"}.issubset(vectors)
    assert all(len(values) == len(eval_rows) for values in vectors.values())


def test_build_rows_can_read_boxoban_folder(tmp_path):
    folder = tmp_path / "levels"
    folder.mkdir()
    (folder / "sample.txt").write_text(
        """
; 0
#####
#@$.#
#   #
#####
""".strip()
    )

    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0, levels_folder=folder, limit=1)

    assert len(rows) == 1
    assert rows[0]["level_id"] == "0"


def test_split_rows_keeps_train_and_eval_disjoint():
    rows = [{"level_id": str(idx)} for idx in range(5)]

    train_rows, eval_rows = split_rows(rows, train_fraction=0.6)

    assert [row["level_id"] for row in train_rows] == ["0", "1", "2"]
    assert [row["level_id"] for row in eval_rows] == ["3", "4"]
