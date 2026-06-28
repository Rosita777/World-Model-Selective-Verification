from scripts.run_stage_a_smoke import build_rows, evaluate_rankers, split_rows


def test_build_rows_returns_label_rows_for_tiny_levels():
    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0, uncertainty_seeds=3)

    assert len(rows) >= 2
    assert {"level_id", "a_c", "a_v", "r_c", "r_v", "label"}.issubset(rows[0])
    assert {"ensemble_action_disagreement", "ensemble_score_variance", "ensemble_uncertainty"}.issubset(rows[0])
    assert rows[0]["ensemble_num_planners"] == 3


def test_evaluate_rankers_reports_core_returns():
    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0)

    result = evaluate_rankers(rows, rows, budget_fraction=0.5)

    assert {
        "cheap_return",
        "always_return",
        "uncertainty_return",
        "random_return",
        "oracle_return",
        "decision_return",
        "uncertainty_selection",
        "random_selection",
        "oracle_selection",
        "decision_selection",
        "eval_rows",
    }.issubset(result)
    assert isinstance(result["cheap_return"], float)
    assert isinstance(result["always_return"], float)
    assert result["eval_rows"] == len(rows)
    assert "helpful_precision" in result["uncertainty_selection"]


def test_evaluate_rankers_reports_random_and_oracle_budget_baselines():
    rows = [
        {
            "a_c": 0,
            "a_v": 1,
            "r_c": 0.0,
            "r_v": 1.0,
            "label": 1,
            "score_margin": 0.1,
            "uncertainty_proxy": 0.0,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 0.0,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 0.0,
        },
        {
            "a_c": 0,
            "a_v": 1,
            "r_c": 0.0,
            "r_v": 0.8,
            "label": 1,
            "score_margin": 0.2,
            "uncertainty_proxy": 1.0,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 1.0,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 1.0,
        },
        {
            "a_c": 0,
            "a_v": 0,
            "r_c": 0.5,
            "r_v": 0.5,
            "label": 0,
            "score_margin": 1.0,
            "uncertainty_proxy": 0.5,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 0.5,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 0.5,
        },
        {
            "a_c": 0,
            "a_v": 1,
            "r_c": 0.5,
            "r_v": 0.0,
            "label": 0,
            "score_margin": 1.1,
            "uncertainty_proxy": 0.4,
            "cheap_score": 0.0,
            "ensemble_action_disagreement": 0.4,
            "ensemble_score_variance": 0.0,
            "ensemble_uncertainty": 0.4,
        },
    ]

    result = evaluate_rankers(rows, rows, budget_fraction=0.25, random_seed=0)

    assert result["oracle_return"] == 0.5
    assert result["oracle_selection"]["helpful_selected"] == 1
    assert isinstance(result["random_return"], float)


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
