from scripts.run_stage_a_smoke import build_rows, evaluate_rankers


def test_build_rows_returns_label_rows_for_tiny_levels():
    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0)

    assert len(rows) >= 2
    assert {"level_id", "a_c", "a_v", "r_c", "r_v", "label"}.issubset(rows[0])


def test_evaluate_rankers_reports_core_returns():
    rows = build_rows(push_error_rate=1.0, corrupt_push_penalty=1.0)

    result = evaluate_rankers(rows, budget_fraction=0.5)

    assert {"cheap_return", "always_return", "uncertainty_return", "decision_return"}.issubset(result)
    assert isinstance(result["cheap_return"], float)
    assert isinstance(result["always_return"], float)


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
