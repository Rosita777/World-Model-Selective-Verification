import json

from scripts.run_phase1_budget_curves import main


def test_phase1_budget_curve_script_writes_dual_environment_report(tmp_path):
    out = tmp_path / "curves.json"

    main(
        [
            "--boxoban-limit",
            "20",
            "--maze-variants",
            "1",
            "--maze-episodes-per-variant",
            "20",
            "--budgets",
            "0,0.5,1",
            "--out",
            str(out),
        ]
    )

    data = json.loads(out.read_text())

    assert set(data["environments"]) == {"boxoban", "maze"}
    assert [item["budget_fraction"] for item in data["boxoban"]["budget_curve"]] == [0.0, 0.5, 1.0]
    assert [item["budget_fraction"] for item in data["maze"]["budget_curve"]] == [0.0, 0.5, 1.0]
    assert "helpful_auroc" in data["boxoban"]["summary"]
    assert "helpful_auroc" in data["maze"]["summary"]


def test_phase1_budget_curve_script_aggregates_multiple_seeds(tmp_path):
    out = tmp_path / "curves.json"

    main(
        [
            "--boxoban-limit",
            "12",
            "--maze-variants",
            "1",
            "--maze-episodes-per-variant",
            "12",
            "--budgets",
            "0,0.5",
            "--seeds",
            "0,1",
            "--out",
            str(out),
        ]
    )

    data = json.loads(out.read_text())

    assert len(data["boxoban"]["seed_runs"]) == 2
    assert len(data["maze"]["seed_runs"]) == 2
    assert data["boxoban"]["aggregate_budget_curve"][0]["budget_fraction"] == 0.0
    assert "dive_return_mean" in data["boxoban"]["aggregate_budget_curve"][0]
    assert "value_rank_return_mean" in data["boxoban"]["aggregate_budget_curve"][0]
    assert "random_return_std" in data["maze"]["aggregate_budget_curve"][1]
