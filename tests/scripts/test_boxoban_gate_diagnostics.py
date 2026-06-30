import json

from scripts.run_boxoban_gate_diagnostics import main


def test_boxoban_gate_diagnostics_writes_correlation_and_selection_report(tmp_path):
    out = tmp_path / "diagnostics.json"

    main(
        [
            "--limit",
            "24",
            "--budgets",
            "0.2,0.5",
            "--out",
            str(out),
        ]
    )

    data = json.loads(out.read_text())

    assert data["environment"] == "boxoban"
    assert data["correlations"]["reference"] == "ensemble_uncertainty"
    assert "impact_uncertainty" in data["correlations"]["correlations"]
    assert [item["budget_fraction"] for item in data["selection_comparisons"]] == [0.2, 0.5]
    assert "left_mean_improvement" in data["selection_comparisons"][0]
