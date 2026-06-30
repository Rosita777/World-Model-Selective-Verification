import json

from scripts.run_boxoban_gate_ablation import main


def test_boxoban_gate_ablation_reports_feature_set_results(tmp_path):
    out = tmp_path / "ablation.json"

    main(
        [
            "--limit",
            "24",
            "--feature-sets",
            "score,uncertainty",
            "--budgets",
            "0,0.2",
            "--out",
            str(out),
        ]
    )

    data = json.loads(out.read_text())

    assert data["environment"] == "boxoban"
    assert [item["feature_set"] for item in data["feature_set_results"]] == ["score", "uncertainty"]
    assert "helpful_auroc" in data["feature_set_results"][0]["summary"]
    assert "value_rank_budget20_return" in data["feature_set_results"][0]["summary"]
    assert "risk_aware_value_budget20_return" in data["feature_set_results"][0]["summary"]
    assert [item["budget_fraction"] for item in data["feature_set_results"][1]["budget_curve"]] == [0.0, 0.2]
