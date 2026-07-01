import json

from scripts.summarize_main_curves import main, render_markdown_summary


def _curve_row(budget: float) -> dict:
    return {
        "budget_fraction": budget,
        "dive_return": 1.0 + budget,
        "value_rank_return": 1.2 + budget,
        "uncertainty_return": 0.8 + budget,
        "random_return": 0.5 + budget,
        "oracle_return": 1.5 + budget,
        "think_longer_return": 0.9,
        "uniform_true_return": 1.1,
        "dive_nodes": 10.0,
        "value_rank_nodes": 10.0,
        "uncertainty_nodes": 10.0,
        "random_nodes": 10.0,
        "oracle_nodes": 10.0,
        "dive_selection": {
            "helpful_precision": 0.7,
            "harmful_selected": 1,
            "wasted_selected": 2,
            "mean_delta_r": 0.3,
        },
        "value_rank_selection": {
            "helpful_precision": 0.6,
            "harmful_selected": 0,
            "wasted_selected": 3,
            "mean_delta_r": 0.4,
        },
        "uncertainty_selection": {
            "helpful_precision": 0.4,
            "harmful_selected": 2,
            "wasted_selected": 4,
            "mean_delta_r": 0.1,
        },
    }


def test_render_markdown_summary_reports_main_budget_tables():
    report = {
        "environments": ["boxoban"],
        "seeds": [0, 1],
        "boxoban": {
            "seed_runs": [
                {"budget_curve": [_curve_row(0.1), _curve_row(0.2)]},
                {"budget_curve": [_curve_row(0.1), _curve_row(0.2)]},
            ],
            "aggregate_budget_curve": [
                {
                    "budget_fraction": 0.1,
                    "dive_return_mean": 1.1,
                    "dive_return_std": 0.01,
                    "value_rank_return_mean": 1.3,
                    "value_rank_return_std": 0.02,
                    "uncertainty_return_mean": 0.9,
                    "uncertainty_return_std": 0.03,
                    "random_return_mean": 0.6,
                    "random_return_std": 0.04,
                    "oracle_return_mean": 1.6,
                    "oracle_return_std": 0.05,
                }
            ],
        },
    }

    markdown = render_markdown_summary(report, budgets=[0.1])

    assert "# Main Curves Summary" in markdown
    assert "## boxoban" in markdown
    assert "DIVE" in markdown
    assert "ValueRank" in markdown
    assert "Uncertainty" in markdown
    assert "selected delta" in markdown


def test_summarize_main_curves_writes_markdown_file(tmp_path):
    source = tmp_path / "curves.json"
    out = tmp_path / "summary.md"
    source.write_text(
        json.dumps(
            {
                "environments": ["maze"],
                "seeds": [0],
                "maze": {
                    "seed_runs": [{"budget_curve": [_curve_row(0.5)]}],
                    "aggregate_budget_curve": [
                        {
                            "budget_fraction": 0.5,
                            "dive_return_mean": 1.5,
                            "dive_return_std": 0.0,
                            "value_rank_return_mean": 1.7,
                            "value_rank_return_std": 0.0,
                            "uncertainty_return_mean": 1.3,
                            "uncertainty_return_std": 0.0,
                            "random_return_mean": 1.0,
                            "random_return_std": 0.0,
                            "oracle_return_mean": 2.0,
                            "oracle_return_std": 0.0,
                        }
                    ],
                },
            }
        )
    )

    main(["--input", str(source), "--budgets", "0.5", "--out", str(out)])

    assert "## maze" in out.read_text()
