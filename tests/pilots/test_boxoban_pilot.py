from pathlib import Path

from wmsv.pilots.boxoban import build_boxoban_pilot_rows


def test_build_boxoban_pilot_rows_returns_dive_labels_and_features():
    rows = build_boxoban_pilot_rows(limit=4, seed=0)

    assert len(rows) == 4
    row = rows[0]
    assert {"a_c", "a_v", "r_c", "r_v", "y_helpful", "delta_r"}.issubset(row)
    assert {"score_margin", "uncertainty_proxy", "cheap_nodes", "verifier_nodes"}.issubset(row)
    assert {"a_t", "r_t", "think_longer_nodes", "a_u", "r_u", "uniform_true_nodes"}.issubset(row)
    assert {
        "cheap_plan_uncertainty_mean",
        "cheap_plan_uncertainty_max",
        "cheap_plan_early_uncertainty",
        "cheap_plan_deadlock_fraction",
        "decision_instability",
        "impact_uncertainty",
        "irreversibility_risk",
        "counterfactual_action_gap",
        "counterfactual_gap_confidence",
        "temporal_consistency",
        "temporal_inconsistency",
    }.issubset(row)


def test_build_boxoban_pilot_rows_can_use_boxoban_level_folder():
    rows = build_boxoban_pilot_rows(
        limit=12,
        seed=0,
        levels_path="data/external/boxoban-sample/medium/train",
        train_level_count=4,
    )

    assert len(rows) == 12
    assert any(row["base_source"] == "boxoban" for row in rows)
    assert any(row["a_c"] != row["a_v"] for row in rows)


def test_build_boxoban_pilot_rows_augments_small_folder_with_generated_levels(tmp_path: Path):
    folder = tmp_path / "tiny_boxoban"
    folder.mkdir()
    (folder / "000.txt").write_text(
        "\n".join(
            [
                "; tiny",
                "#####",
                "#@$.#",
                "#   #",
                "#####",
            ]
        )
    )

    rows = build_boxoban_pilot_rows(
        limit=40,
        seed=5,
        levels_path=str(folder),
        train_level_count=20,
    )

    sources = {row["base_source"] for row in rows}
    level_prefixes = {str(row["level_id"]).split(":")[0] for row in rows}

    assert "generated" in sources
    assert any(level_id.startswith("generated-5-") for level_id in level_prefixes)


def test_boxoban_pilot_has_nonconstant_gate_features():
    rows = build_boxoban_pilot_rows(
        limit=24,
        seed=0,
        levels_path="data/external/boxoban-sample/medium/train",
    )

    varying_features = 0
    for feature in [
        "score_margin",
        "ensemble_uncertainty",
        "cheap_plan_state_change_fraction",
        "cheap_plan_box_change_fraction",
    ]:
        values = {round(float(row[feature]), 6) for row in rows}
        varying_features += int(len(values) > 1)

    assert varying_features >= 2


def test_boxoban_pilot_has_medium_cheap_and_mixed_labels():
    rows = build_boxoban_pilot_rows(
        limit=60,
        seed=0,
        levels_path="data/external/boxoban-sample/medium/train",
    )

    cheap_positive_rate = sum(float(row["r_c"]) > 0.0 for row in rows) / len(rows)
    helpful_rate = sum(int(row["y_helpful"]) for row in rows) / len(rows)

    assert 0.15 <= cheap_positive_rate <= 0.85
    assert 0.15 <= helpful_rate <= 0.50
