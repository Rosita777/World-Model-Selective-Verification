import json
from pathlib import Path

from scripts.run_boxoban_learned_pilot import main as boxoban_main
from scripts.run_maze_learned_pilot import main as maze_main


def test_boxoban_pilot_script_writes_report(tmp_path):
    out = tmp_path / "boxoban.json"

    boxoban_main(["--limit", "8", "--out", str(out)])

    data = json.loads(out.read_text())
    assert data["environment"] == "boxoban"
    assert "go_no_go" in data
    assert data["num_rows"] == 8


def test_maze_pilot_script_writes_report(tmp_path):
    out = tmp_path / "maze.json"

    maze_main(["--variants", "2", "--episodes-per-variant", "3", "--out", str(out)])

    data = json.loads(out.read_text())
    assert data["environment"] == "maze"
    assert "go_no_go" in data
    assert data["num_rows"] == 6
    assert "cheap_success" in data
    assert 0.0 <= data["cheap_success"] <= 1.0
    assert "always_verify_success" in data
    assert 0.0 <= data["always_verify_success"] <= 1.0
