from pathlib import Path

from wmsv.data.boxoban import iter_boxoban_levels, parse_boxoban_text
from wmsv.data.generated_sokoban import generate_sokoban_levels, write_boxoban_levels
from wmsv.envs.sokoban import parse_level


def test_generate_sokoban_levels_is_deterministic_and_unique():
    left = generate_sokoban_levels(count=8, seed=7)
    right = generate_sokoban_levels(count=8, seed=7)

    assert left == right
    assert len({tuple(level.lines) for level in left}) == 8
    assert [level.level_id for level in left] == [f"generated-7-{idx:04d}" for idx in range(8)]


def test_generate_sokoban_levels_are_parseable_and_solvable_by_certificate():
    levels = generate_sokoban_levels(count=12, seed=3)

    for level in levels:
        state = parse_level(level.lines)
        assert state.boxes.sum() == state.goals.sum() == 1
        assert not state.is_solved()
        current = state
        for action in level.solution:
            current, _, done, _ = current.step(action)
            if done:
                break
        assert current.is_solved()


def test_write_boxoban_levels_uses_existing_parser(tmp_path: Path):
    out = tmp_path / "generated.txt"
    levels = generate_sokoban_levels(count=5, seed=11)

    write_boxoban_levels(levels, out)

    parsed = parse_boxoban_text(out.read_text(), source_file=str(out))
    iterated = list(iter_boxoban_levels(tmp_path))
    assert [level.level_id for level in parsed] == [level.level_id for level in levels]
    assert [level.level_id for level in iterated] == [level.level_id for level in levels]
