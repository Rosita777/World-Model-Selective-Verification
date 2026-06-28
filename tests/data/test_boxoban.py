from pathlib import Path

from wmsv.data.boxoban import iter_boxoban_levels, parse_boxoban_text


BOXOBAN_TEXT = """
; 0
#####
#@$.#
#   #
#####
; 1
#####
#.@$#
#   #
#####
""".strip()


def test_parse_boxoban_text_returns_level_blocks():
    levels = parse_boxoban_text(BOXOBAN_TEXT, source_file="sample.txt")

    assert len(levels) == 2
    assert levels[0].level_id == "0"
    assert levels[0].lines[1] == "#@$.#"
    assert levels[0].source_file == "sample.txt"
    assert levels[1].level_id == "1"


def test_iter_boxoban_levels_reads_multiple_files(tmp_path: Path):
    folder = tmp_path / "levels"
    folder.mkdir()
    (folder / "000.txt").write_text(BOXOBAN_TEXT)
    (folder / "001.txt").write_text(BOXOBAN_TEXT)

    levels = list(iter_boxoban_levels(folder, limit=3))

    assert len(levels) == 3
    assert levels[0].source_file.endswith("000.txt")
    assert levels[2].source_file.endswith("001.txt")

