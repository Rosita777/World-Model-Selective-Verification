from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class BoxobanLevel:
    level_id: str
    lines: list[str]
    source_file: str = ""


def parse_boxoban_text(text: str, source_file: str = "") -> list[BoxobanLevel]:
    levels: list[BoxobanLevel] = []
    current_id: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_lines
        if current_id is not None and current_lines:
            levels.append(BoxobanLevel(current_id, current_lines, source_file))
        current_id = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith(";"):
            flush()
            current_id = line[1:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return levels


def iter_boxoban_levels(folder: str | Path, limit: int | None = None) -> Iterator[BoxobanLevel]:
    count = 0
    for path in sorted(Path(folder).glob("*.txt")):
        for level in parse_boxoban_text(path.read_text(), source_file=str(path)):
            yield level
            count += 1
            if limit is not None and count >= limit:
                return
