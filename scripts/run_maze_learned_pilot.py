from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_boxoban_learned_pilot import _report
from wmsv.pilots.maze import build_maze_pilot_rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--episodes-per-variant", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = build_maze_pilot_rows(
        variants=args.variants,
        episodes_per_variant=args.episodes_per_variant,
    )
    report = _report(rows, environment="maze", min_always_gain=0.10, min_auroc=0.60)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
