from __future__ import annotations

from collections.abc import Sequence


def threshold_budget_mask(scores: Sequence[float], budget_fraction: float) -> list[bool]:
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must be in [0, 1]")
    n = len(scores)
    if n == 0:
        return []
    count = int(round(n * float(budget_fraction)))
    if count <= 0:
        return [False] * n
    order = sorted(range(n), key=lambda idx: float(scores[idx]), reverse=True)
    chosen = set(order[:count])
    return [idx in chosen for idx in range(n)]


def mean_return_for_mask(rows: Sequence[dict], verify_mask: Sequence[bool]) -> float:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have equal length")
    if not rows:
        return 0.0
    total = 0.0
    for row, verify in zip(rows, verify_mask):
        total += float(row["r_v"] if verify else row["r_c"])
    return total / len(rows)


def compute_summary(rows: Sequence[dict], verify_mask: Sequence[bool]) -> dict:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have equal length")
    cheap_nodes = sum(int(row.get("cheap_nodes", 0)) for row in rows)
    verifier_nodes = sum(
        int(row.get("verifier_nodes", 0))
        for row, verify in zip(rows, verify_mask)
        if verify
    )
    selected = sum(1 for value in verify_mask if value)
    return {
        "verifier_calls": selected,
        "verifier_call_fraction": selected / len(rows) if rows else 0.0,
        "cheap_nodes": cheap_nodes,
        "verifier_nodes": verifier_nodes,
        "total_nodes": cheap_nodes + verifier_nodes,
    }
