from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeasibilityCounts:
    total: int
    helpful: int
    harmful: int
    wasted: int
    spurious: int

    @property
    def helpful_rate(self) -> float:
        return self.helpful / self.total if self.total else 0.0

    @property
    def harmful_rate(self) -> float:
        return self.harmful / self.total if self.total else 0.0

    @property
    def wasted_rate(self) -> float:
        return self.wasted / self.total if self.total else 0.0

    @property
    def spurious_rate(self) -> float:
        return self.spurious / self.total if self.total else 0.0


def label_from_decisions(a_c: int, a_v: int, r_c: float, r_v: float, epsilon: float) -> int:
    return int(a_v != a_c and r_v > r_c + epsilon)


def summarize_labels(labels: list[dict], epsilon: float) -> FeasibilityCounts:
    helpful = 0
    harmful = 0
    wasted = 0
    spurious = 0

    for row in labels:
        action_changed = row["a_v"] != row["a_c"]
        delta = row["r_v"] - row["r_c"]
        if action_changed and delta > epsilon:
            helpful += 1
        elif action_changed and delta < -epsilon:
            harmful += 1
        elif action_changed:
            spurious += 1
        else:
            wasted += 1

    return FeasibilityCounts(
        total=len(labels),
        helpful=helpful,
        harmful=harmful,
        wasted=wasted,
        spurious=spurious,
    )

