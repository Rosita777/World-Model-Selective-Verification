from __future__ import annotations

from dataclasses import dataclass

from wmsv.envs.sokoban import SokobanState


@dataclass(frozen=True)
class PlanResult:
    action: int
    score: float
    plan: list[int]
    top_action_scores: dict[int, float]
    nodes_expanded: int


class BeamPlanner:
    def __init__(self, evaluator, depth: int, width: int, step_penalty: float = 0.0):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if width < 1:
            raise ValueError("width must be at least 1")
        self.evaluator = evaluator
        self.depth = int(depth)
        self.width = int(width)
        self.step_penalty = float(step_penalty)

    def plan(self, state: SokobanState) -> PlanResult:
        beams = [(0.0, [], state)]
        top_action_scores = {0: float("-inf"), 1: float("-inf"), 2: float("-inf"), 3: float("-inf")}
        nodes_expanded = 0

        for _ in range(self.depth):
            candidates = []
            for score, plan, current in beams:
                for action in range(4):
                    step = self.evaluator.step(current, action)
                    nodes_expanded += 1
                    new_score = score + step.reward - self.step_penalty
                    new_plan = plan + [action]
                    first_action = new_plan[0]
                    top_action_scores[first_action] = max(top_action_scores[first_action], new_score)
                    candidates.append((new_score, new_plan, step.state))
            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = candidates[: self.width]

        best_score, best_plan, _ = beams[0]
        return PlanResult(
            action=best_plan[0],
            score=best_score,
            plan=best_plan,
            top_action_scores=top_action_scores,
            nodes_expanded=nodes_expanded,
        )

