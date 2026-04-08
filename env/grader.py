"""
Graders
========
Deterministic graders for each task.
All graders return a float in [0.0, 1.0].
"""

from typing import List
from env.models import EnvState


def grade(task_id: str, state: EnvState) -> float:
    if task_id == "easy":
        return grade_easy(state)
    elif task_id == "medium":
        return grade_medium(state)
    elif task_id == "hard":
        return grade_hard(state)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


# 🟢 EASY — simple completion
def grade_easy(state: EnvState) -> float:
    total = len(state.objects)
    if total == 0:
        return 0.0

    placed = sum(o.placed for o in state.objects)
    return round(placed / total, 4)


# 🟡 MEDIUM — adds ordering + efficiency (IMPROVED)
def grade_medium(state: EnvState) -> float:
    total = len(state.objects)
    if total == 0:
        return 0.0

    placed = sum(o.placed for o in state.objects)
    completion = placed / total

    # 🔥 Fragility constraint → convert to SCORE (not just penalty)
    fragile_violations = 0
    for obj in state.objects:
        if obj.fragile and obj.placed:
            heavy_remaining = [
                o for o in state.objects
                if not o.placed and not o.fragile
            ]
            if heavy_remaining:
                fragile_violations += 1

    constraint_score = max(0, 1 - (fragile_violations * 0.1))

    # 🔥 Better efficiency scaling
    max_steps = 20
    if state.time_remaining is not None:
        efficiency = state.time_remaining / max_steps
    else:
        efficiency = max(0, 1 - (state.step_number / max_steps))

    # 🔥 NEW BALANCED SCORING
    score = (
        (completion * 0.7)
        + (efficiency * 0.2)
        + (constraint_score * 0.1)
    )

    return round(max(0.0, min(1.0, score)), 4)


# 🔴 HARD — full constraints + better scaling (IMPROVED)
def grade_hard(state: EnvState) -> float:
    total = len(state.objects)
    if total == 0:
        return 0.0

    placed = sum(o.placed for o in state.objects)
    completion = placed / total

    # 🔥 Dependency correctness
    dependency_violations = 0
    for obj in state.objects:
        if obj.depends_on:
            dep = next((o for o in state.objects if o.id == obj.depends_on), None)
            if obj.placed and dep and not dep.placed:
                dependency_violations += 1

    # 🔥 Fragility correctness
    fragile_violations = 0
    for obj in state.objects:
        if obj.fragile and obj.placed:
            heavy_remaining = [
                o for o in state.objects
                if not o.placed and not o.fragile
            ]
            if heavy_remaining:
                fragile_violations += 1

    # 🔥 Convert violations → constraint score
    total_violations = dependency_violations + fragile_violations
    constraint_score = max(0, 1 - (total_violations * 0.1))

    # 🔥 Better efficiency scaling
    max_steps = 20
    if state.time_remaining is not None:
        efficiency = state.time_remaining / max_steps
    else:
        efficiency = max(0, 1 - (state.step_number / max_steps))

    # 🔥 NEW BALANCED SCORING (VERY IMPORTANT CHANGE)
    score = (
        (completion * 0.65)
        + (efficiency * 0.15)
        + (constraint_score * 0.20)
    )

    return round(max(0.0, min(1.0, score)), 4)