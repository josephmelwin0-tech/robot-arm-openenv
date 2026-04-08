"""
Robot Arm Pick & Place Environment (UPGRADED)
=============================================
Improved reward shaping + aligned with grader.
"""

import random
import copy
from typing import Dict, List, Optional, Tuple

from env.models import Object, Observation, Action, EnvState


class RobotAssemblyEnv:

    TASK_IDS = ["easy", "medium", "hard"]

    def __init__(self, task_id: str = "easy", seed: int = 42):
        assert task_id in self.TASK_IDS
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[EnvState] = None
        self.reset()

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)

        self._state = EnvState(
            task_id=self.task_id,
            step_number=0,
            objects=self._generate_objects(),
            arm_position=[0.0, 0.0],
            time_remaining=20 if self.task_id == "hard" else None,
            done=False,
            total_reward=0.0,
            action_history=[]
        )

        return self._build_obs("Episode started")

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:

    # 🔥 FIX: allow dict input
        if isinstance(action, dict):
            action = Action(**action)
        self._state.step_number += 1

        if self._state.time_remaining is not None:
            self._state.time_remaining -= 1

        reward, feedback, error = self._process_action(action)

        # 🔥 Step penalty (important for efficiency learning)
        reward -= 0.01

        self._state.total_reward += reward
        self._state.action_history.append(action.action_type)

        done = self._check_done(action)
        self._state.done = done

        obs = self._build_obs(feedback, action.action_type)

        return obs, round(reward, 2), done, {"error": error}

    def state(self) -> EnvState:
        return copy.deepcopy(self._state)

    def close(self):
        pass

    # ─────────────────────────────────────────
    # CORE LOGIC (IMPROVED)
    # ─────────────────────────────────────────

    def _process_action(self, action: Action):

        # ── SUBMIT ─────────────────────────
        if action.action_type == "submit":
            placed = sum(o.placed for o in self._state.objects)
            total = len(self._state.objects)

            if placed == total:
                return 0.5, "Perfect submission", None
            else:
                return -0.2, "Submitted early", "early_submit"

        # ── SKIP ─────────────────────────
        if action.action_type == "skip":
            return -0.05, "Skipped step", None

        # ── INVALID ACTION ───────────────
        if action.action_type != "pick_place":
            return -0.2, "Invalid action", "invalid_action"

        obj = next((o for o in self._state.objects if o.id == action.object_id), None)

        if not obj:
            return -0.2, "Object not found", "not_found"

        if obj.placed:
            return -0.1, "Already placed", "already_placed"

        # ── DEPENDENCY CHECK ─────────────
        if obj.depends_on:
            dep = next(o for o in self._state.objects if o.id == obj.depends_on)
            if not dep.placed:
                return -0.2, "Dependency violated", "dependency"

        # ── FRAGILITY RULE ───────────────
        if self.task_id != "easy" and obj.fragile:
            heavy_remaining = [
                o for o in self._state.objects
                if not o.placed and not o.fragile and o.id != obj.id
            ]
            if heavy_remaining:
                return -0.15, "Fragile placed too early", "fragility"

        # ── VALID PLACEMENT ──────────────
        obj.placed = True
        self._state.arm_position = obj.target

        reward = 0.3  # base reward

        # 🔥 Dependency reward
        if obj.depends_on:
            reward += 0.2

        # 🔥 Non-fragile early placement reward
        if not obj.fragile:
            reward += 0.1

        # 🔥 Completion shaping
        placed_count = sum(o.placed for o in self._state.objects)
        total = len(self._state.objects)

        reward += (placed_count / total) * 0.2

        return reward, f"Placed {obj.id}", None

    def _check_done(self, action):
        all_done = all(o.placed for o in self._state.objects)

        timeout = (
            self._state.time_remaining is not None
            and self._state.time_remaining <= 0
        )

        return all_done or timeout or action.action_type == "submit"

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _build_obs(self, msg, last_action=None):
        return Observation(
            task_id=self._state.task_id,
            step_number=self._state.step_number,
            objects=copy.deepcopy(self._state.objects),
            last_action=last_action,
            last_action_result=msg,
            arm_position=self._state.arm_position,
            time_remaining=self._state.time_remaining,
            message=msg
        )

    def _generate_objects(self) -> List[Object]:

        if self.task_id == "easy":
            return [
                Object(id="A", name="Block", weight=2, fragile=False,
                       position=[0.1, 0.2], target=[0.8, 0.2]),
                Object(id="B", name="Case", weight=1, fragile=False,
                       position=[0.3, 0.5], target=[0.8, 0.5]),
                Object(id="C", name="Board", weight=0.5, fragile=False,
                       position=[0.6, 0.8], target=[0.8, 0.8]),
            ]

        if self.task_id == "medium":
            return [
                Object(id="1", name="Frame", weight=5, fragile=False,
                       position=[0.1, 0.1], target=[0.9, 0.1]),
                Object(id="2", name="Motor", weight=3, fragile=False,
                       position=[0.2, 0.3], target=[0.9, 0.3]),
                Object(id="3", name="Gasket", weight=1, fragile=False,
                       position=[0.4, 0.5], target=[0.9, 0.5]),
                Object(id="4", name="Glass", weight=1, fragile=True,
                       position=[0.5, 0.6], target=[0.9, 0.6]),
            ]

        if self.task_id == "hard":
            return [
                Object(id="h1", name="Base", weight=8, fragile=False,
                       position=[0.1, 0.1], target=[0.9, 0.1]),
                Object(id="h2", name="Column", weight=6, fragile=False,
                       position=[0.2, 0.2], target=[0.9, 0.2], depends_on="h1"),
                Object(id="h3", name="Shaft", weight=4, fragile=False,
                       position=[0.3, 0.3], target=[0.9, 0.3], depends_on="h2"),
                Object(id="h4", name="PCB", weight=1, fragile=True,
                       position=[0.4, 0.4], target=[0.9, 0.4], depends_on="h3"),
                Object(id="h5", name="PCB", weight=1, fragile=True,
                       position=[0.5, 0.5], target=[0.9, 0.5], depends_on="h4"),
                Object(id="h6", name="PCB", weight=1, fragile=True,
                       position=[0.6, 0.6], target=[0.9, 0.6], depends_on="h5"),
                Object(id="h7", name="PCB", weight=1, fragile=True,
                       position=[0.7, 0.7], target=[0.9, 0.7], depends_on="h6"),
                Object(id="h8", name="PCB", weight=1, fragile=True,
                       position=[0.8, 0.8], target=[0.9, 0.8], depends_on="h7"),
            ]

        return []