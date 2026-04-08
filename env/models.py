from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# 🔵 Object in environment
class Object(BaseModel):
    id: str
    name: str
    weight: float
    fragile: bool

    position: List[float]   # current position
    target: List[float]     # goal position

    depends_on: Optional[str] = None
    placed: bool = False


# 🔵 Observation returned to agent
class Observation(BaseModel):
    task_id: str
    step_number: int

    objects: List[Object]

    arm_position: List[float]

    last_action: Optional[str] = None
    last_action_result: Optional[str] = None

    time_remaining: Optional[int] = None
    message: str = ""


# 🔴 Action given by agent
class Action(BaseModel):
    action_type: str  # "pick_place" | "skip" | "submit"
    object_id: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


# 🟢 Reward (structured — useful for debugging & extension)
class Reward(BaseModel):
    score: float
    breakdown: Dict[str, float]
    feedback: str


# 🔵 Full environment state (for grader + debugging)
class EnvState(BaseModel):
    task_id: str
    step_number: int

    objects: List[Object]

    arm_position: List[float]

    time_remaining: Optional[int]

    done: bool
    total_reward: float

    action_history: List[str]