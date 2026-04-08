"""
Task Definitions
================
Defines task configs for environment, grader, and inference.
"""

TASKS = {
    "easy": {
        "id": "easy",
        "name": "Basic Pick & Place",
        "difficulty": "easy",

        "description": (
            "3 objects on a tray, no constraints. "
            "Place all objects at their target positions in any order."
        ),

        "max_steps": 10,
        "objects_count": 3,

        # 🔥 No constraints
        "constraints": {
            "dependency": False,
            "fragility": False,
            "ordering": False,
        },

        # 🔥 Scoring weights (used by grader)
        "scoring": {
            "completion": 1.0,
            "efficiency": 0.0,
            "constraints": 0.0,
        },
    },

    "medium": {
        "id": "medium",
        "name": "Constrained Assembly",
        "difficulty": "medium",

        "description": (
            "6 objects with weight and fragility constraints. "
            "Heavy objects must be placed before fragile ones."
        ),

        "max_steps": 20,
        "objects_count": 6,

        "constraints": {
            "dependency": False,
            "fragility": True,
            "ordering": True,
        },

        "scoring": {
            "completion": 0.7,
            "efficiency": 0.2,
            "constraints": 0.1,
        },
    },

    "hard": {
        "id": "hard",
        "name": "Cascading Dependency Assembly",
        "difficulty": "hard",

        "description": (
            "10 objects with strict dependency chains, fragility rules, "
            "and a time limit. Objects must be placed in correct order."
        ),

        "max_steps": 20,
        "objects_count": 10,

        "constraints": {
            "dependency": True,
            "fragility": True,
            "ordering": True,
        },

        "scoring": {
            "completion": 0.6,
            "efficiency": 0.2,
            "constraints": 0.2,
        },
    },
}