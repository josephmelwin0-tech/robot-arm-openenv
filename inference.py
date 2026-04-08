# (same imports as before)
import os
import json
from typing import List

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from env.environment import RobotAssemblyEnv
from env.models import Action
from env.grader import grade
from env.tasks import TASKS

from fastapi import FastAPI

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK = "robot_arm_openenv"
MAX_STEPS = 15

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    raise ValueError(
        "Missing required environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN"
    )

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ─────────────────────────────────────────
# PROMPTS (UNCHANGED)
# ─────────────────────────────────────────

def build_system_prompt() -> str:
    return """You are controlling a robot arm.

Rules:
- Place dependencies first
- Place heavy objects before fragile ones
- Complete all objects

Respond ONLY with valid JSON:
{"action_type": "pick_place" | "submit" | "skip", "object_id": "<id or null>"}
"""

def build_user_prompt(obs: dict) -> str:
    lines = []
    for o in obs["objects"]:
        status = "PLACED" if o["placed"] else "UNPLACED"
        frag = " FRAGILE" if o["fragile"] else ""
        dep = f" depends_on={o['depends_on']}" if o.get("depends_on") else ""
        lines.append(f"{o['id']} {status}{frag}{dep}")
    return "Objects:\n" + "\n".join(lines)

# ─────────────────────────────────────────
# SAFE PARSE (UNCHANGED)
# ─────────────────────────────────────────

def safe_parse(raw: str):
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return None

# ─────────────────────────────────────────
# FALLBACK (UNCHANGED)
# ─────────────────────────────────────────

def fallback_action(obs):
    for o in obs.objects:
        if not o.placed:
            if o.depends_on:
                dep = next((x for x in obs.objects if x.id == o.depends_on), None)
                if dep and not dep.placed:
                    continue
            return Action(action_type="pick_place", object_id=o.id)
    return Action(action_type="submit")

def violates_fragility(obs, action):
    if action.action_type != "pick_place":
        return False

    obj = next((o for o in obs.objects if o.id == action.object_id), None)
    if not obj:
        return False

    if obj.fragile:
        for o in obs.objects:
            if not o.placed and not o.fragile:
                return True
    return False

# ─────────────────────────────────────────
# MAIN LOOP (UNCHANGED)
# ─────────────────────────────────────────

def run_task(task_id: str):
    env = RobotAssemblyEnv(task_id=task_id, seed=42)
    obs = env.reset()

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    done = False
    step = 0
    last_error = None

    messages = [{"role": "system", "content": build_system_prompt()}]

    try:
        while not done and step < MAX_STEPS:
            step += 1

            user_msg = build_user_prompt(obs.model_dump())
            messages.append({"role": "user", "content": user_msg})

            action = None

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=150,
                )
                raw = response.choices[0].message.content or ""
                parsed = safe_parse(raw)

                if parsed:
                    action = Action(
                        action_type=parsed.get("action_type", "skip"),
                        object_id=parsed.get("object_id"),
                    )

                messages.append({"role": "assistant", "content": raw})

            except Exception as e:
                last_error = str(e)

            if action is None or violates_fragility(obs, action):
                action = fallback_action(obs)

            action_str = action.action_type

            obs, reward, done, info = env.step(action)
            last_error = info.get("error", None)
            rewards.append(reward)

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={last_error if last_error else 'null'}",
                flush=True
            )

    except Exception as e:
        last_error = str(e)

    score = grade(task_id, env.state())
    success = done

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step} "
        f"score={score:.2f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True
    )

    env.close()

# ─────────────────────────────────────────
# FASTAPI SERVER (UNCHANGED)
# ─────────────────────────────────────────

app = FastAPI()
env_instance = None

@app.post("/reset")
def reset():
    global env_instance
    env_instance = RobotAssemblyEnv(task_id="easy", seed=42)
    obs = env_instance.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: dict):
    global env_instance
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/")
def root():
    return {"status": "running"}

# ─────────────────────────────────────────
# ENTRY POINT (FIXED)
# ─────────────────────────────────────────

if __name__ == "__main__":
    for task_id in TASKS.keys():
        run_task(task_id)

    print("✅ Inference completed. API ready.", flush=True)
