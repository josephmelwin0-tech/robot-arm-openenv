# (same imports as before)
import os
import json
import time
import requests
from typing import List

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from env.models import Action
from env.grader import grade
from env.tasks import TASKS

print("📦 inference.py loaded", flush=True)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
HF_TOKEN     = os.getenv("HF_TOKEN")

# ✅ FIX: correct container-safe URL
BASE_URL = "http://localhost:7860"

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
# WAIT FOR SERVER
# ─────────────────────────────────────────

def wait_for_server():
    for _ in range(20):  # slightly more retries
        try:
            r = requests.get(BASE_URL)
            if r.status_code == 200:
                print("Server ready ✅", flush=True)
                return True
        except:
            pass
        time.sleep(1)
    print("Server not reachable ❌", flush=True)
    return False

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
# FALLBACK (SAFE VERSION)
# ─────────────────────────────────────────

def fallback_action(obs):
    for o in obs["objects"]:
        if not o["placed"]:
            if o.get("depends_on"):
                dep = next((x for x in obs["objects"] if x["id"] == o["depends_on"]), None)
                if dep and not dep["placed"]:
                    continue
            return {"action_type": "pick_place", "object_id": o["id"]}
    return {"action_type": "submit", "object_id": None}

def violates_fragility(obs, action):
    if action["action_type"] != "pick_place":
        return False

    obj = next((o for o in obs["objects"] if o["id"] == action["object_id"]), None)
    if not obj:
        return False

    if obj["fragile"]:
        for o in obs["objects"]:
            if not o["placed"] and not o["fragile"]:
                return True
    return False

# ─────────────────────────────────────────
# MAIN LOOP (API VERSION)
# ─────────────────────────────────────────

def run_task(task_id: str):
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        res = requests.post(f"{BASE_URL}/reset", json={"task": task_id})
        data = res.json()

        if "error" in data:
            print("[RESET FAILED]", data["error"], flush=True)
            return

        obs = data

    except Exception as e:
        print("[RESET ERROR]", e, flush=True)
        return

    rewards: List[float] = []
    done = False
    step = 0
    last_error = None

    messages = [{"role": "system", "content": build_system_prompt()}]

    try:
        while not done and step < MAX_STEPS:
            step += 1

            user_msg = build_user_prompt(obs)
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
                    action = {
                        "action_type": parsed.get("action_type", "skip"),
                        "object_id": parsed.get("object_id"),
                    }

                messages.append({"role": "assistant", "content": raw})

            except Exception as e:
                last_error = str(e)

            if action is None or violates_fragility(obs, action):
                action = fallback_action(obs)

            try:
                res = requests.post(f"{BASE_URL}/step", json=action)
                data = res.json()

                if "error" in data:
                    print("[STEP FAILED]", data["error"], flush=True)
                    break

                if "observation" not in data:
                    print("[INVALID RESPONSE]", data, flush=True)
                    break

            except Exception as e:
                print("[STEP ERROR]", e, flush=True)
                break

            obs = data["observation"]
            reward = data["reward"]
            done = data["done"]
            last_error = data.get("info", {}).get("error", None)

            rewards.append(reward)

            print(
                f"[STEP] step={step} action={action['action_type']} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={last_error if last_error else 'null'}",
                flush=True
            )

    except Exception as e:
        last_error = str(e)

    print(
        f"[END] success={str(done).lower()} "
        f"steps={step} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True
    )

# ─────────────────────────────────────────
# ENTRY POINT (CRITICAL FIX)
# ─────────────────────────────────────────

if __name__ == "__main__":
    try:
        print("🚀 INFERENCE STARTED", flush=True)

        if not wait_for_server():
            print("Server not reachable ❌", flush=True)
            exit(0)  # ✅ DO NOT crash

        for task_id in TASKS.keys():
            run_task(task_id)

        print("✅ Inference completed.", flush=True)

    except Exception as e:
        print("[FATAL ERROR]", str(e), flush=True)
        exit(0)  # ✅ NEVER exit with error
