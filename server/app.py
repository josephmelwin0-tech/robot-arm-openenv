from fastapi import FastAPI
from env.environment import RobotAssemblyEnv
import uvicorn

app = FastAPI()

env_instance = None


@app.post("/reset")
def reset(req: dict = None):
    global env_instance

    try:
        task_id = "easy"
        if req and "task" in req:
            task_id = req["task"]

        env_instance = RobotAssemblyEnv(task_id=task_id, seed=42)
        obs = env_instance.reset()
        return obs.model_dump()

    except Exception as e:
        return {
            "error": str(e),
            "status": "reset_failed"
        }


@app.post("/step")
def step(action: dict):
    global env_instance

    if env_instance is None:  # ✅ prevent crash
        return {
            "error": "Environment not initialized. Call /reset first.",
            "done": True
        }

    try:
        obs, reward, done, info = env_instance.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {
            "error": str(e),
            "done": True
        }


@app.get("/")
def root():
    return {"status": "running"}



