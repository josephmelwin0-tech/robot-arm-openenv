from fastapi import FastAPI
from env.environment import RobotAssemblyEnv

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