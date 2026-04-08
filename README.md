# 🤖 Robot Assembly QA — OpenEnv Environment

> **A real-world robot arm pick & place environment for AI agent training and evaluation.**  
> Agents must sequence object placement on an assembly line, respecting weight, fragility, and dependency constraints — the same reasoning required in industrial automation at companies like Bosch, Toyota, and Foxconn.

---

## 🏭 Motivation

Robot arm sequencing is a critical, unsolved challenge in smart manufacturing. Incorrect placement order causes product defects, line stoppages, and costly rework. This environment lets AI agents learn and be evaluated on constraint-satisfaction reasoning that mirrors real factory automation systems.

---

## 📦 Environment Overview

The agent controls a simulated robot arm over a tray of objects. Each object has:

- A **position** (where it currently is) and a **target** (where it must go)
- A **weight** — heavier objects must generally be placed before lighter ones
- A **fragile** flag — fragile objects must be placed after all heavy objects
- An optional **depends_on** field — the object cannot be placed until its dependency is placed first

The agent issues actions step-by-step and receives shaped rewards throughout the episode.

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task (`easy` / `medium` / `hard`) |
| `step_number` | integer | Current step in the episode |
| `objects` | array | All objects with their full state (see below) |
| `last_action` | string | The last action type taken |
| `last_action_result` | string | Feedback message from the last action |
| `arm_position` | [float, float] | Current XY position of the robot arm |
| `time_remaining` | integer | Steps remaining — `hard` task only |
| `message` | string | Human-readable status message |

**Each object in `objects` contains:**

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique object identifier |
| `name` | string | Human-readable name |
| `weight` | number | Weight in kg |
| `fragile` | boolean | If true, must be placed after all heavy objects |
| `position` | [float, float] | Current XY position on tray |
| `target` | [float, float] | Target XY placement position |
| `depends_on` | string \| null | ID of object that must be placed first |
| `placed` | boolean | Whether this object has been placed |

---

## 🎮 Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | string | `"pick_place"` / `"skip"` / `"submit"` |
| `object_id` | string \| null | ID of the object to place (required for `pick_place`) |
| `metadata` | object | Optional extra metadata |

**Example actions:**

```json
{ "action_type": "pick_place", "object_id": "h1" }
{ "action_type": "submit", "object_id": null }
{ "action_type": "skip", "object_id": null }
```

---

## 🏆 Reward Function

Rewards are shaped across the full trajectory — not just at the end.

| Event | Reward |
|---|---|
| Valid object placement | `+0.30` base |
| Dependency chain completion bonus | `+0.20` |
| Non-fragile object placed correctly | `+0.10` |
| Progress bonus (scales with completion %) | `+0.00 → +0.20` |
| Step penalty (efficiency) | `-0.01` per step |
| Fragility rule violated | `-0.15` |
| Dependency violated | `-0.20` |
| Skip action | `-0.05` |
| Invalid / unknown action | `-0.20` |
| Early submit (incomplete) | `-0.20` |
| Perfect submission (all placed) | `+0.50` |

**Final grader score: `0.0 – 1.0`** — based on completion rate, constraint adherence, and efficiency.

---

## 📋 Tasks

### 🟢 Easy — Basic Pick & Place
- **3 objects**, no constraints
- Place all objects at their target positions in any order
- Max steps: `10`
- Expected baseline score: `~1.00`

### 🟡 Medium — Constrained Assembly  
- **6 objects** — 4 heavy, 2 fragile
- All heavy objects must be placed **before** any fragile object
- Agent must infer the valid ordering from object properties
- Max steps: `20`
- Expected baseline score: `~0.80`

### 🔴 Hard — Dependency Assembly
- **8 objects** with strict dependency chains + fragility rules + **20-step time limit**
- Some objects cannot be placed until their dependency is satisfied
- Mixed constraints: dependency chain `h1→h2→h3`, independent heavy objects, fragile chain
- Agent must reason about both ordering and time pressure simultaneously
- Max steps: `20`
- Expected baseline score: `~0.75`

---

## 🚀 Setup & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
python main.py
# API available at http://localhost:7860
```

### Run with Docker

```bash
docker build -t robot-assembly-qa .

docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
  robot-assembly-qa
```

### Run Inference Script

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

python inference.py
```

**On Windows PowerShell:**
```powershell
$env:HF_TOKEN = "your_hf_token"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
python inference.py
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment info and available endpoints |
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/tasks` | GET | List all tasks with metadata |
| `/reset?task_id=easy` | POST | Start a fresh episode |
| `/step?task_id=easy` | POST | Submit an action, receive observation + reward |
| `/state?task_id=easy` | GET | Get full internal environment state |
| `/grade?task_id=easy` | GET | Get current grader score (0.0–1.0) |

**Example reset:**
```bash
curl -X POST http://localhost:7860/reset?task_id=easy
```

**Example step:**
```bash
curl -X POST http://localhost:7860/step?task_id=easy \
  -H "Content-Type: application/json" \
  -d '{"action_type": "pick_place", "object_id": "A"}'
```

---

## 📊 Baseline Scores

Measured with `Qwen/Qwen2.5-7B-Instruct` via Hugging Face Inference Router:

| Task | Score | Success | Steps |
|---|---|---|---|
| `easy` | `1.00` | ✅ | 3 |
| `medium` | `~0.80` | ✅ | 8 |
| `hard` | `~0.75` | ✅ | 12 |

---

## 🗂️ Project Structure

```
robot-arm-openenv/
├── env/
│   ├── __init__.py
│   ├── environment.py     ← Core env logic (step / reset / state)
│   ├── models.py          ← Pydantic typed models (Observation, Action, EnvState)
│   ├── grader.py          ← Deterministic graders per task (0.0–1.0)
│   └── tasks.py           ← Task metadata definitions
├── inference.py           ← Mandatory baseline runner ([START][STEP][END] logs)
├── main.py                ← FastAPI server
├── openenv.yaml           ← OpenEnv manifest
├── pyproject.toml         ← Package metadata for openenv validate
├── Dockerfile             ← Container definition for HF Spaces
├── requirements.txt       ← Python dependencies
└── README.md
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-7B-Instruct` | Model identifier |

---

## ✅ Validation

```bash
pip install openenv-core
openenv validate . --verbose
```

---

*Built for the Meta × Hugging Face OpenEnv Hackathon — Round 1*