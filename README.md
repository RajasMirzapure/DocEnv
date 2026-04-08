# 🏥 DocEnv — Hospital Shift Scheduling Environment

> A reinforcement learning environment where an AI agent routes patients to doctors
> across a simulated workday — respecting labor laws, specialty constraints,
> ER coverage minimums, and real-time disruptions.
>
> Built for the **OpenEnv Hackathon**.

![Framework](https://img.shields.io/badge/Framework-OpenEnv-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Why DocEnv?

Real hospital dispatchers make hundreds of scheduling decisions per shift under
high stakes: wrong assignments waste specialist time, violated labor laws create
liability, and failing to maintain ER coverage can be dangerous.

DocEnv models this as a sequential decision problem. Each step the agent sees
one incoming event — a patient arrival or a disruption — and must decide where
to route it. The simulation tracks shift hours, doctor fatigue, specialty
matching, and ER coverage in real time.

---

## How an Episode Works

```
reset(task_id)
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│  Observation                                                   │
│  ─────────────────────────────────────────────────────────    │
│  incoming_event  │  current_hour  │  doctors  │  queue_size   │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
Agent picks an action
    book_appointment(doctor_id)   or   waitlist
    │
    ▼
step(action)  ──►  new observation  +  intermediate reward
    │
    ▼
done = True  ──►  final score  (0.0 – 1.0)
```

---

## Project Structure

```
DocEnv/
│
├── server/                         Core API package
│   ├── __init__.py                 Package marker
│   ├── app.py                      FastAPI application. Wires /reset, /step,
│   │                               /state endpoints to HospitalEnvironment
│   │                               via openenv's create_app() factory.
│   └── environment.py              Core simulation engine. Manages the doctor
│                                   roster, event queue, shift clock, constraint
│                                   validation, reward calculation, and full
│                                   episode lifecycle including sick-doctor
│                                   disruptions and patient bumping.
│
├── bonus_rl/                       Bonus — trained PPO agent (Gymnasium-based)
│   ├── gym_wrapper.py              Wraps HospitalEnvironment as a standard
│   │                               Gymnasium env with a flat 56-dim float32
│   │                               observation vector and a Discrete(7) action
│   │                               space (6 doctors + waitlist).
│   ├── train_ppo.py                Trains a PPO agent using Stable-Baselines3.
│   │                               Runs 4 parallel envs, evaluates every 5k
│   │                               steps, and saves best/final model checkpoints
│   │                               to ./ppo_hospital/.
│   ├── evaluate.py                 Loads a saved PPO model and benchmarks it
│   │                               over N episodes. Prints a comparison table
│   │                               against the random and heuristic baselines.
│   │                               Supports --render for a live terminal view.
│   └── gemini_agent.py             Zero-shot Gemini agent. Feeds the full
│                                   observation as JSON to gemini-2.5-flash and
│                                   parses the structured JSON action response.
│                                   Requires GEMINI_API_KEY env var.
│
├── ppo_hospital/                   Saved PPO checkpoints (auto-generated)
│   ├── best_model.zip              Best checkpoint by eval reward during training
│   ├── final_model.zip             Model weights at end of training run
│   └── evaluations.npz             Evaluation reward history logged by SB3
│
├── __init__.py                     Top-level package marker
├── models.py                       Typed Pydantic contracts shared by client
│                                   and server:
│                                     DocAction      — agent's action each step
│                                     DocObservation — what the agent sees
│                                     DocState       — episode metadata (/state)
│
├── client.py                       OpenEnv EnvClient subclass. Handles HTTP
│                                   serialization so callers work with typed
│                                   Python objects instead of raw dicts.
│
├── inference.py                    Hackathon-compliant LLM baseline. Drives
│                                   any OpenAI-compatible model through all 3
│                                   tasks. Reads OPENAI_API_KEY (or HF_TOKEN),
│                                   emits strict [START] / [STEP] / [END] log
│                                   lines for automated evaluator scoring.
│
├── test_heuristic.py               Rule-based heuristic agent. Always routes
│                                   to the best eligible doctor (fewest hours,
│                                   correct specialty) and waitlists only when
│                                   no legal assignment exists. Runs 100 episodes
│                                   and prints avg reward + violation rate.
│
├── test_random.py                  Random baseline. Picks a random doctor each
│                                   step with a 10% chance of waitlisting.
│                                   Used to verify the environment doesn't crash
│                                   and to establish a lower-bound score.
│
├── openenv.yaml                    OpenEnv metadata manifest. Declares the env
│                                   name, version, description, and all 3 task
│                                   IDs (easy_shift, medium_shift, hard_shift).
│
├── pyproject.toml                  Project dependencies (openenv-core, FastAPI,
│                                   uvicorn, pydantic, openai, stable-baselines3,
│                                   google-genai, etc.) and the doc-env-server
│                                   console script entry point.
│
├── Dockerfile                      Multi-stage container build. Installs deps
│                                   with uv, copies the app, launches uvicorn
│                                   on port 8000, includes a /health check.
│
├── uv.lock                         Pinned dependency lockfile for fully
│                                   reproducible installs across machines.
│
└── .gitignore                      Excludes .venv/, __pycache__/, .env files,
                                    build artifacts, and OS noise.
```

---

## Quick Start

**1. Install dependencies**

```bash
uv sync
```

**2. Start the server**

```bash
uv run doc-env-server
# or: uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The API is live at `http://localhost:8000`.
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

**3. Run the LLM baseline**

```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o"               # optional, default: gpt-4o
export OPENENV_BASE_URL="http://127.0.0.1:8000"

uv run python inference.py
```

**4. Run the rule-based heuristic (no API key needed)**

```bash
uv run python test_heuristic.py
```

---

## Docker

```bash
docker build -t docenv-app .
docker run -p 8000:8000 docenv-app
```

---

## API Reference

### POST /reset

Starts a new episode. Returns the first observation.

```json
// Request (all fields optional)
{
  "task_id": "easy_shift",
  "seed": 42
}
```

| Field     | Default        | Description                                      |
| --------- | -------------- | ------------------------------------------------ |
| `task_id` | `"easy_shift"` | Which task to run                                |
| `seed`    | per-task value | Integer seed for deterministic episode replay    |


### POST /step

Sends an action and advances the simulation by one event.

```json
// Request
{
  "action_type": "book_appointment",
  "assigned_doctor_id": "Doc_2"
}

// Response
{
  "observation": { ... },
  "reward": 10.0,
  "done": false
}
```


### GET /state

Returns episode-level metadata without advancing the clock.

```json
{
  "episode_id": "a3f9...",
  "step_count": 7,
  "current_hour": 11,
  "total_score": 60.0,
  "patients_scheduled": 6,
  "patients_waitlisted": 1,
  "violations": 0
}
```

---

## Action Space

Every step the agent sends exactly one of these:

```json
{ "action_type": "book_appointment", "assigned_doctor_id": "Doc_2" }
{ "action_type": "waitlist",         "assigned_doctor_id": "Doc_0" }
```

`assigned_doctor_id` is required for `book_appointment` but ignored for `waitlist`.

**Routing rules the agent must follow:**

- Emergency patients → ER doctors only (`Doc_0`, `Doc_1`)
- Routine patients → matching specialty first; ER doctors are valid overflow
- Cannot book a doctor who called in sick (`is_available = false`)
- Cannot book a night-shift doctor before 12:00 (`worked_last_night = true`)
- Cannot book a doctor who has reached 8 hours worked

Breaking any of these triggers a **fatal violation** — the episode ends immediately
with a −100 penalty.

---

## Observation Space

```json
{
  "incoming_event": {
    "event_id":          "EVT_09_3",
    "event_type":        "patient_routine",
    "patient_id":        "P_0003",
    "specialty_needed":  "Cardiology",
    "priority":          "routine",
    "affected_doctor_id": null,
    "description":       "Routine cardiology patient arrived"
  },
  "current_hour": 9,
  "doctors": [
    {
      "doctor_id":        "Doc_4",
      "specialty":        "Cardiology",
      "hours_worked":     2,
      "max_hours":        8,
      "worked_last_night": false,
      "is_available":     true,
      "schedule": {
        "8": "P_0001",
        "9": "free",
        "10": "free"
      }
    }
  ],
  "queue_size": 2,
  "stats": {
    "patients_scheduled": 3,
    "patients_waitlisted": 0,
    "violations": 0,
    "total_score": 30.0
  },
  "done": false,
  "reward": 10.0
}
```

| Field                        | Type         | Description                                                    |
| ---------------------------- | ------------ | -------------------------------------------------------------- |
| `incoming_event`             | dict / null  | The event the agent must handle right now                      |
| `incoming_event.event_type`  | string       | `patient_routine`, `patient_emergency`, or `doctor_sick`       |
| `incoming_event.specialty_needed` | string  | Required specialty for routine patients                        |
| `current_hour`               | int          | Simulated hour of the day (8 – 19)                             |
| `doctors`                    | list         | Full live status of every doctor                               |
| `doctors[].schedule`         | dict         | Hour → patient ID or `"free"` (keys are strings)              |
| `queue_size`                 | int          | Events still waiting after the current one                     |
| `stats`                      | dict         | Running episode totals (scheduled, waitlisted, violations)     |
| `done`                       | bool         | `true` when the episode has ended                              |
| `reward`                     | float        | Step reward; **normalized to [0.0 – 1.0] on the final step**  |

---

## Reward System

### Per-step signals

| Signal        | Value     | When it fires                                               |
| ------------- | --------- | ----------------------------------------------------------- |
| `R_BOOK`      | **+10.0** | Patient successfully booked to a valid doctor               |
| `R_FRICTION`  | **−2.0**  | Emergency bumped a routine patient off an ER slot           |
| `R_WAITLIST`  | **−10.0** | Agent chose to waitlist the patient                         |
| `R_IDLE_GAP`  | **−1.0**  | Each idle gap in a doctor's schedule (applied end-of-day)   |
| `R_FATAL`     | **−100.0**| Hard constraint violated — episode terminates immediately   |

### Terminal score

On the final step (`done = True`), the reward field returns a normalized score:

```
final_score = clamp( total_score / (patients_seen × 10.0), 0.0, 1.0 )
```

This is the number the hackathon evaluator reads as the episode grade.

### Emergency bumping

If an emergency arrives and the only available ER doctor is full but has a
routine patient scheduled, the agent may still assign the emergency there.
The routine patient is automatically re-queued at the front of the line.
This is allowed but penalized with `R_FRICTION = −2.0` to discourage over-
relying on ER overflow.

### Summary — what to optimize for

```
✅  Book every patient to a correctly matched doctor
✅  Balance load across doctors (minimize idle gaps)

❌  Waitlisting patients            →  −10 each
❌  Dropping ER coverage below 2    →  fatal, episode over
❌  Exceeding 8-hour shift limit     →  fatal, episode over
❌  Booking a night-shift doc early  →  fatal, episode over
❌  Specialty mismatch               →  fatal, episode over
❌  Booking a sick doctor            →  fatal, episode over
```

---

## Doctor Roster

Six doctors are always present. Each has a fixed specialty and an 8-hour cap.

| ID      | Specialty    | Role                      |
| ------- | ------------ | ------------------------- |
| `Doc_0` | ER           | Emergency first-responder |
| `Doc_1` | ER           | Emergency first-responder |
| `Doc_2` | General      | General medicine          |
| `Doc_3` | General      | General medicine          |
| `Doc_4` | Cardiology   | Specialist                |
| `Doc_5` | Orthopedics  | Specialist                |

At least 2 ER doctors must remain available at all times.
In `hard_shift`, roughly 30% of doctors have worked the previous night and
cannot be scheduled before 12:00.

---

## Tasks

| Task           | Shift Length       | Events / Hour | Sick Events | Night Shifts | Seed |
| -------------- | ------------------ | ------------- | ----------- | ------------ | ---- |
| `easy_shift`   | 8 h (08:00–16:00)  | 0 – 2         | 0           | No           | 42   |
| `medium_shift` | 12 h (08:00–20:00) | 1 – 3         | 1           | No           | 7    |
| `hard_shift`   | 12 h (08:00–20:00) | 1 – 4         | 2           | Yes          | 13   |

**What each task tests:**

- `easy_shift` — No disruptions, low volume. Tests basic specialty routing and load balancing.
- `medium_shift` — One doctor may call in sick mid-shift, orphaning their patients back into the queue. Requires ER overflow management.
- `hard_shift` — Two sick events, maximum patient load, and fatigued doctors. Full constraint awareness under pressure.

**Event probabilities per hour:**

| Event Type        | Probability |
| ----------------- | ----------- |
| Routine patient   | 70%         |
| Emergency patient | 20%         |
| Doctor sick       | 10%         |

Routine patient specialties: General 50% · Cardiology 25% · Orthopedics 25%

---

## Baseline Scores

Produced by the rule-based heuristic (`test_heuristic.py`) using the default per-task seeds.

| Task           | Fixed-seed Score | Avg (20 runs) | Min   | Max   |
| -------------- | ---------------- | ------------- | ----- | ----- |
| `easy_shift`   | **0.975**        | 0.943         | 0.860 | 1.000 |
| `medium_shift` | **0.835**        | 0.903         | 0.812 | 0.988 |
| `hard_shift`   | **0.560**        | 0.828         | 0.524 | 1.000 |

A **random agent** typically scores below 0.20 on all tasks due to hard-constraint violations.
A well-tuned LLM agent is expected to reach ≥ 0.75 on `hard_shift`.

---

## Inference Log Format

`inference.py` emits these exact lines to stdout for hackathon scoring:

```
[START] task=easy_shift env=doc_env model=gpt-4o
[STEP] step=1 action='{"assigned_doctor_id": "Doc_2", "action_type": "book_appointment"}' reward=10.0 done=False error=None
[STEP] step=2 action='{"assigned_doctor_id": "Doc_4", "action_type": "book_appointment"}' reward=10.0 done=False error=None
...
[END] success=True steps=6 score=0.97 rewards=[10.0, 10.0, ...]
```

One `[START]` / `[END]` block is emitted per task. All three tasks run sequentially.

---

## Environment Variables

| Variable           | Required | Default                        | Description                                       |
| ------------------ | -------- | ------------------------------ | ------------------------------------------------- |
| `OPENAI_API_KEY`   | Yes*     | —                              | OpenAI-compatible API key                         |
| `HF_TOKEN`         | Yes*     | —                              | Hugging Face token (fallback if no OPENAI_API_KEY)|
| `API_BASE_URL`     | No       | `https://api.openai.com/v1`    | LLM API base URL (swap for any compatible server) |
| `MODEL_NAME`       | No       | `gpt-4o`                       | Model name passed to the API                      |
| `OPENENV_BASE_URL` | No       | `http://0.0.0.0:8000`          | URL of the running DocEnv server                  |

*One of `OPENAI_API_KEY` or `HF_TOKEN` must be set to run `inference.py`.

---
