"""Baseline inference script for the Hackathon Submission.

Uses the OpenAI API client and strictly formats outputs with [START], [STEP], [END].
Reads API credentials from OPENAI_API_KEY (or HF_TOKEN as fallback).
"""

import json
import os
import sys
from typing import List, Any
from pathlib import Path

import httpx
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import DocAction, DocObservation

# ── Credentials & config ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")

# Support both OPENAI_API_KEY (spec requirement) and HF_TOKEN (HF Spaces style)
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY  = os.getenv("OPENAI_API_KEY") or HF_TOKEN
if not API_KEY:
    raise ValueError(
        "API key not found. Set OPENAI_API_KEY or HF_TOKEN."
    )

BACKEND_URL = os.getenv("OPENENV_BASE_URL", "http://0.0.0.0:8000")
BENCHMARK   = "doc_env"
TASKS       = ["easy_shift", "medium_shift", "hard_shift"]
MAX_STEPS   = 100
SUCCESS_SCORE_THRESHOLD = 0.5


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any):
    action_str = action.replace('\n', ' ')
    done_str   = "True" if done else "False"
    err_str    = "None" if error is None else str(error)
    print(
        f"[STEP] step={step} action='{action_str}' reward={reward:.1f} "
        f"done={done_str} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    success_str  = "True" if success else "False"
    rewards_str  = str([round(r, 2) for r in rewards])
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ── LLM action helper ─────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    current_hour: int,
    queue_size: int,
    event: dict,
    doctors: list,
) -> str:
    prompt = f"""
You are an expert hospital admission scheduler AI.
Your goal is to maximize utilization and avoid hard violations.

Current Hour: {current_hour}:00
Remaining events in queue this hour: {queue_size}

Incoming Event to handle:
{json.dumps(event, indent=2) if event else "None"}

Available Doctors & Status:
{json.dumps(doctors, indent=2)}

Rules & Constraints:
1. Emergency patients MUST go to an ER doctor.
2. Routine patients MUST go to a matching specialty, OR an ER doctor (overflow).
3. Do not book doctors who exceed their max_hours (8).
4. Do not book doctors with worked_last_night=True if current hour < 12.
5. Do not book sick doctors (is_available=False).
6. Pick the doctor with the fewest hours_worked to balance load.
7. If no legal doctor exists for this patient, use "waitlist" to avoid a fatal penalty.

Output a strictly valid JSON object:
{{
  "assigned_doctor_id": "Doc_X",
  "action_type": "book_appointment"
}}
(Use "action_type": "waitlist" when no doctor is legal.)
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"assigned_doctor_id": "Doc_0", "action_type": "waitlist"}'


# ── Per-task runner (synchronous) ─────────────────────────────────────────────

def _parse_obs(payload: dict) -> DocObservation:
    obs_data = payload.get("observation", payload)   # /reset wraps in "observation"
    return DocObservation(
        incoming_event=obs_data.get("incoming_event"),
        current_hour=obs_data.get("current_hour", 8),
        doctors=obs_data.get("doctors", []),
        queue_size=obs_data.get("queue_size", 0),
        stats=obs_data.get("stats", {}),
        done=payload.get("done", obs_data.get("done", False)),
        reward=payload.get("reward", obs_data.get("reward", 0.0)),
    )


def run_task(client: OpenAI, backend_url: str, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Single reset call with task_id
        reset_payload = httpx.post(
            f"{backend_url}/reset",
            json={"task_id": task_name},
            timeout=30,
        ).json()
        obs = _parse_obs(reset_payload)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_text = get_model_action(
                client,
                current_hour=obs.current_hour,
                queue_size=obs.queue_size,
                event=obs.incoming_event,
                doctors=obs.doctors,
            )

            try:
                data   = json.loads(action_text)
                action = DocAction(
                    assigned_doctor_id=data.get("assigned_doctor_id", "Doc_0"),
                    action_type=data.get("action_type", "waitlist"),
                )
            except Exception as exc:
                print(f"[DEBUG] Action parse error: {exc}", flush=True)
                action = DocAction(assigned_doctor_id="Doc_0", action_type="waitlist")

            step_payload = httpx.post(
                f"{backend_url}/step",
                json={
                    "assigned_doctor_id": action.assigned_doctor_id,
                    "action_type": action.action_type,
                },
                timeout=30,
            ).json()
            obs = _parse_obs(step_payload)

            reward = obs.reward or 0.0
            done   = obs.done
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_text, reward=reward, done=done, error=None)

            if done:
                score = reward   # final reward is already normalized [0, 1]
                break

        EPS     = 1e-4
        score   = min(max(score, EPS), 1.0 - EPS)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task failed with exception: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_task(client, BACKEND_URL, task)


if __name__ == "__main__":
    main()
