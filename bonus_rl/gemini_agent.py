"""Gemini-based Agent for the Hospital Scheduler environment.

This script uses the Gemini API to act as the scheduling agent.
It feeds the environment state to Gemini as JSON and parses its response
to make decisions zero-shot.

Usage:
    export GEMINI_API_KEY="your-api-key"
    uv run python gemini_agent.py
"""

import json
import os
import sys
from pathlib import Path

# Insert local path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import DocAction, DocObservation
from server.environment import HospitalEnvironment

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install google-genai:")
    print("uv add google-genai")
    sys.exit(1)

def gemini_action(obs: DocObservation, client: genai.Client) -> DocAction:
    """Uses Gemini to pick the next action based on the observation."""
    if obs.incoming_event is None:
        return DocAction(assigned_doctor_id="Doc_0", action_type="waitlist")
    
    prompt = f"""
You are an expert hospital admission scheduler AI. 
Your goal is to maximize utilization and avoid hard violations.

Current Hour: {obs.current_hour}:00
Remaining events in queue this hour: {obs.queue_size}

Incoming Event to handle:
{json.dumps(obs.incoming_event, indent=2)}

Available Doctors & Status:
{json.dumps(obs.doctors, indent=2)}

Rules & Constraints:
1. Emergency patients MUST go to an ER doctor.
2. Routine patients MUST go to a matching specialty, OR an ER doctor (overflow).
3. Do not book doctors who exceed their max_hours (8).
4. Do not book doctors with `worked_last_night=True` if current hour < 12.
5. Do not book sick doctors (`is_available=False`).
6. Try to pick the doctor with the fewest hours worked to balance load.
7. If absolutely no doctor is legal for this patient, you MUST use "waitlist" to avoid a fatal penalty.

You must output a strictly valid JSON object with your action:
{{
  "assigned_doctor_id": "Doc_X",
  "action_type": "book_appointment" // or "waitlist"
}}
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        # Parse the JSON response
        data = json.loads(response.text)
        return DocAction(
            assigned_doctor_id=data.get("assigned_doctor_id", "Doc_0"),
            action_type=data.get("action_type", "waitlist")
        )
    except Exception as e:
        print(f"\n[!] Error parsing Gemini response: {e}")
        # Safe fallback
        return DocAction(assigned_doctor_id="Doc_0", action_type="waitlist")

def run_episode(env: HospitalEnvironment, episode_idx: int, client: genai.Client) -> dict:
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    print(f"\n--- Episode {episode_idx + 1} ---")
    while not obs.done:
        action = gemini_action(obs, client)
        
        event_name = obs.incoming_event.get("event_id") if obs.incoming_event else "None"
        print(f"Step {steps+1} | Event: {event_name} -> Action: {action.action_type} for {action.assigned_doctor_id}")
        
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
        
        if steps > 200: # Safety break
            break

    return {
        "steps": steps,
        "reward": total_reward,
        "scheduled": obs.stats.get("patients_scheduled", 0),
        "waitlisted": obs.stats.get("patients_waitlisted", 0),
        "violations": obs.stats.get("violations", 0),
    }

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️ Environment variable GEMINI_API_KEY is not set!")
        print("Please set it before running. Example:")
        print("export GEMINI_API_KEY='AIzaSy...'")
        sys.exit(1)

    # Initialize Client with API key from environment
    client = genai.Client()
    env = HospitalEnvironment()
    num_episodes = 5

    print("=" * 70)
    print(f"  ✨ GEMINI AGENT BASELINE — {num_episodes} episodes")
    print("=" * 70)

    results = []
    for i in range(num_episodes):
        r = run_episode(env, i)
        results.append(r)
        print(f"Reward: {r['reward']}, Violations: {r['violations']}")

    avg_reward = sum(r["reward"] for r in results) / len(results)
    violation_pct = sum(1 for r in results if r["violations"] > 0) / len(results) * 100

    print("\n" + "─" * 70)
    print(f"  Gemini Avg reward:     {avg_reward:>8.1f}")
    print(f"  Gemini Violation rate: {violation_pct:>7.1f}%")
    print("─" * 70)


if __name__ == "__main__":
    main()
