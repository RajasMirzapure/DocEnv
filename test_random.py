"""Random Baseline — validate the environment doesn't crash.

Usage:  python test_random.py
        (no server needed — tests environment directly)
"""

import random
import sys
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import DocAction, DocObservation, DocState
from server.environment import HospitalEnvironment

NUM_EPISODES = 100


def pick_random_action(obs: DocObservation) -> DocAction:
    """Pick a random valid-ish action from the observation."""
    doctors = obs.doctors
    if not doctors or random.random() < 0.1:
        return DocAction(assigned_doctor_id="Doc_0", action_type="waitlist")

    doc = random.choice(doctors)
    return DocAction(
        assigned_doctor_id=doc["doctor_id"],
        action_type="book_appointment",
    )


def run_episode(env: HospitalEnvironment) -> dict:
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while not obs.done:
        action = pick_random_action(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
        if steps > 500:  # safety valve
            break

    return {
        "steps": steps,
        "reward": total_reward,
        "scheduled": obs.stats.get("patients_scheduled", 0),
        "waitlisted": obs.stats.get("patients_waitlisted", 0),
        "violations": obs.stats.get("violations", 0),
    }


def main():
    env = HospitalEnvironment()

    print("=" * 70)
    print(f"  🎲 RANDOM BASELINE — {NUM_EPISODES} episodes")
    print("=" * 70)

    results = []
    for i in range(NUM_EPISODES):
        r = run_episode(env)
        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{NUM_EPISODES} done")

    # Aggregate
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_sched = sum(r["scheduled"] for r in results) / len(results)
    avg_wait = sum(r["waitlisted"] for r in results) / len(results)
    violation_pct = sum(1 for r in results if r["violations"] > 0) / len(results) * 100

    print("\n" + "─" * 70)
    print(f"  Avg reward:        {avg_reward:>8.1f}")
    print(f"  Avg steps/episode: {avg_steps:>8.1f}")
    print(f"  Avg scheduled:     {avg_sched:>8.1f}")
    print(f"  Avg waitlisted:    {avg_wait:>8.1f}")
    print(f"  Violation rate:    {violation_pct:>7.1f}%")
    print("─" * 70)

    if violation_pct > 0:
        print(f"\n  ⚠️  {violation_pct:.0f}% of episodes ended in fatal violations.")
        print("     (Expected for random play — the smart baseline will do better.)")
    else:
        print("\n  ✅  No violations! (Surprising for random play.)")

    print(f"\n  ✅ Environment ran {NUM_EPISODES} episodes with no crashes.\n")


if __name__ == "__main__":
    main()
