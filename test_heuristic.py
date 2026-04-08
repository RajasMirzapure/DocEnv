"""Heuristic (Smart) Baseline — rule-based scheduling.

Strategy:
  1. Emergency → pick ER doc with fewest hours (and a free slot).
  2. Routine   → pick matching-specialty doc with fewest hours.
  3. Overflow  → try ER docs for routine patients.
  4. Last resort → waitlist.

Usage:  python test_heuristic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import DocAction, DocObservation
from server.environment import HospitalEnvironment, NIGHT_SHIFT_CUTOFF

NUM_EPISODES = 100


def heuristic_action(obs: DocObservation) -> DocAction:
    """Smart rule-based policy."""
    event = obs.incoming_event
    if event is None:
        return DocAction(assigned_doctor_id="Doc_0", action_type="waitlist")

    hour = obs.current_hour
    doctors = obs.doctors
    etype = event["event_type"]
    specialty = event.get("specialty_needed")

    def _can_schedule(doc: dict) -> bool:
        """Basic eligibility check."""
        if not doc.get("is_available", False):
            return False
        if doc["hours_worked"] >= doc["max_hours"]:
            return False
        if doc.get("worked_last_night", False) and hour < NIGHT_SHIFT_CUTOFF:
            return False
        # Must have at least one free slot
        sched = doc.get("schedule", {})
        has_free = any(v == "free" for k, v in sched.items() if int(k) >= hour)
        return has_free

    def _pick_best(candidates: list) -> str:
        """Pick candidate with fewest hours worked (most capacity left)."""
        candidates.sort(key=lambda d: d["hours_worked"])
        return candidates[0]["doctor_id"]

    # ── EMERGENCY ─────────────────────────────────────────────────────
    if etype == "patient_emergency":
        er_docs = [d for d in doctors if d["specialty"] == "ER" and _can_schedule(d)]
        if er_docs:
            return DocAction(
                assigned_doctor_id=_pick_best(er_docs),
                action_type="book_appointment",
            )
        # No ER doc available — must waitlist
        return DocAction(assigned_doctor_id="Doc_0", action_type="waitlist")

    # ── ROUTINE ───────────────────────────────────────────────────────
    # First: try matching specialty
    match_docs = [
        d for d in doctors
        if d["specialty"] == specialty and _can_schedule(d)
    ]
    if match_docs:
        return DocAction(
            assigned_doctor_id=_pick_best(match_docs),
            action_type="book_appointment",
        )

    # Fallback: try ER docs as overflow
    er_docs = [d for d in doctors if d["specialty"] == "ER" and _can_schedule(d)]
    if er_docs:
        return DocAction(
            assigned_doctor_id=_pick_best(er_docs),
            action_type="book_appointment",
        )

    # No valid doctor → waitlist
    return DocAction(assigned_doctor_id="Doc_0", action_type="waitlist")


def run_episode(env: HospitalEnvironment) -> dict:
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while not obs.done:
        action = heuristic_action(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
        if steps > 500:
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
    print(f"  🧠 HEURISTIC BASELINE — {NUM_EPISODES} episodes")
    print("=" * 70)

    results = []
    for i in range(NUM_EPISODES):
        r = run_episode(env)
        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{NUM_EPISODES} done")

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

    if violation_pct < 10:
        print(f"\n  ✅ Heuristic achieves {violation_pct:.0f}% violation rate (should beat random!)")
    else:
        print(f"\n  ⚠️  {violation_pct:.0f}% violation rate — heuristic may need tuning")

    print(f"\n  📊 This is the score an RL agent needs to beat.\n")


if __name__ == "__main__":
    main()
