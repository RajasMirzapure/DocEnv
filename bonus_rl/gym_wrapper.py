"""Gymnasium wrapper for the Hospital Scheduler environment.

Converts the rich dict-based DocEnv into a standard Gymnasium env
with flat numpy observations and a single discrete action space.

Action mapping
--------------
  0–5  →  book_appointment with Doc_0 … Doc_5
  6    →  waitlist (doctor_id is ignored)

Observation vector  (56 float32 values, all in [0, 1])
----------------------------------------------------
  [0]       current_hour  (normalised)
  [1]       queue_size    (normalised, clipped to 1.0)
  [2:4]     event type    (one-hot: routine / emergency)
  [4:8]     specialty     (one-hot: ER / General / Cardiology / Orthopedics)
  [8:56]    6 doctors × 8 features each:
              hours_worked (norm), is_available, worked_last_night,
              specialty (4 one-hot), free_slots_ratio
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.environment import (
    HospitalEnvironment,
    START_HOUR,
    END_HOUR,
    DEFAULT_ROSTER,
)
from models import DocAction


# ── encoding maps ─────────────────────────────────────────────────────────────
SPECIALTY_IDX = {"ER": 0, "General": 1, "Cardiology": 2, "Orthopedics": 3}
NUM_DOCTORS = len(DEFAULT_ROSTER)
DOC_FEAT = 8          # features per doctor
EVT_FEAT = 6          # event type (2) + specialty (4)
GLOBAL_FEAT = 2       # hour + queue_size
OBS_DIM = GLOBAL_FEAT + EVT_FEAT + NUM_DOCTORS * DOC_FEAT  # 56
NUM_ACTIONS = NUM_DOCTORS + 1   # 0-5 = book Doc_i, 6 = waitlist


class HospitalGymEnv(gym.Env):
    """Gymnasium-compatible wrapper around HospitalEnvironment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, roster=None, render_mode=None):
        super().__init__()
        self._roster = roster or DEFAULT_ROSTER
        self._env = HospitalEnvironment(roster=self._roster)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._doc_ids = [c["doctor_id"] for c in self._roster]
        self._last_obs = None

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)
        obs = self._env.reset()
        self._last_obs = obs
        return self._vectorise(obs), self._info(obs)

    def step(self, action: int):
        doc_action = self._decode(int(action))
        obs = self._env.step(doc_action)
        self._last_obs = obs
        if self.render_mode == "human":
            self.render()
        return (
            self._vectorise(obs),
            float(obs.reward),
            bool(obs.done),   # terminated
            False,            # truncated
            self._info(obs),
        )

    def render(self):
        obs = self._last_obs
        if obs is None:
            return
        ev = obs.incoming_event
        print(f"\n⏰ {obs.current_hour}:00  |  queue={obs.queue_size}")
        if ev:
            print(f"  📋 {ev.get('description', '?')}")
        for d in obs.doctors:
            icon = "🟢" if d.get("is_available") else "🔴"
            print(f"  {icon} {d['doctor_id']} ({d['specialty']}) {d['hours_worked']}h")

    # ── helpers ───────────────────────────────────────────────────────────

    def _decode(self, action: int) -> DocAction:
        if action >= NUM_DOCTORS:
            return DocAction(
                assigned_doctor_id=self._doc_ids[0],
                action_type="waitlist",
            )
        return DocAction(
            assigned_doctor_id=self._doc_ids[action],
            action_type="book_appointment",
        )

    def _vectorise(self, obs) -> np.ndarray:
        v = np.zeros(OBS_DIM, dtype=np.float32)
        i = 0

        # ── global ────────────────────────────────────────────────────
        hour_range = max(END_HOUR - START_HOUR, 1)
        v[i] = (obs.current_hour - START_HOUR) / hour_range;  i += 1
        v[i] = min(obs.queue_size / 10.0, 1.0);               i += 1

        # ── event ─────────────────────────────────────────────────────
        ev = obs.incoming_event
        if ev:
            etype = ev.get("event_type", "")
            if etype == "patient_routine":
                v[i] = 1.0
            elif etype == "patient_emergency":
                v[i + 1] = 1.0
            i += 2

            spec = ev.get("specialty_needed")
            if spec and spec in SPECIALTY_IDX:
                v[i + SPECIALTY_IDX[spec]] = 1.0
            i += 4
        else:
            i += EVT_FEAT

        # ── doctors ───────────────────────────────────────────────────
        for doc in obs.doctors:
            v[i] = doc.get("hours_worked", 0) / max(doc.get("max_hours", 8), 1)
            i += 1
            v[i] = 1.0 if doc.get("is_available") else 0.0
            i += 1
            v[i] = 1.0 if doc.get("worked_last_night") else 0.0
            i += 1
            spec = doc.get("specialty", "")
            if spec in SPECIALTY_IDX:
                v[i + SPECIALTY_IDX[spec]] = 1.0
            i += 4
            sched = doc.get("schedule", {})
            total = max(len(sched), 1)
            free = sum(1 for val in sched.values() if val == "free")
            v[i] = free / total
            i += 1

        return v

    @staticmethod
    def _info(obs) -> dict:
        return {
            "scheduled": obs.stats.get("patients_scheduled", 0),
            "waitlisted": obs.stats.get("patients_waitlisted", 0),
            "violations": obs.stats.get("violations", 0),
            "total_score": obs.stats.get("total_score", 0.0),
        }
