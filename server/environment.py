"""Hospital Scheduler Environment — Core simulation logic.

Simulates a 12-hour hospital day (08:00–20:00).  Patients arrive
dynamically (0-3 per hour), and the agent must assign each one to a
doctor while obeying hard constraints (labor law, specialty match,
ER coverage, night-shift rest).  Disruptions (doctor-sick) are
auto-processed; orphaned patients re-enter the queue.
"""

import uuid
import random
from dataclasses import dataclass, field as dc_field
from typing import List, Dict, Optional, Any, Tuple

from openenv.core.env_server import Environment

try:
    from ..models import DocObservation, DocAction, DocState
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import DocObservation, DocAction, DocState


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

START_HOUR = 8
END_HOUR = 20
NIGHT_SHIFT_CUTOFF = 12
WORKED_LAST_NIGHT_PROB = 0.30
MIN_ER_COVERAGE = 2

# Event probabilities (must sum to 1.0)
P_ROUTINE = 0.70
P_EMERGENCY = 0.20
P_SICK = 0.10

# Routine-patient specialty weights (must sum to 1.0)
SPEC_WEIGHTS = {"General": 0.50, "Cardiology": 0.25, "Orthopedics": 0.25}

# Rewards
R_BOOK = 10.0
R_FRICTION = -2.0
R_WAITLIST = -10.0
R_IDLE_GAP = -1.0
R_FATAL = -100.0

DEFAULT_ROSTER = [
    {"doctor_id": "Doc_0", "specialty": "ER"},
    {"doctor_id": "Doc_1", "specialty": "ER"},
    {"doctor_id": "Doc_2", "specialty": "General"},
    {"doctor_id": "Doc_3", "specialty": "General"},
    {"doctor_id": "Doc_4", "specialty": "Cardiology"},
    {"doctor_id": "Doc_5", "specialty": "Orthopedics"},
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class _Doctor:
    doctor_id: str
    specialty: str
    hours_worked: int = 0
    max_hours: int = 8
    worked_last_night: bool = False
    is_available: bool = True
    # hour -> patient_id (None = free)
    schedule: Dict[int, Optional[str]] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doctor_id": self.doctor_id,
            "specialty": self.specialty,
            "hours_worked": self.hours_worked,
            "max_hours": self.max_hours,
            "worked_last_night": self.worked_last_night,
            "is_available": self.is_available,
            "schedule": {str(h): (v or "free") for h, v in sorted(self.schedule.items())},
        }


@dataclass
class _Event:
    event_id: str
    event_type: str          # patient_routine | patient_emergency | doctor_sick
    patient_id: Optional[str] = None
    specialty_needed: Optional[str] = None
    priority: str = "routine"  # routine | emergency | disruption
    affected_doctor_id: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "patient_id": self.patient_id,
            "specialty_needed": self.specialty_needed,
            "priority": self.priority,
            "affected_doctor_id": self.affected_doctor_id,
            "description": self.description,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HospitalEnvironment(Environment):
    """
    RL environment: schedule patients to doctors across a 12-hour day.

    Rules
    -----
    * Emergency patients → ER doctors only.
    * Routine patients   → matching specialty OR any ER doctor (overflow).
    * ER doctors accepting routine patients risk having them bumped
      if an emergency arrives and the ER doc has no free slot.
    * Sick-doctor events are auto-processed; orphaned patients
      re-enter the queue at the front.

    Hard constraints (fatal -100, episode ends)
    -----
    * Doctor works > 8 h.
    * Doctor with worked_last_night scheduled before 12:00.
    * ER coverage drops below 2.
    * Specialty mismatch.
    """

    def __init__(self, roster: Optional[List[Dict[str, str]]] = None):
        super().__init__()
        self._roster_cfg = roster or DEFAULT_ROSTER
        self.reset()

    # ── public API ────────────────────────────────────────────────────────────

    # Default seeds per task for deterministic grading
    TASK_SEEDS = {
        "easy_shift": 42,
        "medium_shift": 7,
        "hard_shift": 13,
    }

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> DocObservation:
        self._task_id = task_id or "easy_shift"
        self._setup_task(self._task_id)

        # Use provided seed, or fall back to the deterministic per-task default
        _seed = seed if seed is not None else self.TASK_SEEDS.get(self._task_id, 0)
        random.seed(_seed)

        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._hour = START_HOUR
        self._score = 0.0
        self._n_scheduled = 0
        self._n_waitlisted = 0
        self._n_violations = 0
        self._done = False
        self._pctr = 0  # patient counter
        self._queue: List[_Event] = []
        self._sick_set: set = set()  # doctors already targeted by sick events
        self._sick_count = 0
        # patient_id -> {"type": ..., "specialty": ...}
        self._patient_registry: Dict[str, Dict[str, str]] = {}

        # Build roster
        self._docs: Dict[str, _Doctor] = {}
        for cfg in self._roster_cfg:
            worked_night = False
            if self._enable_night_shifts and random.random() < WORKED_LAST_NIGHT_PROB:
                worked_night = True
                
            d = _Doctor(
                doctor_id=cfg["doctor_id"],
                specialty=cfg["specialty"],
                worked_last_night=worked_night,
            )
            for h in range(START_HOUR, self._end_hour):
                d.schedule[h] = None
            self._docs[d.doctor_id] = d

        # Generate first hour
        self._gen_events(self._hour)
        self._drain_disruptions()
        self._ensure_event_or_advance()
        self._sync_state()
        return self._obs(0.0)

    def step(self, action: DocAction) -> DocObservation:
        if self._done:
            return self._obs(0.0)
        self._step_count += 1

        # Must have an event to act on
        if not self._queue:
            self._finish_day()
            return self._obs(0.0)

        event = self._queue[0]
        reward = 0.0

        # ── Validate action_type ──────────────────────────────────────────
        if action.action_type not in ("book_appointment", "waitlist"):
            return self._fatal()

        # ── WAITLIST ──────────────────────────────────────────────────────
        if action.action_type == "waitlist":
            self._queue.pop(0)
            self._n_waitlisted += 1
            reward = R_WAITLIST
            self._score += reward
            self._post_step()
            return self._obs(reward)

        # ── BOOK APPOINTMENT ─────────────────────────────────────────────
        doc_id = action.assigned_doctor_id
        if doc_id not in self._docs:
            return self._fatal()

        doc = self._docs[doc_id]

        # Hard-constraint checks
        viol = self._validate(doc, event)
        if viol:
            return self._fatal()

        # Find a free slot
        slot = self._next_free_slot(doc)

        if slot is not None:
            # Clean booking
            self._queue.pop(0)
            self._book(doc, slot, event)
            reward = R_BOOK
        elif event.event_type == "patient_emergency":
            # Try to bump a routine patient from this ER doc
            bump = self._try_bump(doc)
            if bump is not None:
                bumped_slot, bumped_pid = bump
                self._queue.pop(0)
                self._book(doc, bumped_slot, event)
                reward = R_BOOK + R_FRICTION
            else:
                return self._fatal()
        else:
            # Routine patient, doctor full → agent should have picked
            # another doc or waitlisted.  Fatal violation.
            return self._fatal()

        self._score += reward

        # ER-coverage post-check
        if not self._er_ok():
            return self._fatal()

        self._post_step()
        return self._obs(reward)

    @property
    def state(self) -> DocState:
        return self._state

    # ── booking / bumping ─────────────────────────────────────────────────

    def _book(self, doc: _Doctor, slot: int, event: _Event):
        doc.schedule[slot] = event.patient_id
        doc.hours_worked += 1
        self._n_scheduled += 1
        # Register the patient for bump-eligibility lookups
        self._patient_registry[event.patient_id] = {
            "type": event.event_type,
            "specialty": event.specialty_needed or "ER",
        }

    def _try_bump(self, doc: _Doctor) -> Optional[Tuple[int, str]]:
        """Bump the earliest routine patient from *doc*'s schedule."""
        for h in range(self._hour, self._end_hour):
            pid = doc.schedule.get(h)
            if pid is None:
                continue
            info = self._patient_registry.get(pid, {})
            if info.get("type") == "patient_routine":
                # Bump this patient
                doc.schedule[h] = None
                doc.hours_worked -= 1
                # Re-queue at front
                orphan = _Event(
                    event_id=f"BUMP_{pid}",
                    event_type="patient_routine",
                    patient_id=pid,
                    specialty_needed=info.get("specialty", "General"),
                    priority="routine",
                    description=f"Patient {pid} bumped from {doc.doctor_id}",
                )
                self._queue.insert(0, orphan)
                return (h, pid)
        return None

    def _next_free_slot(self, doc: _Doctor) -> Optional[int]:
        """Find the next free slot for *doc* from current hour onward."""
        for h in range(self._hour, self._end_hour):
            if doc.schedule.get(h) is None:
                # Respect night-shift cutoff
                if doc.worked_last_night and h < NIGHT_SHIFT_CUTOFF:
                    continue
                return h
        return None

    # ── constraint validation ─────────────────────────────────────────────

    def _validate(self, doc: _Doctor, event: _Event) -> Optional[str]:
        if not doc.is_available:
            return "Doctor is sick"
        if doc.worked_last_night and self._hour < NIGHT_SHIFT_CUTOFF:
            return "Night-shift rest violation"
        if doc.hours_worked >= doc.max_hours:
            return "Labor-law: max hours exceeded"
        # Specialty rules
        if event.event_type == "patient_emergency":
            if doc.specialty != "ER":
                return "Emergency must go to ER"
        elif event.event_type == "patient_routine":
            if doc.specialty != event.specialty_needed and doc.specialty != "ER":
                return "Specialty mismatch"
        return None

    def _er_ok(self) -> bool:
        n = sum(
            1 for d in self._docs.values()
            if d.specialty == "ER" and d.is_available and d.hours_worked < d.max_hours
        )
        return n >= MIN_ER_COVERAGE

    def _setup_task(self, task_id: str):
        if task_id == "easy_shift":
            self._end_hour = 16
            self._enable_night_shifts = False
            self._max_sick = 0
            self._events_per_hour = (0, 2)
        elif task_id == "medium_shift":
            self._end_hour = 20
            self._enable_night_shifts = False
            self._max_sick = 1
            self._events_per_hour = (1, 3)
        else: # hard_shift
            self._end_hour = 20
            self._enable_night_shifts = True
            self._max_sick = 2
            self._events_per_hour = (1, 4)

    # ── event generation ──────────────────────────────────────────────────

    def _gen_events(self, hour: int):
        num_events = random.randint(*self._events_per_hour)
        for _ in range(num_events):
            ev = self._make_event(hour)
            if ev:
                self._queue.append(ev)

    def _make_event(self, hour: int) -> Optional[_Event]:
        roll = random.random()
        if roll < P_ROUTINE:
            self._pctr += 1
            spec = random.choices(
                list(SPEC_WEIGHTS.keys()),
                weights=list(SPEC_WEIGHTS.values()),
                k=1,
            )[0]
            return _Event(
                event_id=f"EVT_{hour:02d}_{self._pctr}",
                event_type="patient_routine",
                patient_id=f"P_{self._pctr:04d}",
                specialty_needed=spec,
                priority="routine",
                description=f"Routine {spec.lower()} patient arrived",
            )
        if roll < P_ROUTINE + P_EMERGENCY:
            self._pctr += 1
            return _Event(
                event_id=f"EVT_{hour:02d}_{self._pctr}",
                event_type="patient_emergency",
                patient_id=f"P_{self._pctr:04d}",
                priority="emergency",
                description="Emergency patient — requires ER",
            )
        # Doctor sick (non-ER only)
        if self._sick_count >= self._max_sick:
            return None
            
        candidates = [
            d for d in self._docs.values()
            if d.is_available and d.specialty != "ER" and d.doctor_id not in self._sick_set
        ]
        if not candidates:
            return None
        target = random.choice(candidates)
        self._sick_set.add(target.doctor_id)
        self._sick_count += 1
        return _Event(
            event_id=f"EVT_{hour:02d}_SICK",
            event_type="doctor_sick",
            priority="disruption",
            affected_doctor_id=target.doctor_id,
            description=f"{target.doctor_id} ({target.specialty}) called in sick!",
        )

    # ── disruption auto-processing ────────────────────────────────────────

    def _drain_disruptions(self):
        """Auto-process all leading doctor_sick events."""
        while self._queue and self._queue[0].event_type == "doctor_sick":
            ev = self._queue.pop(0)
            self._process_sick(ev)

    def _process_sick(self, ev: _Event):
        doc = self._docs[ev.affected_doctor_id]
        doc.is_available = False
        orphans: List[_Event] = []
        for h in range(self._hour, self._end_hour):
            pid = doc.schedule.get(h)
            if pid:
                info = self._patient_registry.get(pid, {})
                orphans.append(_Event(
                    event_id=f"ORPHAN_{pid}",
                    event_type="patient_routine",
                    patient_id=pid,
                    specialty_needed=info.get("specialty", doc.specialty),
                    priority="routine",
                    description=f"Patient {pid} orphaned — {doc.doctor_id} is sick",
                ))
                doc.schedule[h] = None
                doc.hours_worked -= 1
        self._queue = orphans + self._queue

    # ── clock management ──────────────────────────────────────────────────

    def _ensure_event_or_advance(self):
        """Advance the clock until we have an event or the day ends."""
        while not self._queue and self._hour < self._end_hour - 1:
            self._hour += 1
            self._gen_events(self._hour)
            self._drain_disruptions()
        if not self._queue:
            self._finish_day()

    def _post_step(self):
        """After an action: drain disruptions, advance if empty."""
        self._drain_disruptions()
        if not self._queue:
            self._ensure_event_or_advance()

    def _finish_day(self):
        gap_penalty = self._idle_gaps()
        self._score += gap_penalty
        self._done = True
        self._sync_state()

    # ── idle-gap scoring ──────────────────────────────────────────────────

    def _idle_gaps(self) -> float:
        penalty = 0.0
        for doc in self._docs.values():
            if not doc.is_available:
                continue
            booked = sorted(h for h, p in doc.schedule.items() if p is not None)
            if len(booked) < 2:
                continue
            for h in range(booked[0] + 1, booked[-1]):
                if doc.schedule.get(h) is None:
                    penalty += R_IDLE_GAP
        return penalty

    def _calculate_normalized_score(self) -> float:
        """Returns score in strictly (0, 1) range for hackathon grader."""
        EPS = 0.01  # 1% buffer on each side — robust against rounding
        max_theoretical_reward = max(self._pctr * R_BOOK, 10.0)
        score = max(self._score, 0.0)
        normalized = min(score / max_theoretical_reward, 1.0)
        # Clamp to open interval (0, 1) — validator rejects 0.0 and 1.0
        return min(max(normalized, EPS), 1.0 - EPS)

    # ── observation / state helpers ───────────────────────────────────────

    def _obs(self, reward: float) -> DocObservation:
        incoming = None
        if self._queue and not self._done:
            incoming = self._queue[0].to_dict()
            
        final_reward = reward
        if self._done:
            final_reward = self._calculate_normalized_score()
            
        return DocObservation(
            incoming_event=incoming,
            current_hour=min(self._hour, self._end_hour - 1),
            doctors=[d.to_dict() for d in self._docs.values()],
            queue_size=len(self._queue),
            stats={
                "patients_scheduled": self._n_scheduled,
                "patients_waitlisted": self._n_waitlisted,
                "violations": self._n_violations,
                "total_score": self._score,
            },
            done=self._done,
            reward=final_reward,
        )

    def _fatal(self) -> DocObservation:
        self._n_violations += 1
        self._score += R_FATAL
        self._done = True
        self._sync_state()
        return self._obs(R_FATAL)

    def _sync_state(self):
        self._state = DocState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            current_hour=self._hour,
            total_score=self._score,
            patients_scheduled=self._n_scheduled,
            patients_waitlisted=self._n_waitlisted,
            violations=self._n_violations,
        )
