"""DocEnv — Type-safe contracts for the Hospital Scheduler environment."""

from __future__ import annotations

from typing import List, Dict, Optional, Any

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


# ── Action ────────────────────────────────────────────────────────────────────

class DocAction(Action):
    """Action the agent sends each step."""

    assigned_doctor_id: str = Field(
        description="ID of the doctor to assign, e.g. 'Doc_0'. Ignored when action_type='waitlist'."
    )
    action_type: str = Field(
        description="'book_appointment' or 'waitlist'"
    )


# ── Observation ───────────────────────────────────────────────────────────────

class DocObservation(Observation):
    """Snapshot of the hospital the agent sees after each step."""

    incoming_event: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current event to handle (patient arrival or None when done)",
    )
    current_hour: int = Field(
        default=8,
        description="Simulated hour of the day (8–19)",
    )
    doctors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Status snapshot of every doctor",
    )
    queue_size: int = Field(
        default=0,
        description="Remaining events in the current queue",
    )
    stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Running totals: patients_scheduled, waitlisted, etc.",
    )


# ── State (episode metadata) ─────────────────────────────────────────────────

class DocState(State):
    """Episode-level metadata returned by /state."""

    episode_id: str = Field(default="")
    step_count: int = Field(default=0)
    current_hour: int = Field(default=8)
    total_score: float = Field(default=0.0)
    patients_scheduled: int = Field(default=0)
    patients_waitlisted: int = Field(default=0)
    violations: int = Field(default=0)
