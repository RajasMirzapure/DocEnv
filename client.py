from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import DocAction, DocObservation, DocState
except ImportError:
    from models import DocAction, DocObservation, DocState

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class DocEnv(EnvClient[DocAction, DocObservation, DocState]):
    """HTTP / WebSocket client for the Hospital Scheduler environment."""

    def _step_payload(self, action: DocAction) -> Dict[str, Any]:
        return {
            "assigned_doctor_id": action.assigned_doctor_id,
            "action_type": action.action_type,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DocObservation]:
        obs_data = payload.get("observation", {})
        observation = DocObservation(
            incoming_event=obs_data.get("incoming_event"),
            current_hour=obs_data.get("current_hour", 8),
            doctors=obs_data.get("doctors", []),
            queue_size=obs_data.get("queue_size", 0),
            stats=obs_data.get("stats", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DocState:
        return DocState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_hour=payload.get("current_hour", 8),
            total_score=payload.get("total_score", 0.0),
            patients_scheduled=payload.get("patients_scheduled", 0),
            patients_waitlisted=payload.get("patients_waitlisted", 0),
            violations=payload.get("violations", 0),
        )
