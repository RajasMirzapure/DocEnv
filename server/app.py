"""FastAPI application for the Hospital Scheduler Environment."""

from openenv.core.env_server import create_app

try:
    from ..models import DocAction, DocObservation
    from .environment import HospitalEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import DocAction, DocObservation
    from server.environment import HospitalEnvironment

# Pass the *class* (factory) for WebSocket per-session support
app = create_app(
    HospitalEnvironment, DocAction, DocObservation, env_name="doc_env"
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
