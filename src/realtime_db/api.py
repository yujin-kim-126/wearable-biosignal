from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import load_config
from .db import DbClient
from .intervention import run_intervention_once
from .simulator import run_session_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SimulateRequest(BaseModel):
    subject_code: str = Field(..., examples=["S1"])
    session_name: str = Field(..., examples=["Final", "Midterm 1", "Midterm 2"])
    session_id: int = Field(..., examples=[1])
    speed_multiplier: float = Field(default=30.0, gt=0)
    max_events: int | None = Field(default=None, gt=0)
    sync_method: str = Field(default="zoh", pattern="^(zoh|linear)$")
    dry_run: bool = False


class InterventionRequest(BaseModel):
    session_id: int = Field(..., examples=[31])
    window_seconds: int = Field(default=60, gt=0)
    cooldown_seconds: int = Field(default=300, ge=0)


app = FastAPI(
    title="Realtime Integrated DB Simulator",
    description="FastAPI-based dynamic simulator for wearable bio-signal event streaming.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/simulate/session")
async def simulate_session(req: SimulateRequest) -> dict[str, float | int]:
    cfg = load_config()
    db_client = DbClient(cfg.database_url)

    result = await run_session_simulation(
        dataset_root=cfg.dataset_root,
        subject_code=req.subject_code,
        session_name=req.session_name,
        session_id=req.session_id,
        db_client=db_client,
        speed_multiplier=req.speed_multiplier,
        max_events=req.max_events,
        dry_run=req.dry_run,
        sync_method=req.sync_method,
    )
    return {
        "generated_events": result.generated_events,
        "inserted_events": result.inserted_events,
        "dropped_events": result.dropped_events,
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "throughput_req_per_sec": round(result.throughput_req_per_sec, 4),
        "avg_latency_ms": round(result.avg_latency_ms, 4),
    }


@app.post("/intervention/analyze")
def intervention_analyze(req: InterventionRequest) -> dict[str, str | int | float | bool | None]:
    cfg = load_config()
    result = run_intervention_once(
        database_url=cfg.database_url,
        project_root=PROJECT_ROOT,
        session_id=req.session_id,
        window_seconds=req.window_seconds,
        cooldown_seconds=req.cooldown_seconds,
    )
    if result is None:
        return {"status": "no_new_window"}

    return {
        "status": "ok",
        "session_id": result.session_id,
        "learner_id": result.learner_id,
        "window_end_ts": result.window_end_ts,
        "model_version": result.model_version,
        "predicted_label": result.predicted_label,
        "predicted_confidence": round(result.predicted_confidence, 6),
        "risk_score": round(result.risk_score, 6),
        "recommended_action": result.recommended_action,
        "should_alert": result.should_alert,
        "alert_suppressed": result.alert_suppressed,
        "reason_text": result.reason_text,
        "inference_id": result.inference_id,
        "alert_id": result.alert_id,
    }
