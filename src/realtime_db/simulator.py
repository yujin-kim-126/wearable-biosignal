from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from .dataset_loader import load_session_events
from .db import DbClient


@dataclass
class SimulationResult:
    generated_events: int
    inserted_events: int
    dropped_events: int
    elapsed_seconds: float
    throughput_req_per_sec: float
    avg_latency_ms: float


async def run_session_simulation(
    dataset_root: Path,
    subject_code: str,
    session_name: str,
    session_id: int,
    db_client: DbClient,
    speed_multiplier: float = 30.0,
    max_events: int | None = None,
    dry_run: bool = False,
    sync_method: str = "zoh",
) -> SimulationResult:
    events = load_session_events(
        dataset_root=dataset_root,
        subject_code=subject_code,
        session_name=session_name,
        max_events=max_events,
    )
    if not events:
        return SimulationResult(0, 0, 0, 0.0, 0.0, 0.0)

    generated = len(events)
    inserted = 0
    latency_sum_ms = 0.0
    start = time.perf_counter()
    first_ts = events[0].source_ts
    wall_start = time.perf_counter()

    conn = None if dry_run else db_client.connect()
    try:
        for event in events:
            if not dry_run:
                source_offset_sec = (event.source_ts - first_ts).total_seconds()
                target_elapsed = source_offset_sec / speed_multiplier
                while (time.perf_counter() - wall_start) < target_elapsed:
                    await asyncio.sleep(0.0005)

            step_start = time.perf_counter()

            if not dry_run and conn is not None:
                db_client.insert_sensor_event(conn=conn, session_id=session_id, event=event)
            step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0
            latency_sum_ms += step_elapsed_ms
            inserted += 1

        if not dry_run and conn is not None:
            conn.commit()
            db_client.run_pipeline_functions(
                conn=conn,
                session_id=session_id,
                sync_method=sync_method,
                expected_events=generated,
            )
            conn.commit()
    finally:
        if conn is not None:
            conn.close()

    elapsed = time.perf_counter() - start
    throughput = inserted / elapsed if elapsed > 0 else 0.0
    avg_latency_ms = latency_sum_ms / inserted if inserted > 0 else 0.0
    dropped = generated - inserted

    return SimulationResult(
        generated_events=generated,
        inserted_events=inserted,
        dropped_events=dropped,
        elapsed_seconds=elapsed,
        throughput_req_per_sec=throughput,
        avg_latency_ms=avg_latency_ms,
    )
