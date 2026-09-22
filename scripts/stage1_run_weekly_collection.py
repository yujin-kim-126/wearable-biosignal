from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_db.config import load_config
from realtime_db.db import DbClient
from realtime_db.simulator import run_session_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute planned onboarding collection sessions for one batch."
    )
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--speed", type=float, default=1000.0)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--sync-method", choices=["zoh", "linear"], default="zoh")
    parser.add_argument("--max-planned-sessions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _set_batch_status(conn: psycopg.Connection, batch_id: int, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE collection_batches
            SET status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE batch_id = %s
            """,
            (status, batch_id),
        )


def _load_planned_sessions(
    conn: psycopg.Connection,
    batch_id: int,
    max_rows: int | None,
) -> list[tuple[int, int, str, str, str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                cbs.batch_session_id,
                cbs.session_id,
                cbs.template_subject_code,
                cbs.template_session_name,
                cbs.collection_date::text,
                cbs.session_order
            FROM collection_batch_sessions cbs
            WHERE cbs.batch_id = %s
              AND cbs.status IN ('planned', 'failed')
            ORDER BY cbs.collection_date, cbs.session_order
            """,
            (batch_id,),
        )
        rows = [tuple(r) for r in cur.fetchall()]
    if max_rows is None:
        return rows
    return rows[:max_rows]


def _finalize_batch_status(conn: psycopg.Connection, batch_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_cnt,
                COUNT(*) FILTER (WHERE status = 'completed') AS done_cnt,
                COUNT(*) FILTER (WHERE status = 'failed') AS fail_cnt,
                COUNT(*) FILTER (WHERE status = 'collecting') AS collecting_cnt
            FROM collection_batch_sessions
            WHERE batch_id = %s
            """,
            (batch_id,),
        )
        total_cnt, done_cnt, fail_cnt, collecting_cnt = cur.fetchone()

    if fail_cnt > 0:
        status = "failed"
    elif done_cnt == total_cnt and total_cnt > 0:
        status = "completed"
    elif collecting_cnt > 0 or done_cnt > 0:
        status = "collecting"
    else:
        status = "planned"

    _set_batch_status(conn, batch_id, status)

    if status == "completed":
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE learner_profiles lp
                SET onboarding_status = 'ready_for_training',
                    onboarding_completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE lp.learner_id = (
                    SELECT learner_id FROM collection_batches WHERE batch_id = %s
                )
                """,
                (batch_id,),
            )
    return status


async def main() -> None:
    args = parse_args()
    cfg = load_config()
    db = DbClient(cfg.database_url)
    rows: list[tuple[int, int, str, str, str, int]]

    with db.connect() as conn:
        if not args.dry_run:
            _set_batch_status(conn, args.batch_id, "collecting")
            conn.commit()
        rows = _load_planned_sessions(conn, args.batch_id, args.max_planned_sessions)

    if not rows:
        print(f"No planned sessions to run for batch_id={args.batch_id}")
        return

    run_results: list[dict[str, object]] = []
    for batch_session_id, session_id, subject_code, session_name, collection_date, session_order in rows:
        started_at = datetime.now()
        try:
            with db.connect() as conn:
                if not args.dry_run:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE collection_batch_sessions
                            SET status = 'collecting', started_at = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE batch_session_id = %s
                            """,
                            (started_at, batch_session_id),
                        )
                    conn.commit()

            result = await run_session_simulation(
                dataset_root=cfg.dataset_root,
                subject_code=subject_code,
                session_name=session_name,
                session_id=session_id,
                db_client=db,
                speed_multiplier=args.speed,
                max_events=args.max_events,
                dry_run=args.dry_run,
                sync_method=args.sync_method,
            )

            ended_at = datetime.now()
            if not args.dry_run:
                with db.connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE collection_batch_sessions
                            SET status = 'completed',
                                ended_at = %s,
                                event_count = %s,
                                error_message = NULL,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE batch_session_id = %s
                            """,
                            (ended_at, result.inserted_events, batch_session_id),
                        )
                    conn.commit()

            run_results.append(
                {
                    "batch_session_id": batch_session_id,
                    "session_id": session_id,
                    "subject_code": subject_code,
                    "session_name": session_name,
                    "collection_date": collection_date,
                    "session_order": session_order,
                    "status": "completed",
                    "generated": result.generated_events,
                    "inserted": result.inserted_events,
                    "dropped": result.dropped_events,
                    "throughput": result.throughput_req_per_sec,
                    "avg_latency_ms": result.avg_latency_ms,
                }
            )
            print(
                f"completed batch_session_id={batch_session_id}, "
                f"session_id={session_id}, inserted={result.inserted_events}"
            )

        except Exception as exc:
            ended_at = datetime.now()
            if not args.dry_run:
                with db.connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE collection_batch_sessions
                            SET status = 'failed',
                                ended_at = %s,
                                error_message = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE batch_session_id = %s
                            """,
                            (ended_at, str(exc)[:255], batch_session_id),
                        )
                    conn.commit()
            run_results.append(
                {
                    "batch_session_id": batch_session_id,
                    "session_id": session_id,
                    "subject_code": subject_code,
                    "session_name": session_name,
                    "collection_date": collection_date,
                    "session_order": session_order,
                    "status": "failed",
                    "generated": 0,
                    "inserted": 0,
                    "dropped": 0,
                    "throughput": 0.0,
                    "avg_latency_ms": 0.0,
                }
            )
            print(f"failed batch_session_id={batch_session_id}: {exc}")

    final_status = "planned"
    if not args.dry_run:
        with db.connect() as conn:
            final_status = _finalize_batch_status(conn, args.batch_id)
            conn.commit()

    out_path = ROOT / "outputs" / f"stage1_batch_{args.batch_id}_run.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "batch_session_id,session_id,subject_code,session_name,collection_date,session_order,"
            "status,generated,inserted,dropped,throughput,avg_latency_ms\n"
        )
        for r in run_results:
            f.write(
                f"{r['batch_session_id']},{r['session_id']},{r['subject_code']},{r['session_name']},"
                f"{r['collection_date']},{r['session_order']},{r['status']},"
                f"{r['generated']},{r['inserted']},{r['dropped']},{r['throughput']},{r['avg_latency_ms']}\n"
            )

    print(
        f"batch_id={args.batch_id} run completed. rows={len(run_results)}, "
        f"dry_run={args.dry_run}, final_status={final_status}, output={out_path.name}"
    )


if __name__ == "__main__":
    asyncio.run(main())
