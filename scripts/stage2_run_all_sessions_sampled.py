from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_db.config import load_config
from realtime_db.db import DbClient
from realtime_db.simulator import run_session_simulation


def get_mapping(conn: psycopg.Connection) -> list[tuple[str, str, int]]:
    data_root = ROOT / 'a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0' / 'Data'
    subjects = sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    with conn.cursor() as cur:
        cur.execute('SELECT learner_id FROM learners ORDER BY learner_id')
        learner_ids = [r[0] for r in cur.fetchall()]

    if len(subjects) != len(learner_ids):
        raise RuntimeError(f'subject/learner mismatch: {len(subjects)} vs {len(learner_ids)}')

    mapping: list[tuple[str, str, int]] = []
    with conn.cursor() as cur:
        for subject, learner_id in zip(subjects, learner_ids):
            cur.execute(
                'SELECT session_id, activity_type FROM learning_sessions WHERE learner_id=%s ORDER BY session_id',
                (learner_id,),
            )
            rows = cur.fetchall()
            session_by_name = {name: sid for sid, name in rows}
            for session_name in ['Final', 'Midterm 1', 'Midterm 2']:
                mapping.append((subject, session_name, session_by_name[session_name]))
    return mapping


async def main() -> None:
    cfg = load_config()
    db = DbClient(cfg.database_url)

    max_events = int(os.getenv('STAGE2_MAX_EVENTS_PER_SESSION', '20000'))
    speed = float(os.getenv('STAGE2_SPEED', '1000000'))
    sync_method = os.getenv('STAGE2_SYNC_METHOD', 'zoh')

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute('TRUNCATE TABLE system_metrics, realtime_features, biometric_logs, preprocessed_stream_32hz, sensor_event_stream RESTART IDENTITY CASCADE;')
        conn.commit()
        mapping = get_mapping(conn)

    results = []
    for subject, session_name, session_id in mapping:
        result = await run_session_simulation(
            dataset_root=cfg.dataset_root,
            subject_code=subject,
            session_name=session_name,
            session_id=session_id,
            db_client=db,
            speed_multiplier=speed,
            max_events=max_events,
            dry_run=False,
            sync_method=sync_method,
        )
        results.append((subject, session_name, session_id, result.generated_events, result.inserted_events, result.dropped_events, result.throughput_req_per_sec, result.avg_latency_ms))
        print(subject, session_name, session_id, result.inserted_events)

    out = ROOT / 'outputs' / 'stage2_batch_results.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('subject,session_name,session_id,generated,inserted,dropped,throughput_req_per_sec,avg_latency_ms\n')
        for r in results:
            f.write(','.join(str(x) for x in r) + '\n')


if __name__ == '__main__':
    asyncio.run(main())
