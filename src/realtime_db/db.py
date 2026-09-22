from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb

from .dataset_loader import SensorEvent


@dataclass
class DbClient:
    database_url: str

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def insert_sensor_event(
        self,
        conn: psycopg.Connection,
        session_id: int,
        event: SensorEvent,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sensor_event_stream (
                    session_id, sensor_type, source_ts, emitted_at, ingest_ts, sample_rate_hz,
                    value_1, value_2, value_3, emitted_seq, payload_json
                ) VALUES (
                    %(session_id)s, %(sensor_type)s, %(source_ts)s, %(emitted_at)s, %(ingest_ts)s, %(sample_rate_hz)s,
                    %(value_1)s, %(value_2)s, %(value_3)s, %(emitted_seq)s, %(payload_json)s
                )
                """,
                {
                    "session_id": session_id,
                    "sensor_type": event.sensor_type,
                    "source_ts": event.source_ts,
                    "emitted_at": datetime.now(),
                    "ingest_ts": datetime.now(),
                    "sample_rate_hz": event.sample_rate_hz,
                    "value_1": event.value_1,
                    "value_2": event.value_2,
                    "value_3": event.value_3,
                    "emitted_seq": event.emitted_seq,
                    "payload_json": Jsonb(event.payload_json),
                },
            )

    def run_pipeline_functions(
        self,
        conn: psycopg.Connection,
        session_id: int,
        sync_method: str,
        expected_events: int,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT sync_preprocessed_32hz(%s, %s)", (session_id, sync_method))
            cur.execute("SELECT integrate_biometric_logs(%s)", (session_id,))
            cur.execute("SELECT materialize_realtime_features(%s, %s)", (session_id, 60))
            cur.execute(
                "SELECT evaluate_system_metrics(%s, %s)",
                (session_id, expected_events),
            )
