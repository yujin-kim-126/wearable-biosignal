from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import sys
import time

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_db.intervention import run_intervention_once


def fetch_db_metrics(database_url: str, batch_id: int, learner_code: str) -> dict[str, object]:
    out: dict[str, object] = {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT learner_id, learner_code, onboarding_status
                FROM learner_profiles
                WHERE learner_code = %s
                """,
                (learner_code,),
            )
            out["learner_profile"] = cur.fetchone()

            cur.execute(
                """
                SELECT
                    batch_id, learner_id, batch_status, total_sessions,
                    completed_sessions, failed_sessions, completed_days,
                    days_planned, total_events_collected
                FROM v_collection_batch_progress
                WHERE batch_id = %s
                """,
                (batch_id,),
            )
            out["batch_progress"] = cur.fetchone()

            cur.execute(
                """
                SELECT model_id, learner_id, batch_id, model_version, status, artifact_path
                FROM personal_models
                WHERE batch_id = %s
                ORDER BY model_id DESC
                LIMIT 1
                """,
                (batch_id,),
            )
            out["personal_model"] = cur.fetchone()

            cur.execute(
                """
                SELECT split_name, accuracy, f1_macro, auc_ovr_macro
                FROM personal_model_metrics
                WHERE model_id = (
                    SELECT max(model_id) FROM personal_models WHERE batch_id = %s
                )
                ORDER BY metric_id
                """,
                (batch_id,),
            )
            out["personal_model_metrics"] = cur.fetchall()

            cur.execute(
                """
                SELECT
                    COUNT(*) AS n_sessions,
                    AVG(latency_ms) AS avg_latency_ms,
                    AVG(throughput_req_per_sec) AS avg_throughput,
                    AVG(data_loss_rate_pct) AS avg_loss_pct,
                    SUM((latency_ms < 100.0)::int) AS pass_latency,
                    SUM((throughput_req_per_sec > 1000.0)::int) AS pass_throughput,
                    SUM((data_loss_rate_pct < 0.1)::int) AS pass_loss
                FROM system_metrics
                WHERE session_id IN (
                    SELECT session_id FROM collection_batch_sessions WHERE batch_id = %s
                )
                """,
                (batch_id,),
            )
            out["system_metrics"] = cur.fetchone()

            cur.execute(
                """
                SELECT COUNT(*) FROM intervention_inference_log
                WHERE session_id IN (
                    SELECT session_id FROM collection_batch_sessions WHERE batch_id = %s
                )
                """,
                (batch_id,),
            )
            out["inference_count_batch"] = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*) FROM intervention_alerts
                WHERE session_id IN (
                    SELECT session_id FROM collection_batch_sessions WHERE batch_id = %s
                )
                """,
                (batch_id,),
            )
            out["alert_count_batch"] = cur.fetchone()[0]
    return out


def benchmark_intervention(
    database_url: str,
    session_id: int,
    iterations: int,
) -> dict[str, float | int]:
    rows: list[float] = []
    should_alert = 0
    suppressed = 0

    start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        r = run_intervention_once(
            database_url=database_url,
            project_root=ROOT,
            session_id=session_id,
            window_seconds=60,
            cooldown_seconds=300,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if r is None:
            break
        rows.append(dt)
        should_alert += int(r.should_alert)
        suppressed += int(r.alert_suppressed)
    total_ms = (time.perf_counter() - start) * 1000.0

    if rows:
        rows_sorted = sorted(rows)
        p95_idx = max(0, int(len(rows_sorted) * 0.95) - 1)
        avg_ms = float(sum(rows) / len(rows))
        p95_ms = float(rows_sorted[p95_idx])
    else:
        avg_ms = 0.0
        p95_ms = 0.0

    return {
        "iterations_requested": iterations,
        "iterations_processed": len(rows),
        "avg_inference_ms": avg_ms,
        "p95_inference_ms": p95_ms,
        "total_elapsed_ms": float(total_ms),
        "should_alert_count": should_alert,
        "suppressed_count": suppressed,
    }


def main() -> None:
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    batch_id = int(os.getenv("FINAL_EVAL_BATCH_ID", "1"))
    learner_code = os.getenv("FINAL_EVAL_LEARNER_CODE", "L2026_STAGE1_001")
    benchmark_session_id = int(os.getenv("FINAL_EVAL_SESSION_ID", "31"))
    benchmark_iterations = int(os.getenv("FINAL_EVAL_ITERATIONS", "120"))

    db_metrics = fetch_db_metrics(database_url, batch_id=batch_id, learner_code=learner_code)
    bench = benchmark_intervention(
        database_url=database_url,
        session_id=benchmark_session_id,
        iterations=benchmark_iterations,
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_json = ROOT / "outputs" / "final_framework_evaluation.json"
    out_md = ROOT / "outputs" / "final_framework_evaluation.md"
    out_csv = ROOT / "outputs" / "final_framework_evaluation_summary.csv"

    payload = {
        "evaluated_at": now,
        "batch_id": batch_id,
        "learner_code": learner_code,
        "db_metrics": db_metrics,
        "intervention_benchmark": bench,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    learner_profile = db_metrics.get("learner_profile")
    batch_progress = db_metrics.get("batch_progress")
    model_row = db_metrics.get("personal_model")
    model_metrics = db_metrics.get("personal_model_metrics") or []
    sysm = db_metrics.get("system_metrics")

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Final Framework Performance Evaluation\n\n")
        f.write(f"- evaluated_at: {now}\n")
        f.write(f"- batch_id: {batch_id}\n")
        f.write(f"- learner_code: {learner_code}\n\n")

        f.write("## Completion Check\n")
        f.write(f"- step1_onboarding_week_collection: {bool(batch_progress and batch_progress[2] == 'completed')}\n")
        f.write(f"- step2_personal_model_training: {bool(model_row is not None)}\n")
        f.write(f"- step3_realtime_intervention_active: {bool((db_metrics.get('inference_count_batch') or 0) > 0)}\n\n")

        f.write("## Learner/Batch Status\n")
        f.write(f"- learner_profile: {learner_profile}\n")
        f.write(f"- batch_progress: {batch_progress}\n")
        f.write(f"- personal_model: {model_row}\n")
        f.write(f"- personal_model_metrics: {model_metrics}\n\n")

        if sysm:
            n_sessions, avg_latency, avg_throughput, avg_loss, pass_lat, pass_thr, pass_loss = sysm
            f.write("## System Metrics (batch sessions)\n")
            f.write(f"- n_sessions: {n_sessions}\n")
            f.write(f"- avg_latency_ms: {avg_latency}\n")
            f.write(f"- avg_throughput_req_per_sec: {avg_throughput}\n")
            f.write(f"- avg_data_loss_rate_pct: {avg_loss}\n")
            f.write(f"- pass_latency_target(<100ms): {pass_lat}/{n_sessions}\n")
            f.write(f"- pass_throughput_target(>1000req/s): {pass_thr}/{n_sessions}\n")
            f.write(f"- pass_loss_target(<0.1%): {pass_loss}/{n_sessions}\n\n")

        f.write("## Intervention Runtime Benchmark\n")
        for k, v in bench.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"- inference_count_batch: {db_metrics.get('inference_count_batch')}\n")
        f.write(f"- alert_count_batch: {db_metrics.get('alert_count_batch')}\n")

    summary_rows = [
        {"metric": "batch_status", "value": batch_progress[2] if batch_progress else None},
        {"metric": "completed_sessions", "value": batch_progress[4] if batch_progress else None},
        {"metric": "planned_sessions", "value": batch_progress[3] if batch_progress else None},
        {"metric": "completed_days", "value": batch_progress[6] if batch_progress else None},
        {"metric": "planned_days", "value": batch_progress[7] if batch_progress else None},
        {"metric": "total_events_collected", "value": batch_progress[8] if batch_progress else None},
        {"metric": "model_version", "value": model_row[3] if model_row else None},
        {"metric": "model_status", "value": model_row[4] if model_row else None},
        {"metric": "test_accuracy", "value": model_metrics[0][1] if model_metrics else None},
        {"metric": "test_f1_macro", "value": model_metrics[0][2] if model_metrics else None},
        {"metric": "test_auc_ovr_macro", "value": model_metrics[0][3] if model_metrics else None},
        {"metric": "avg_system_latency_ms", "value": sysm[1] if sysm else None},
        {"metric": "avg_system_throughput_req_per_sec", "value": sysm[2] if sysm else None},
        {"metric": "avg_system_data_loss_pct", "value": sysm[3] if sysm else None},
        {"metric": "benchmark_avg_inference_ms", "value": bench["avg_inference_ms"]},
        {"metric": "benchmark_p95_inference_ms", "value": bench["p95_inference_ms"]},
        {"metric": "benchmark_processed_windows", "value": bench["iterations_processed"]},
        {"metric": "inference_count_batch", "value": db_metrics.get("inference_count_batch")},
        {"metric": "alert_count_batch", "value": db_metrics.get("alert_count_batch")},
    ]
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding="utf-8")

    print(out_md)
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
