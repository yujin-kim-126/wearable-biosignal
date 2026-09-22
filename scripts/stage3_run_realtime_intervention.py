from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_db.config import load_config
from realtime_db.intervention import run_intervention_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime intervention analysis for a session.")
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--apply-sql", action="store_true")
    return parser.parse_args()


def apply_sql(conn: psycopg.Connection) -> None:
    sql_path = ROOT / "db" / "stage3_realtime_intervention.sql"
    sql_text = sql_path.read_text(encoding="utf-8-sig")
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    if args.apply_sql:
        with psycopg.connect(cfg.database_url) as conn:
            apply_sql(conn)

    rows: list[dict[str, object]] = []
    for i in range(args.iterations):
        result = run_intervention_once(
            database_url=cfg.database_url,
            project_root=ROOT,
            session_id=args.session_id,
            window_seconds=args.window_seconds,
            cooldown_seconds=args.cooldown_seconds,
        )
        if result is None:
            print(f"[{i+1}/{args.iterations}] no_new_window")
        else:
            row = {
                "session_id": result.session_id,
                "learner_id": result.learner_id,
                "window_end_ts": result.window_end_ts,
                "model_version": result.model_version,
                "predicted_label": result.predicted_label,
                "predicted_confidence": result.predicted_confidence,
                "risk_score": result.risk_score,
                "recommended_action": result.recommended_action,
                "should_alert": result.should_alert,
                "alert_suppressed": result.alert_suppressed,
                "reason_text": result.reason_text,
                "inference_id": result.inference_id,
                "alert_id": result.alert_id,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

        if i < args.iterations - 1:
            time.sleep(args.sleep_seconds)

    out_path = ROOT / "outputs" / f"stage3_realtime_intervention_session{args.session_id}.csv"
    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    else:
        out_path.write_text(
            "session_id,learner_id,window_end_ts,model_version,predicted_label,predicted_confidence,"
            "risk_score,recommended_action,should_alert,alert_suppressed,reason_text,inference_id,alert_id\n",
            encoding="utf-8",
        )
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
