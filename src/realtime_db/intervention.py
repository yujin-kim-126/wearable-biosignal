from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class InterventionResult:
    session_id: int
    learner_id: int
    window_end_ts: str
    model_version: str
    predicted_label: str
    predicted_confidence: float
    risk_score: float
    recommended_action: str
    should_alert: bool
    alert_suppressed: bool
    reason_text: str
    inference_id: int
    alert_id: int | None


def _safe_z(value: float | None, mean: float | None, std: float | None) -> float:
    if value is None or mean is None or std is None or std == 0:
        return 0.0
    return float((value - mean) / std)


def _compute_risk_and_action(hr_z: float, hrv_z: float, acc_z: float) -> tuple[float, str, str]:
    risk = 0.0
    reasons: list[str] = []
    if hr_z >= 1.2:
        risk += 0.4
        reasons.append("hr_high")
    if hrv_z <= -1.0:
        risk += 0.4
        reasons.append("hrv_low")
    if acc_z <= -0.8:
        risk += 0.2
        reasons.append("movement_drop")

    risk = max(0.0, min(1.0, risk))
    if risk >= 0.75:
        action = "short_break"
    elif risk >= 0.5:
        action = "task_pacing"
    elif risk >= 0.35:
        action = "attention_reminder"
    else:
        action = "continue_learning"

    reason = ",".join(reasons) if reasons else "stable_pattern"
    return risk, action, reason


def _severity_from_risk(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "high"
    if risk_score >= 0.5:
        return "medium"
    return "low"


def _message_for_action(action: str) -> str:
    if action == "short_break":
        return "생체신호 변동이 높습니다. 3~5분 휴식을 권고합니다."
    if action == "task_pacing":
        return "학습 부담 신호가 감지되었습니다. 문제 난이도/속도를 조절하세요."
    if action == "attention_reminder":
        return "집중 저하 신호가 감지되었습니다. 자세를 바꾸고 목표를 재확인하세요."
    return "현재 학습 상태가 안정적입니다. 학습을 이어가세요."


def _load_model_payload(root: Path, artifact_path: str) -> dict[str, Any]:
    model_path = root / artifact_path
    with model_path.open("rb") as f:
        return pickle.load(f)


def _json_safe_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
            continue
        out[k] = float(v) if isinstance(v, np.floating) else v
    return out


def _fetch_latest_unprocessed_window(
    conn: psycopg.Connection,
    session_id: int,
    window_seconds: int,
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                rf.session_id,
                s.learner_id,
                s.activity_type,
                rf.window_end_ts,
                rf.hr_mean,
                rf.hrv_sdnn,
                rf.acc_moving_avg,
                b.hr_mean_baseline,
                b.hrv_sdnn_baseline,
                b.acc_moving_avg_baseline,
                b.hr_mean_std,
                b.hrv_sdnn_std,
                b.acc_moving_avg_std
            FROM realtime_features rf
            JOIN learning_sessions s ON s.session_id = rf.session_id
            LEFT JOIN feature_baselines b
              ON b.learner_id = s.learner_id
             AND b.activity_type = s.activity_type
             AND b.window_seconds = rf.window_seconds
            WHERE rf.session_id = %s
              AND rf.window_seconds = %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM intervention_inference_log i
                  WHERE i.session_id = rf.session_id
                    AND i.window_end_ts = rf.window_end_ts
              )
            ORDER BY rf.window_end_ts DESC
            LIMIT 1
            """,
            (session_id, window_seconds),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = [d.name for d in cur.description]
    return dict(zip(keys, row))


def _fetch_latest_personal_model(
    conn: psycopg.Connection,
    learner_id: int,
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_id, model_version, artifact_path
            FROM personal_models
            WHERE learner_id = %s
              AND status IN ('trained', 'active')
            ORDER BY trained_at DESC, model_id DESC
            LIMIT 1
            """,
            (learner_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
    return {"model_id": int(row[0]), "model_version": str(row[1]), "artifact_path": str(row[2])}


def _is_alert_suppressed(
    conn: psycopg.Connection,
    session_id: int,
    action_type: str,
    window_end_ts: Any,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM intervention_alerts
            WHERE session_id = %s
              AND action_type = %s
              AND cooldown_until IS NOT NULL
              AND cooldown_until > %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id, action_type, window_end_ts),
        )
        return cur.fetchone() is not None


def run_intervention_once(
    *,
    database_url: str,
    project_root: Path,
    session_id: int,
    window_seconds: int = 60,
    cooldown_seconds: int = 300,
) -> InterventionResult | None:
    with psycopg.connect(database_url) as conn:
        source = _fetch_latest_unprocessed_window(conn, session_id=session_id, window_seconds=window_seconds)
        if source is None:
            return None

        learner_id = int(source["learner_id"])
        model_row = _fetch_latest_personal_model(conn, learner_id=learner_id)
        if model_row is None:
            raise RuntimeError(f"No personal model found for learner_id={learner_id}")

        payload = _load_model_payload(project_root, model_row["artifact_path"])
        model = payload["model"]
        imputer = payload["imputer"]
        label_encoder = payload["label_encoder"]
        feature_cols: list[str] = payload["feature_cols"]

        window_end_ts = pd.Timestamp(source["window_end_ts"])
        seconds_of_day = window_end_ts.hour * 3600 + window_end_ts.minute * 60 + window_end_ts.second
        dow = window_end_ts.dayofweek

        hr_z = _safe_z(source["hr_mean"], source["hr_mean_baseline"], source["hr_mean_std"])
        hrv_z = _safe_z(source["hrv_sdnn"], source["hrv_sdnn_baseline"], source["hrv_sdnn_std"])
        acc_z = _safe_z(source["acc_moving_avg"], source["acc_moving_avg_baseline"], source["acc_moving_avg_std"])

        feature_row = {
            "hr_mean": source["hr_mean"],
            "hrv_sdnn": source["hrv_sdnn"],
            "acc_moving_avg": source["acc_moving_avg"],
            "hr_mean_z": hr_z,
            "hrv_sdnn_z": hrv_z,
            "acc_moving_avg_z": acc_z,
            "hr_acc_interaction": hr_z * acc_z,
            "hrv_acc_interaction": hrv_z * acc_z,
            "hr_baseline_delta": (source["hr_mean"] or 0.0) - (source["hr_mean_baseline"] or 0.0),
            "hrv_baseline_delta": (source["hrv_sdnn"] or 0.0) - (source["hrv_sdnn_baseline"] or 0.0),
            "acc_baseline_delta": (source["acc_moving_avg"] or 0.0) - (source["acc_moving_avg_baseline"] or 0.0),
            "seconds_of_day": seconds_of_day,
            "dow": dow,
        }
        snapshot = _json_safe_dict(feature_row)
        x_df = pd.DataFrame([{c: feature_row.get(c, np.nan) for c in feature_cols}])
        x = imputer.transform(x_df)
        proba = model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = str(label_encoder.classes_[pred_idx])
        pred_conf = float(proba[pred_idx])

        risk_score, action, reason_text = _compute_risk_and_action(
            hr_z=hr_z,
            hrv_z=hrv_z,
            acc_z=acc_z,
        )
        should_alert = action != "continue_learning"
        suppressed = False
        if should_alert:
            suppressed = _is_alert_suppressed(
                conn,
                session_id=session_id,
                action_type=action,
                window_end_ts=window_end_ts.to_pydatetime(),
            )

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO intervention_inference_log (
                    session_id, learner_id, window_end_ts,
                    model_id, model_version, predicted_label, predicted_confidence,
                    risk_score, recommended_action, should_alert, alert_suppressed,
                    reason_text, feature_snapshot
                )
                VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s
                )
                RETURNING inference_id
                """,
                (
                    session_id,
                    learner_id,
                    window_end_ts.to_pydatetime(),
                    model_row["model_id"],
                    model_row["model_version"],
                    pred_label,
                    pred_conf,
                    risk_score,
                    action,
                    should_alert,
                    suppressed,
                    reason_text,
                    Jsonb(snapshot),
                ),
            )
            inference_id = int(cur.fetchone()[0])

            alert_id: int | None = None
            if should_alert and not suppressed:
                cooldown_until = window_end_ts.to_pydatetime() + timedelta(seconds=cooldown_seconds)
                cur.execute(
                    """
                    INSERT INTO intervention_alerts (
                        session_id, learner_id, inference_id,
                        action_type, severity, message_text, risk_score, cooldown_until
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING alert_id
                    """,
                    (
                        session_id,
                        learner_id,
                        inference_id,
                        action,
                        _severity_from_risk(risk_score),
                        _message_for_action(action),
                        risk_score,
                        cooldown_until,
                    ),
                )
                alert_id = int(cur.fetchone()[0])

        conn.commit()

    return InterventionResult(
        session_id=session_id,
        learner_id=learner_id,
        window_end_ts=str(window_end_ts),
        model_version=model_row["model_version"],
        predicted_label=pred_label,
        predicted_confidence=pred_conf,
        risk_score=risk_score,
        recommended_action=action,
        should_alert=should_alert,
        alert_suppressed=suppressed,
        reason_text=reason_text,
        inference_id=inference_id,
        alert_id=alert_id,
    )
