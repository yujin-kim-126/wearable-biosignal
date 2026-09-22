from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import psycopg
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_db.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train personalized model from one completed collection batch.")
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--model-version", default=None, help="Default: auto timestamp version")
    parser.add_argument("--apply-sql", action="store_true", help="Apply stage2_personal_training.sql before training")
    return parser.parse_args()


def apply_sql(conn: psycopg.Connection) -> None:
    sql_path = ROOT / "db" / "stage2_personal_training.sql"
    sql_text = sql_path.read_text(encoding="utf-8-sig")
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()


def refresh_baselines(conn: psycopg.Connection, window_seconds: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT refresh_feature_baselines(%s)", (window_seconds,))
    conn.commit()


def load_batch_data(conn: psycopg.Connection, batch_id: int, window_seconds: int) -> tuple[pd.DataFrame, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT learner_id FROM collection_batches WHERE batch_id=%s", (batch_id,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"batch_id={batch_id} not found")
        learner_id = int(row[0])

    sql = """
        SELECT
            v.learner_id,
            v.session_id,
            v.activity_type,
            v.window_end_ts,
            v.hr_mean,
            v.hrv_sdnn,
            v.acc_moving_avg,
            v.hr_mean_baseline,
            v.hrv_sdnn_baseline,
            v.acc_moving_avg_baseline,
            v.hr_mean_z,
            v.hrv_sdnn_z,
            v.acc_moving_avg_z
        FROM v_model_input_features_norm_v1 v
        JOIN collection_batch_sessions cbs
          ON cbs.session_id = v.session_id
        WHERE cbs.batch_id = %(batch_id)s
          AND cbs.status = 'completed'
          AND v.window_seconds = %(window_seconds)s
        ORDER BY v.session_id, v.window_end_ts
    """
    df = pd.read_sql_query(sql, conn, params={"batch_id": batch_id, "window_seconds": window_seconds})
    if df.empty:
        raise RuntimeError("No completed batch session data found for training")
    return df, learner_id


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["window_end_ts"] = pd.to_datetime(out["window_end_ts"])
    out["seconds_of_day"] = (
        out["window_end_ts"].dt.hour * 3600
        + out["window_end_ts"].dt.minute * 60
        + out["window_end_ts"].dt.second
    )
    out["dow"] = out["window_end_ts"].dt.dayofweek

    out["hr_acc_interaction"] = out["hr_mean_z"] * out["acc_moving_avg_z"]
    out["hrv_acc_interaction"] = out["hrv_sdnn_z"] * out["acc_moving_avg_z"]
    out["hr_baseline_delta"] = out["hr_mean"] - out["hr_mean_baseline"]
    out["hrv_baseline_delta"] = out["hrv_sdnn"] - out["hrv_sdnn_baseline"]
    out["acc_baseline_delta"] = out["acc_moving_avg"] - out["acc_moving_avg_baseline"]

    feature_cols = [
        "hr_mean",
        "hrv_sdnn",
        "acc_moving_avg",
        "hr_mean_z",
        "hrv_sdnn_z",
        "acc_moving_avg_z",
        "hr_acc_interaction",
        "hrv_acc_interaction",
        "hr_baseline_delta",
        "hrv_baseline_delta",
        "acc_baseline_delta",
        "seconds_of_day",
        "dow",
    ]
    return out, feature_cols


def split_by_session(df: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    session_map = df[["session_id", "activity_type"]].drop_duplicates().sort_values("session_id")
    train_sessions, test_sessions = train_test_split(
        session_map,
        test_size=0.3,
        random_state=42,
        stratify=session_map["activity_type"],
    )
    train_idx = np.where(df["session_id"].isin(train_sessions["session_id"]))[0]
    test_idx = np.where(df["session_id"].isin(test_sessions["session_id"]))[0]
    return train_idx, test_idx


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auc_ovr_macro": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config()

    with psycopg.connect(cfg.database_url) as conn:
        if args.apply_sql:
            apply_sql(conn)
        refresh_baselines(conn, args.window_seconds)
        df, learner_id = load_batch_data(conn, args.batch_id, args.window_seconds)

    feature_df, feature_cols = build_features(df)
    encoder = LabelEncoder()
    y_all = encoder.fit_transform(feature_df["activity_type"])
    train_idx, test_idx = split_by_session(feature_df, y_all)

    imp = SimpleImputer(strategy="median")
    x_train = imp.fit_transform(feature_df.iloc[train_idx][feature_cols])
    x_test = imp.transform(feature_df.iloc[test_idx][feature_cols])
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(encoder.classes_),
        n_estimators=280,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)
    metrics = evaluate(y_test, pred, proba)

    version = args.model_version or datetime.now().strftime("v%Y%m%d_%H%M%S")
    artifact_name = f"personal_model_batch{args.batch_id}_{version}.pkl"
    artifact_path = ROOT / "outputs" / artifact_name
    artifact_payload = {
        "model": model,
        "imputer": imp,
        "label_encoder": encoder,
        "feature_cols": feature_cols,
        "learner_id": learner_id,
        "batch_id": args.batch_id,
        "window_seconds": args.window_seconds,
    }
    with artifact_path.open("wb") as f:
        pickle.dump(artifact_payload, f)

    metrics_row = {
        "batch_id": args.batch_id,
        "learner_id": learner_id,
        "model_version": version,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_sessions": int(feature_df.iloc[train_idx]["session_id"].nunique()),
        "test_sessions": int(feature_df.iloc[test_idx]["session_id"].nunique()),
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "auc_ovr_macro": metrics["auc_ovr_macro"],
        "classes": ",".join(str(c) for c in encoder.classes_),
        "artifact_path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
    }
    metrics_df = pd.DataFrame([metrics_row])
    metrics_csv = ROOT / "outputs" / f"stage2_personal_model_metrics_batch{args.batch_id}.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")

    report_md = ROOT / "outputs" / f"stage2_personal_model_report_batch{args.batch_id}.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Stage2 Personalized Model Report\n\n")
        for k, v in metrics_row.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Feature Columns\n")
        for c in feature_cols:
            f.write(f"- {c}\n")

    with psycopg.connect(cfg.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO personal_models (
                    learner_id, batch_id, model_version, target_name, feature_set, algorithm,
                    train_rows, test_rows, train_sessions, test_sessions, artifact_path, status, notes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'trained', %s)
                RETURNING model_id
                """,
                (
                    learner_id,
                    args.batch_id,
                    version,
                    "activity_type",
                    "physio+normalized+context_time",
                    "xgboost",
                    metrics_row["train_rows"],
                    metrics_row["test_rows"],
                    metrics_row["train_sessions"],
                    metrics_row["test_sessions"],
                    metrics_row["artifact_path"],
                    "stage2 personalized training from one-week batch",
                ),
            )
            model_id = int(cur.fetchone()[0])

            cur.execute(
                """
                INSERT INTO personal_model_metrics (model_id, split_name, accuracy, f1_macro, auc_ovr_macro)
                VALUES (%s, 'test', %s, %s, %s)
                """,
                (model_id, metrics["accuracy"], metrics["f1_macro"], metrics["auc_ovr_macro"]),
            )

            cur.execute(
                """
                UPDATE learner_profiles
                SET onboarding_status = 'ready_for_training', updated_at = CURRENT_TIMESTAMP
                WHERE learner_id = %s
                """,
                (learner_id,),
            )
        conn.commit()

    print(json.dumps(metrics_row, ensure_ascii=False, indent=2))
    print(f"report={report_md}")
    print(f"metrics={metrics_csv}")
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
