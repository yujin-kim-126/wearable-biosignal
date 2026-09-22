from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]


def load_data(database_url: str) -> pd.DataFrame:
    sql = """
        SELECT
            learner_id,
            session_id,
            activity_type,
            window_end_ts,
            hr_mean,
            hrv_sdnn,
            acc_moving_avg,
            hr_mean_baseline,
            hrv_sdnn_baseline,
            acc_moving_avg_baseline,
            hr_mean_z,
            hrv_sdnn_z,
            acc_moving_avg_z
        FROM v_model_input_features_norm_v1
        WHERE learner_id BETWEEN 1 AND 10
        ORDER BY learner_id, session_id, window_end_ts
    """
    with psycopg.connect(database_url) as conn:
        df = pd.read_sql_query(sql, conn)
    df["window_end_ts"] = pd.to_datetime(df["window_end_ts"])
    return df


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
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

    cols = [
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
    return out, cols


def split_per_session_time(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    train_idx_parts: list[np.ndarray] = []
    test_idx_parts: list[np.ndarray] = []
    for session_id, g in df.groupby("session_id", sort=True):
        idx = g.sort_values("window_end_ts").index.to_numpy()
        cut = int(len(idx) * train_ratio)
        cut = max(1, min(cut, len(idx) - 1))
        train_idx_parts.append(idx[:cut])
        test_idx_parts.append(idx[cut:])

    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    return train_idx, test_idx


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auc_ovr_macro": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
    }


def main() -> None:
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(database_url)
    feat, feature_cols = make_features(raw)

    rows: list[dict[str, float | int | str]] = []
    for learner_id in range(1, 11):
        df_l = feat[feat["learner_id"] == learner_id].copy()
        train_idx, test_idx = split_per_session_time(df_l, train_ratio=0.7)

        encoder = LabelEncoder()
        y_all = encoder.fit_transform(df_l["activity_type"])

        imp = SimpleImputer(strategy="median")
        x_train = imp.fit_transform(df_l.loc[train_idx, feature_cols])
        x_test = imp.transform(df_l.loc[test_idx, feature_cols])
        y_train = y_all[np.isin(df_l.index, train_idx)]
        y_test = y_all[np.isin(df_l.index, test_idx)]

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
        m = evaluate(y_test, pred, proba)

        rows.append(
            {
                "learner_id": learner_id,
                "subject_code_assumed": f"S{learner_id}",
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "auc_ovr_macro": m["auc_ovr_macro"],
            }
        )

    result = pd.DataFrame(rows)
    result_csv = out_dir / "final_eval_existing_10_students.csv"
    result.to_csv(result_csv, index=False, encoding="utf-8")

    summary = out_dir / "final_eval_existing_10_students.md"
    with summary.open("w", encoding="utf-8") as f:
        f.write("# Existing 10 Students Performance Evaluation\n\n")
        f.write("- model: personalized xgboost (same hyperparameters as stage2)\n")
        f.write("- split: per-student, per-session chronological 70% train / 30% test\n")
        f.write("- features: physio + baseline-normalized + time context\n\n")
        f.write("## Per Student Metrics\n")
        for _, r in result.iterrows():
            f.write(
                f"- learner_id={int(r['learner_id'])} ({r['subject_code_assumed']}): "
                f"acc={r['accuracy']:.6f}, f1={r['f1_macro']:.6f}, auc={r['auc_ovr_macro']:.6f}, "
                f"train={int(r['train_rows'])}, test={int(r['test_rows'])}\n"
            )

        f.write("\n## Aggregate\n")
        f.write(f"- accuracy_mean: {result['accuracy'].mean():.6f}\n")
        f.write(f"- f1_macro_mean: {result['f1_macro'].mean():.6f}\n")
        f.write(f"- auc_ovr_macro_mean: {result['auc_ovr_macro'].mean():.6f}\n")
        f.write(f"- accuracy_min: {result['accuracy'].min():.6f}\n")
        f.write(f"- f1_macro_min: {result['f1_macro'].min():.6f}\n")
        f.write(f"- auc_ovr_macro_min: {result['auc_ovr_macro'].min():.6f}\n")

    print(result_csv)
    print(summary)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
