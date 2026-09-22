from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]


def load_data(database_url: str) -> pd.DataFrame:
    sql = """
        SELECT
            learner_id,
            session_id,
            activity_type,
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
    """
    with psycopg.connect(database_url) as conn:
        return pd.read_sql_query(sql, conn)


def make_fused_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hr_acc_interaction"] = out["hr_mean_z"] * out["acc_moving_avg_z"]
    out["hrv_acc_interaction"] = out["hrv_sdnn_z"] * out["acc_moving_avg_z"]
    out["hr_baseline_delta"] = out["hr_mean"] - out["hr_mean_baseline"]
    out["hrv_baseline_delta"] = out["hrv_sdnn"] - out["hrv_sdnn_baseline"]
    out["acc_baseline_delta"] = out["acc_moving_avg"] - out["acc_moving_avg_baseline"]
    return out


def session_split(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_map = df[["session_id", "activity_type"]].drop_duplicates().sort_values("session_id")
    train_sessions, test_sessions = train_test_split(
        session_map,
        test_size=test_size,
        random_state=seed,
        stratify=session_map["activity_type"],
    )
    train_df = df[df["session_id"].isin(train_sessions["session_id"])].copy()
    test_df = df[df["session_id"].isin(test_sessions["session_id"])].copy()
    return train_df, test_df


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auc_ovr_macro": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
    }


def main() -> None:
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(database_url)
    df = make_fused_features(df)

    # core columns
    baseline_cols = ["hr_mean", "hrv_sdnn", "acc_moving_avg"]
    fused_cols = baseline_cols + [
        "hr_mean_baseline",
        "hrv_sdnn_baseline",
        "acc_moving_avg_baseline",
        "hr_mean_z",
        "hrv_sdnn_z",
        "acc_moving_avg_z",
        "hr_acc_interaction",
        "hrv_acc_interaction",
        "hr_baseline_delta",
        "hrv_baseline_delta",
        "acc_baseline_delta",
    ]

    train_df, test_df = session_split(df)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["activity_type"])
    y_test = encoder.transform(test_df["activity_type"])
    class_names = list(encoder.classes_)

    # baseline model
    base_imp = SimpleImputer(strategy="median")
    x_train_base = base_imp.fit_transform(train_df[baseline_cols])
    x_test_base = base_imp.transform(test_df[baseline_cols])

    base_model = LogisticRegression(max_iter=500, multi_class="auto", n_jobs=1)
    base_model.fit(x_train_base, y_train)
    base_pred = base_model.predict(x_test_base)
    base_proba = base_model.predict_proba(x_test_base)
    base_metrics = evaluate_multiclass(y_test, base_pred, base_proba)

    # main fused model (XGBoost)
    main_imp = SimpleImputer(strategy="median")
    x_train_main = main_imp.fit_transform(train_df[fused_cols])
    x_test_main = main_imp.transform(test_df[fused_cols])

    main_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(class_names),
        n_estimators=220,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )
    main_model.fit(x_train_main, y_train)
    main_pred = main_model.predict(x_test_main)
    main_proba = main_model.predict_proba(x_test_main)
    main_metrics = evaluate_multiclass(y_test, main_pred, main_proba)

    # save confusion matrices
    base_cm = confusion_matrix(y_test, base_pred, labels=np.arange(len(class_names)))
    main_cm = confusion_matrix(y_test, main_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(base_cm, index=class_names, columns=class_names).to_csv(
        outputs_dir / "stage4_confusion_baseline.csv",
        encoding="utf-8",
    )
    pd.DataFrame(main_cm, index=class_names, columns=class_names).to_csv(
        outputs_dir / "stage4_confusion_xgboost.csv",
        encoding="utf-8",
    )

    metrics_df = pd.DataFrame(
        [
            {"model": "baseline_logreg", **base_metrics},
            {"model": "xgboost_fused", **main_metrics},
        ]
    )
    metrics_path = outputs_dir / "stage4_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    report_path = outputs_dir / "stage4_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Stage 4 Modeling Report\n\n")
        f.write(f"- total_rows: {len(df)}\n")
        f.write(f"- train_rows: {len(train_df)}\n")
        f.write(f"- test_rows: {len(test_df)}\n")
        f.write(f"- train_sessions: {train_df['session_id'].nunique()}\n")
        f.write(f"- test_sessions: {test_df['session_id'].nunique()}\n")
        f.write(f"- classes: {class_names}\n\n")
        f.write("## Metrics\n")
        for _, row in metrics_df.iterrows():
            f.write(
                f"- {row['model']}: accuracy={row['accuracy']:.6f}, "
                f"f1_macro={row['f1_macro']:.6f}, auc_ovr_macro={row['auc_ovr_macro']:.6f}\n"
            )
        f.write("\n## Outputs\n")
        f.write("- stage4_metrics.csv\n")
        f.write("- stage4_confusion_baseline.csv\n")
        f.write("- stage4_confusion_xgboost.csv\n")

    print(report_path)
    print(metrics_path)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

