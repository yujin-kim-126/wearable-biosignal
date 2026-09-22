from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.impute import SimpleImputer
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


def session_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def add_learner_robust_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        train_df.groupby("learner_id")[["hr_mean", "hrv_sdnn", "acc_moving_avg"]]
        .agg(["mean", "std", "median", lambda s: s.quantile(0.75) - s.quantile(0.25)])
        .reset_index()
    )
    stats.columns = [
        "learner_id",
        "hr_mean_mean",
        "hr_mean_std",
        "hr_mean_median",
        "hr_mean_iqr",
        "hrv_sdnn_mean",
        "hrv_sdnn_std",
        "hrv_sdnn_median",
        "hrv_sdnn_iqr",
        "acc_moving_avg_mean",
        "acc_moving_avg_std",
        "acc_moving_avg_median",
        "acc_moving_avg_iqr",
    ]
    out = target_df.merge(stats, on="learner_id", how="left")
    for feat in ["hr_mean", "hrv_sdnn", "acc_moving_avg"]:
        out[f"{feat}_learner_z"] = (out[feat] - out[f"{feat}_mean"]) / out[f"{feat}_std"].replace(0, np.nan)
        out[f"{feat}_learner_robust"] = (out[feat] - out[f"{feat}_median"]) / out[f"{feat}_iqr"].replace(0, np.nan)
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auc_ovr_macro": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
    }


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().sort_index()
    inv = 1.0 / counts
    mapping = (inv / inv.mean()).to_dict()
    return np.array([mapping[int(v)] for v in y], dtype=float)


def tune_xgboost(
    x_subtrain: np.ndarray,
    y_subtrain: np.ndarray,
    w_subtrain: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    candidates = [
        {"n_estimators": 220, "max_depth": 5, "learning_rate": 0.08, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 280, "max_depth": 6, "learning_rate": 0.06, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 320, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 240, "max_depth": 7, "learning_rate": 0.07, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 360, "max_depth": 5, "learning_rate": 0.04, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float] = (-1.0, -1.0)

    for cand in candidates:
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            **cand,
        )
        model.fit(x_subtrain, y_subtrain, sample_weight=w_subtrain)
        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)
        metrics = evaluate(y_val, pred, proba)
        key = (metrics["f1_macro"], metrics["auc_ovr_macro"])
        if key > best_key:
            best_key = key
            best_params = cand
            best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("Tuning failed to produce best params")
    return best_params, best_metrics


def main() -> None:
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(database_url)
    df = make_fused_features(df)

    train_df, test_df = session_split(df, test_size=0.3, seed=42)
    train_df = add_learner_robust_features(train_df, train_df)
    test_df = add_learner_robust_features(train_df, test_df)

    # Holdout for tuning: split train sessions into subtrain/val
    subtrain_df, val_df = session_split(train_df, test_size=0.25, seed=7)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["activity_type"])
    y_test = encoder.transform(test_df["activity_type"])
    y_subtrain = encoder.transform(subtrain_df["activity_type"])
    y_val = encoder.transform(val_df["activity_type"])
    class_names = list(encoder.classes_)

    feature_cols = [
        "hr_mean",
        "hrv_sdnn",
        "acc_moving_avg",
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
        "hr_mean_learner_z",
        "hrv_sdnn_learner_z",
        "acc_moving_avg_learner_z",
        "hr_mean_learner_robust",
        "hrv_sdnn_learner_robust",
        "acc_moving_avg_learner_robust",
    ]

    imp = SimpleImputer(strategy="median")
    x_train = imp.fit_transform(train_df[feature_cols])
    x_test = imp.transform(test_df[feature_cols])
    x_subtrain = imp.transform(subtrain_df[feature_cols])
    x_val = imp.transform(val_df[feature_cols])

    w_train = class_weights(y_train)
    w_subtrain = class_weights(y_subtrain)

    best_params, val_metrics = tune_xgboost(
        x_subtrain=x_subtrain,
        y_subtrain=y_subtrain,
        w_subtrain=w_subtrain,
        x_val=x_val,
        y_val=y_val,
        n_classes=len(class_names),
    )

    final_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(class_names),
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        **best_params,
    )
    final_model.fit(x_train, y_train, sample_weight=w_train)
    test_pred = final_model.predict(x_test)
    test_proba = final_model.predict_proba(x_test)
    test_metrics = evaluate(y_test, test_pred, test_proba)

    cm = confusion_matrix(y_test, test_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        outputs_dir / "stage5_confusion_xgboost_tuned.csv",
        encoding="utf-8",
    )

    metrics_df = pd.DataFrame(
        [
            {"split": "validation", "model": "xgboost_tuned", **val_metrics},
            {"split": "test", "model": "xgboost_tuned", **test_metrics},
        ]
    )
    metrics_df.to_csv(outputs_dir / "stage5_metrics.csv", index=False, encoding="utf-8")

    with (outputs_dir / "stage5_best_params.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(best_params, f, ensure_ascii=False, indent=2)

    report = outputs_dir / "stage5_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Stage 5 Tuning Report\n\n")
        f.write(f"- total_rows: {len(df)}\n")
        f.write(f"- train_rows: {len(train_df)}\n")
        f.write(f"- test_rows: {len(test_df)}\n")
        f.write(f"- subtrain_rows: {len(subtrain_df)}\n")
        f.write(f"- val_rows: {len(val_df)}\n")
        f.write(f"- classes: {class_names}\n\n")
        f.write("## Best Params\n")
        for k, v in best_params.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Metrics\n")
        for _, row in metrics_df.iterrows():
            f.write(
                f"- {row['split']}: accuracy={row['accuracy']:.6f}, "
                f"f1_macro={row['f1_macro']:.6f}, auc_ovr_macro={row['auc_ovr_macro']:.6f}\n"
            )
        f.write("\n## Outputs\n")
        f.write("- stage5_metrics.csv\n")
        f.write("- stage5_best_params.json\n")
        f.write("- stage5_confusion_xgboost_tuned.csv\n")

    print(report)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

