from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg


ROOT = Path(__file__).resolve().parents[1]


def _sync_compare_for_session(
    conn: psycopg.Connection,
    session_id: int,
) -> list[dict[str, object]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sensor_type, source_ts, value_1, value_2, value_3
            FROM sensor_event_stream
            WHERE session_id = %s
              AND sensor_type IN ('EDA', 'HR', 'TEMP', 'ACC')
            ORDER BY source_ts
            """,
            (session_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return []

    frame = pd.DataFrame(rows, columns=["sensor_type", "source_ts", "value_1", "value_2", "value_3"])
    frame["source_ts"] = pd.to_datetime(frame["source_ts"])
    start_ts = frame["source_ts"].min()
    end_ts = frame["source_ts"].max()
    grid = pd.date_range(start=start_ts, end=end_ts, freq="31250us")

    outputs: list[dict[str, object]] = []

    def build_series(sensor: str, col: str) -> pd.Series:
        s = (
            frame.loc[frame["sensor_type"] == sensor, ["source_ts", col]]
            .dropna()
            .drop_duplicates(subset=["source_ts"], keep="last")
            .set_index("source_ts")[col]
            .sort_index()
        )
        return s

    signals = [
        ("EDA", "value_1", "eda"),
        ("HR", "value_1", "hr"),
        ("TEMP", "value_1", "temp"),
        ("ACC", "value_1", "acc_x"),
        ("ACC", "value_2", "acc_y"),
        ("ACC", "value_3", "acc_z"),
    ]

    for sensor, col, signal_name in signals:
        series = build_series(sensor, col)
        if series.empty:
            outputs.append(
                {
                    "session_id": session_id,
                    "signal_name": signal_name,
                    "grid_points": len(grid),
                    "raw_points": 0,
                    "null_ratio_zoh": 1.0,
                    "null_ratio_linear": 1.0,
                    "interp_ratio": 1.0,
                    "mean_abs_diff_zoh_linear": None,
                }
            )
            continue

        zoh = series.reindex(grid).ffill()
        linear = series.reindex(grid).interpolate(method="time")
        null_ratio_zoh = float(zoh.isna().mean())
        null_ratio_linear = float(linear.isna().mean())
        exact_match_ratio = float(series.index.isin(grid).mean())
        interp_ratio = 1.0 - exact_match_ratio
        mad = (zoh - linear).abs().mean()
        mad_val = None if pd.isna(mad) else float(mad)

        outputs.append(
            {
                "session_id": session_id,
                "signal_name": signal_name,
                "grid_points": len(grid),
                "raw_points": int(series.shape[0]),
                "null_ratio_zoh": null_ratio_zoh,
                "null_ratio_linear": null_ratio_linear,
                "interp_ratio": interp_ratio,
                "mean_abs_diff_zoh_linear": mad_val,
            }
        )

    return outputs


def main() -> None:
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(database_url) as conn:
        # 1) Baseline SQL apply
        baseline_sql = (ROOT / "db" / "stage3_baseline.sql").read_text(encoding="utf-8-sig")
        with conn.cursor() as cur:
            cur.execute(baseline_sql)
        conn.commit()

        # 2) Refresh baselines
        with conn.cursor() as cur:
            cur.execute("SELECT refresh_feature_baselines(%s)", (60,))
            baseline_rows = cur.fetchone()[0]
        conn.commit()

        # 3) Build per-session sync comparison
        with conn.cursor() as cur:
            cur.execute("SELECT session_id FROM learning_sessions ORDER BY session_id")
            session_ids = [r[0] for r in cur.fetchall()]

        sync_rows: list[dict[str, object]] = []
        for sid in session_ids:
            sync_rows.extend(_sync_compare_for_session(conn, sid))

        sync_df = pd.DataFrame(sync_rows)
        sync_csv = outputs_dir / "stage3_sync_compare.csv"
        sync_df.to_csv(sync_csv, index=False, encoding="utf-8")

        # 4) Session quality from preprocessed_stream_32hz
        quality_sql = """
            SELECT
                session_id,
                COUNT(*) AS n_rows,
                AVG(CASE WHEN eda IS NULL THEN 1.0 ELSE 0.0 END) AS eda_null_ratio,
                AVG(CASE WHEN hr IS NULL THEN 1.0 ELSE 0.0 END) AS hr_null_ratio,
                AVG(CASE WHEN acc_x IS NULL OR acc_y IS NULL OR acc_z IS NULL THEN 1.0 ELSE 0.0 END) AS acc_null_ratio,
                AVG(CASE WHEN ibi_seconds IS NULL THEN 1.0 ELSE 0.0 END) AS ibi_null_ratio
            FROM preprocessed_stream_32hz
            GROUP BY session_id
            ORDER BY session_id
        """
        quality_df = pd.read_sql_query(quality_sql, conn)
        quality_csv = outputs_dir / "stage3_session_quality.csv"
        quality_df.to_csv(quality_csv, index=False, encoding="utf-8")

        # 5) Counts for normalized model input
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feature_baselines")
            baseline_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM v_model_input_features_norm_v1")
            norm_count = cur.fetchone()[0]

    # 6) Markdown summary
    md = outputs_dir / "stage3_summary.md"
    sync_mad = sync_df["mean_abs_diff_zoh_linear"].dropna()
    mean_mad = None if sync_mad.empty else float(sync_mad.mean())
    with md.open("w", encoding="utf-8") as f:
        f.write("# Stage 3 Summary\n\n")
        f.write(f"- feature_baselines rows: {baseline_count}\n")
        f.write(f"- normalized model rows: {norm_count}\n")
        f.write(f"- sync compare rows: {len(sync_df)}\n")
        f.write(f"- session quality rows: {len(quality_df)}\n")
        f.write(f"- mean(|zoh-linear|): {mean_mad}\n")
        f.write("\n## Files\n")
        f.write(f"- {sync_csv.name}\n")
        f.write(f"- {quality_csv.name}\n")

    print(md)
    print(sync_csv)
    print(quality_csv)
    print("baseline_rows", baseline_count, "norm_rows", norm_count)


if __name__ == "__main__":
    main()

