BEGIN;

CREATE OR REPLACE FUNCTION sync_preprocessed_32hz(
    p_session_id BIGINT,
    p_sync_method VARCHAR DEFAULT 'zoh'
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_start_ts TIMESTAMP;
    v_end_ts TIMESTAMP;
    v_rows INT := 0;
BEGIN
    IF p_sync_method NOT IN ('zoh', 'linear') THEN
        RAISE EXCEPTION 'Unsupported sync method: %', p_sync_method;
    END IF;

    SELECT MIN(source_ts), MAX(source_ts)
    INTO v_start_ts, v_end_ts
    FROM sensor_event_stream
    WHERE session_id = p_session_id
      AND sensor_type IN ('EDA', 'HR', 'ACC', 'IBI');

    IF v_start_ts IS NULL OR v_end_ts IS NULL THEN
        RETURN 0;
    END IF;

    DELETE FROM preprocessed_stream_32hz WHERE session_id = p_session_id;

    WITH grid AS (
        SELECT generate_series(
            v_start_ts,
            v_end_ts,
            INTERVAL '0.03125 second'
        ) AS ts_32hz
    ),
    joined AS (
        SELECT
            g.ts_32hz,
            (
                SELECT s.value_1
                FROM sensor_event_stream s
                WHERE s.session_id = p_session_id AND s.sensor_type = 'EDA'
                  AND s.source_ts <= g.ts_32hz
                ORDER BY s.source_ts DESC
                LIMIT 1
            ) AS eda,
            (
                SELECT s.value_1
                FROM sensor_event_stream s
                WHERE s.session_id = p_session_id AND s.sensor_type = 'HR'
                  AND s.source_ts <= g.ts_32hz
                ORDER BY s.source_ts DESC
                LIMIT 1
            ) AS hr,
            (
                SELECT s.value_1
                FROM sensor_event_stream s
                WHERE s.session_id = p_session_id AND s.sensor_type = 'ACC'
                  AND s.source_ts <= g.ts_32hz
                ORDER BY s.source_ts DESC
                LIMIT 1
            ) AS acc_x,
            (
                SELECT s.value_2
                FROM sensor_event_stream s
                WHERE s.session_id = p_session_id AND s.sensor_type = 'ACC'
                  AND s.source_ts <= g.ts_32hz
                ORDER BY s.source_ts DESC
                LIMIT 1
            ) AS acc_y,
            (
                SELECT s.value_3
                FROM sensor_event_stream s
                WHERE s.session_id = p_session_id AND s.sensor_type = 'ACC'
                  AND s.source_ts <= g.ts_32hz
                ORDER BY s.source_ts DESC
                LIMIT 1
            ) AS acc_z,
            (
                SELECT s.value_1
                FROM sensor_event_stream s
                WHERE s.session_id = p_session_id AND s.sensor_type = 'IBI'
                  AND s.source_ts <= g.ts_32hz
                ORDER BY s.source_ts DESC
                LIMIT 1
            ) AS ibi_seconds
        FROM grid g
    )
    INSERT INTO preprocessed_stream_32hz (
        session_id, ts_32hz, sync_method, is_interpolated, source_event_count,
        eda, hr, acc_x, acc_y, acc_z, ibi_seconds
    )
    SELECT
        p_session_id,
        j.ts_32hz,
        p_sync_method,
        FALSE,
        1,
        j.eda,
        j.hr,
        j.acc_x,
        j.acc_y,
        j.acc_z,
        j.ibi_seconds
    FROM joined j;

    GET DIAGNOSTICS v_rows = ROW_COUNT;
    RETURN v_rows;
END;
$$;

CREATE OR REPLACE FUNCTION integrate_biometric_logs(p_session_id BIGINT)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows INT := 0;
BEGIN
    DELETE FROM biometric_logs WHERE session_id = p_session_id;

    INSERT INTO biometric_logs (
        session_id, ts, eda, hr, acc_x, acc_y, acc_z, ibi_seconds
    )
    SELECT
        session_id,
        ts_32hz,
        eda,
        hr,
        acc_x,
        acc_y,
        acc_z,
        ibi_seconds
    FROM preprocessed_stream_32hz
    WHERE session_id = p_session_id
    ORDER BY ts_32hz;

    GET DIAGNOSTICS v_rows = ROW_COUNT;
    RETURN v_rows;
END;
$$;

CREATE OR REPLACE FUNCTION materialize_realtime_features(
    p_session_id BIGINT,
    p_window_seconds INT DEFAULT 60
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows INT := 0;
BEGIN
    DELETE FROM realtime_features WHERE session_id = p_session_id AND window_seconds = p_window_seconds;

    INSERT INTO realtime_features (
        session_id, window_end_ts, window_seconds, hr_mean, hrv_sdnn, acc_moving_avg
    )
    SELECT
        v.session_id,
        v.window_end_ts,
        p_window_seconds,
        v.hr_mean_60s,
        v.hrv_sdnn_60s,
        v.acc_moving_avg_60s
    FROM v_realtime_feature_60s v
    WHERE v.session_id = p_session_id;

    GET DIAGNOSTICS v_rows = ROW_COUNT;
    RETURN v_rows;
END;
$$;

CREATE OR REPLACE FUNCTION evaluate_system_metrics(
    p_session_id BIGINT,
    p_expected_events INT
)
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    v_inserted INT := 0;
    v_latency_ms DOUBLE PRECISION := NULL;
    v_throughput DOUBLE PRECISION := NULL;
    v_loss_rate DOUBLE PRECISION := NULL;
    v_metric_id BIGINT;
BEGIN
    SELECT COUNT(*)
    INTO v_inserted
    FROM sensor_event_stream
    WHERE session_id = p_session_id;

    SELECT AVG(EXTRACT(EPOCH FROM (ingest_ts - emitted_at)) * 1000.0)
    INTO v_latency_ms
    FROM sensor_event_stream
    WHERE session_id = p_session_id;

    SELECT
        CASE
            WHEN (MAX(ingest_ts) - MIN(ingest_ts)) > INTERVAL '0 second'
            THEN COUNT(*) / EXTRACT(EPOCH FROM (MAX(ingest_ts) - MIN(ingest_ts)))
            ELSE NULL
        END
    INTO v_throughput
    FROM sensor_event_stream
    WHERE session_id = p_session_id;

    IF p_expected_events > 0 THEN
        v_loss_rate := ((p_expected_events - v_inserted)::DOUBLE PRECISION / p_expected_events::DOUBLE PRECISION) * 100.0;
    END IF;

    INSERT INTO system_metrics (
        session_id,
        latency_ms,
        throughput_req_per_sec,
        data_loss_rate_pct,
        note
    ) VALUES (
        p_session_id,
        v_latency_ms,
        v_throughput,
        v_loss_rate,
        'PDF target check: latency<100ms, throughput>1000req/s, loss<0.1%'
    )
    RETURNING metric_id INTO v_metric_id;

    RETURN v_metric_id;
END;
$$;

COMMIT;
