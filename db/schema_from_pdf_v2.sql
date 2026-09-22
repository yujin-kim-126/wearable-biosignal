-- Schema v2 based only on:
-- "Design of a Real-time Integrated Database Based on Wearable Bio-signals for Personalized Learning Guidance"
-- Goal: cover simulation layer + preprocessing layer + integration layer + realtime features + reliability metrics.

BEGIN;

-- 1) Core entities from the paper
CREATE TABLE IF NOT EXISTS learners (
    learner_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    age INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_learners_age CHECK (age > 0 AND age < 120),
    CONSTRAINT chk_learners_gender CHECK (gender IN ('male', 'female', 'unknown'))
);

CREATE TABLE IF NOT EXISTS learning_sessions (
    session_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    learner_id BIGINT NOT NULL REFERENCES learners(learner_id) ON DELETE RESTRICT,
    activity_type VARCHAR(50) NOT NULL,
    achievement DOUBLE PRECISION,
    session_start TIMESTAMP,
    session_end TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_learning_sessions_achievement CHECK (
        achievement IS NULL OR (achievement >= 0.0 AND achievement <= 100.0)
    ),
    CONSTRAINT chk_learning_sessions_time CHECK (
        session_end IS NULL OR session_start IS NULL OR session_end >= session_start
    )
);

-- 2) Simulation layer (dynamic event stream from wearable-like emitters)
CREATE TABLE IF NOT EXISTS sensor_event_stream (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    sensor_type VARCHAR(20) NOT NULL,
    source_ts TIMESTAMP NOT NULL,
    emitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingest_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sample_rate_hz DOUBLE PRECISION,
    value_1 DOUBLE PRECISION,
    value_2 DOUBLE PRECISION,
    value_3 DOUBLE PRECISION,
    emitted_seq BIGINT,
    payload_json JSONB,
    CONSTRAINT chk_sensor_type CHECK (
        sensor_type IN ('ACC', 'BVP', 'EDA', 'HR', 'IBI', 'TEMP', 'TAG')
    ),
    CONSTRAINT chk_sample_rate_hz CHECK (sample_rate_hz IS NULL OR sample_rate_hz > 0)
);

-- 3) Preprocessing layer (32Hz synchronized stream)
CREATE TABLE IF NOT EXISTS preprocessed_stream_32hz (
    pp_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    ts_32hz TIMESTAMP NOT NULL,
    sync_method VARCHAR(20) NOT NULL,
    is_interpolated BOOLEAN NOT NULL DEFAULT FALSE,
    source_event_count INT NOT NULL DEFAULT 0,
    eda DOUBLE PRECISION,
    hr DOUBLE PRECISION,
    acc_x DOUBLE PRECISION,
    acc_y DOUBLE PRECISION,
    acc_z DOUBLE PRECISION,
    ibi_seconds DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_preprocessed_stream_32hz UNIQUE (session_id, ts_32hz),
    CONSTRAINT chk_sync_method CHECK (sync_method IN ('zoh', 'linear')),
    CONSTRAINT chk_source_event_count CHECK (source_event_count >= 0),
    CONSTRAINT chk_preprocessed_nonnegative CHECK (
        (eda IS NULL OR eda >= 0.0) AND
        (hr IS NULL OR hr >= 0.0) AND
        (ibi_seconds IS NULL OR ibi_seconds >= 0.0)
    )
);

-- 4) Integration layer (final integrated biometric logs)
CREATE TABLE IF NOT EXISTS biometric_logs (
    log_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    ts TIMESTAMP NOT NULL,
    eda DOUBLE PRECISION,
    hr DOUBLE PRECISION,
    acc_x DOUBLE PRECISION,
    acc_y DOUBLE PRECISION,
    acc_z DOUBLE PRECISION,
    ibi_seconds DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_biometric_logs_session_ts UNIQUE (session_id, ts),
    CONSTRAINT chk_biometric_logs_nonnegative CHECK (
        (eda IS NULL OR eda >= 0.0) AND
        (hr IS NULL OR hr >= 0.0) AND
        (ibi_seconds IS NULL OR ibi_seconds >= 0.0)
    )
);

-- 5) Realtime feature table (paper: 60s rolling HR / HRV-SDNN + acceleration smoothing)
CREATE TABLE IF NOT EXISTS realtime_features (
    feature_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    window_end_ts TIMESTAMP NOT NULL,
    window_seconds INT NOT NULL DEFAULT 60,
    hr_mean DOUBLE PRECISION,
    hrv_sdnn DOUBLE PRECISION,
    acc_moving_avg DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_realtime_features_window UNIQUE (session_id, window_end_ts, window_seconds),
    CONSTRAINT chk_realtime_features_window CHECK (window_seconds > 0)
);

-- 6) Reliability/performance metrics table (paper targets)
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_id BIGINT REFERENCES learning_sessions(session_id) ON DELETE SET NULL,
    latency_ms DOUBLE PRECISION,
    throughput_req_per_sec DOUBLE PRECISION,
    data_loss_rate_pct DOUBLE PRECISION,
    note VARCHAR(255),
    CONSTRAINT chk_system_metrics_nonnegative CHECK (
        (latency_ms IS NULL OR latency_ms >= 0.0) AND
        (throughput_req_per_sec IS NULL OR throughput_req_per_sec >= 0.0) AND
        (data_loss_rate_pct IS NULL OR (data_loss_rate_pct >= 0.0 AND data_loss_rate_pct <= 100.0))
    )
);

-- 7) Window-function view for 60-second realtime calculation
CREATE OR REPLACE VIEW v_realtime_feature_60s AS
SELECT
    b.session_id,
    b.ts AS window_end_ts,
    AVG(b.hr) OVER (
        PARTITION BY b.session_id
        ORDER BY b.ts
        RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW
    ) AS hr_mean_60s,
    STDDEV_SAMP(b.ibi_seconds) OVER (
        PARTITION BY b.session_id
        ORDER BY b.ts
        RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW
    ) AS hrv_sdnn_60s,
    AVG(SQRT(COALESCE(b.acc_x, 0)^2 + COALESCE(b.acc_y, 0)^2 + COALESCE(b.acc_z, 0)^2)) OVER (
        PARTITION BY b.session_id
        ORDER BY b.ts
        RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW
    ) AS acc_moving_avg_60s
FROM biometric_logs b;

-- 8) Target-check view for performance validation
CREATE OR REPLACE VIEW v_system_metric_targets AS
SELECT
    metric_id,
    measured_at,
    session_id,
    latency_ms,
    throughput_req_per_sec,
    data_loss_rate_pct,
    (latency_ms IS NOT NULL AND latency_ms < 100.0) AS pass_latency_target,
    (throughput_req_per_sec IS NOT NULL AND throughput_req_per_sec > 1000.0) AS pass_throughput_target,
    (data_loss_rate_pct IS NOT NULL AND data_loss_rate_pct < 0.1) AS pass_loss_target
FROM system_metrics;

CREATE INDEX IF NOT EXISTS idx_learning_sessions_learner_id
    ON learning_sessions (learner_id);

CREATE INDEX IF NOT EXISTS idx_sensor_event_stream_session_ts
    ON sensor_event_stream (session_id, source_ts);

CREATE INDEX IF NOT EXISTS idx_sensor_event_stream_sensor_ts
    ON sensor_event_stream (sensor_type, source_ts);

CREATE INDEX IF NOT EXISTS idx_preprocessed_stream_session_ts
    ON preprocessed_stream_32hz (session_id, ts_32hz);

CREATE INDEX IF NOT EXISTS idx_biometric_logs_session_ts
    ON biometric_logs (session_id, ts);

CREATE INDEX IF NOT EXISTS idx_realtime_features_session_window_end
    ON realtime_features (session_id, window_end_ts);

CREATE INDEX IF NOT EXISTS idx_system_metrics_measured_at
    ON system_metrics (measured_at);

COMMIT;
