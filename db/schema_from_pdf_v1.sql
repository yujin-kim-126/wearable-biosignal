-- Schema v1 based only on:
-- "맞춤형 학습 지도를 위한 웨어러블 생체 신호 기반 실시간 통합 데이터베이스 설계_김유진,서지훈.pdf"

BEGIN;

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

CREATE TABLE IF NOT EXISTS biometric_logs (
    log_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    ts TIMESTAMP NOT NULL,
    eda DOUBLE PRECISION,
    hr DOUBLE PRECISION,
    acc_x DOUBLE PRECISION,
    acc_y DOUBLE PRECISION,
    acc_z DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_biometric_logs_session_ts UNIQUE (session_id, ts),
    CONSTRAINT chk_biometric_logs_nonnegative CHECK (
        (eda IS NULL OR eda >= 0.0) AND
        (hr IS NULL OR hr >= 0.0)
    )
);

-- Real-time window function outputs (paper: 60-second HR/HRV features)
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

CREATE INDEX IF NOT EXISTS idx_learning_sessions_learner_id
    ON learning_sessions (learner_id);

CREATE INDEX IF NOT EXISTS idx_biometric_logs_session_ts
    ON biometric_logs (session_id, ts);

CREATE INDEX IF NOT EXISTS idx_realtime_features_session_window_end
    ON realtime_features (session_id, window_end_ts);

COMMIT;
