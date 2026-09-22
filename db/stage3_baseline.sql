BEGIN;

CREATE TABLE IF NOT EXISTS feature_baselines (
    baseline_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    learner_id BIGINT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL,
    window_seconds INT NOT NULL DEFAULT 60,
    n_rows INT NOT NULL,
    hr_mean_baseline DOUBLE PRECISION,
    hrv_sdnn_baseline DOUBLE PRECISION,
    acc_moving_avg_baseline DOUBLE PRECISION,
    hr_mean_std DOUBLE PRECISION,
    hrv_sdnn_std DOUBLE PRECISION,
    acc_moving_avg_std DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_feature_baselines UNIQUE (learner_id, activity_type, window_seconds),
    CONSTRAINT chk_feature_baselines_rows CHECK (n_rows >= 0)
);

CREATE OR REPLACE FUNCTION refresh_feature_baselines(p_window_seconds INT DEFAULT 60)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows INT := 0;
BEGIN
    DELETE FROM feature_baselines WHERE window_seconds = p_window_seconds;

    INSERT INTO feature_baselines (
        learner_id,
        activity_type,
        window_seconds,
        n_rows,
        hr_mean_baseline,
        hrv_sdnn_baseline,
        acc_moving_avg_baseline,
        hr_mean_std,
        hrv_sdnn_std,
        acc_moving_avg_std
    )
    SELECT
        s.learner_id,
        s.activity_type,
        p_window_seconds,
        COUNT(*)::INT,
        AVG(rf.hr_mean),
        AVG(rf.hrv_sdnn),
        AVG(rf.acc_moving_avg),
        STDDEV_SAMP(rf.hr_mean),
        STDDEV_SAMP(rf.hrv_sdnn),
        STDDEV_SAMP(rf.acc_moving_avg)
    FROM realtime_features rf
    JOIN learning_sessions s ON s.session_id = rf.session_id
    WHERE rf.window_seconds = p_window_seconds
    GROUP BY s.learner_id, s.activity_type;

    GET DIAGNOSTICS v_rows = ROW_COUNT;
    RETURN v_rows;
END;
$$;

CREATE OR REPLACE VIEW v_model_input_features_norm_v1 AS
SELECT
    l.learner_id,
    s.session_id,
    s.activity_type,
    rf.window_end_ts,
    rf.window_seconds,
    rf.hr_mean,
    rf.hrv_sdnn,
    rf.acc_moving_avg,
    b.hr_mean_baseline,
    b.hrv_sdnn_baseline,
    b.acc_moving_avg_baseline,
    (rf.hr_mean - b.hr_mean_baseline) / NULLIF(b.hr_mean_std, 0.0) AS hr_mean_z,
    (rf.hrv_sdnn - b.hrv_sdnn_baseline) / NULLIF(b.hrv_sdnn_std, 0.0) AS hrv_sdnn_z,
    (rf.acc_moving_avg - b.acc_moving_avg_baseline) / NULLIF(b.acc_moving_avg_std, 0.0) AS acc_moving_avg_z,
    s.achievement,
    l.gender,
    l.age
FROM realtime_features rf
JOIN learning_sessions s ON s.session_id = rf.session_id
JOIN learners l ON l.learner_id = s.learner_id
LEFT JOIN feature_baselines b
    ON b.learner_id = s.learner_id
   AND b.activity_type = s.activity_type
   AND b.window_seconds = rf.window_seconds;

CREATE INDEX IF NOT EXISTS idx_feature_baselines_learner_activity
    ON feature_baselines (learner_id, activity_type, window_seconds);

COMMIT;

