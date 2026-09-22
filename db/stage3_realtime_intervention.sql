BEGIN;

CREATE TABLE IF NOT EXISTS intervention_inference_log (
    inference_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    learner_id BIGINT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    window_end_ts TIMESTAMP NOT NULL,
    model_id BIGINT REFERENCES personal_models(model_id) ON DELETE SET NULL,
    model_version VARCHAR(50),
    predicted_label VARCHAR(100) NOT NULL,
    predicted_confidence DOUBLE PRECISION NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    recommended_action VARCHAR(50) NOT NULL,
    should_alert BOOLEAN NOT NULL DEFAULT FALSE,
    alert_suppressed BOOLEAN NOT NULL DEFAULT FALSE,
    reason_text VARCHAR(255),
    feature_snapshot JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_intervention_inference_window UNIQUE (session_id, window_end_ts),
    CONSTRAINT chk_intervention_inference_conf CHECK (
        predicted_confidence >= 0.0 AND predicted_confidence <= 1.0
    ),
    CONSTRAINT chk_intervention_inference_risk CHECK (
        risk_score >= 0.0 AND risk_score <= 1.0
    ),
    CONSTRAINT chk_intervention_inference_action CHECK (
        recommended_action IN ('short_break', 'task_pacing', 'attention_reminder', 'continue_learning')
    )
);

CREATE TABLE IF NOT EXISTS intervention_alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    learner_id BIGINT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    inference_id BIGINT REFERENCES intervention_inference_log(inference_id) ON DELETE SET NULL,
    action_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message_text VARCHAR(255) NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    cooldown_until TIMESTAMP,
    acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_intervention_alerts_action CHECK (
        action_type IN ('short_break', 'task_pacing', 'attention_reminder')
    ),
    CONSTRAINT chk_intervention_alerts_severity CHECK (
        severity IN ('low', 'medium', 'high')
    ),
    CONSTRAINT chk_intervention_alerts_risk CHECK (
        risk_score >= 0.0 AND risk_score <= 1.0
    )
);

CREATE OR REPLACE VIEW v_recent_intervention_status AS
SELECT
    i.session_id,
    i.learner_id,
    i.window_end_ts,
    i.model_version,
    i.predicted_label,
    i.predicted_confidence,
    i.risk_score,
    i.recommended_action,
    i.should_alert,
    i.alert_suppressed,
    i.reason_text,
    i.created_at AS inferred_at
FROM intervention_inference_log i
ORDER BY i.created_at DESC;

CREATE INDEX IF NOT EXISTS idx_intervention_inference_session_ts
    ON intervention_inference_log (session_id, window_end_ts DESC);

CREATE INDEX IF NOT EXISTS idx_intervention_alerts_session_ts
    ON intervention_alerts (session_id, created_at DESC);

COMMIT;
