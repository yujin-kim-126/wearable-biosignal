BEGIN;

CREATE TABLE IF NOT EXISTS learner_profiles (
    learner_id BIGINT PRIMARY KEY REFERENCES learners(learner_id) ON DELETE CASCADE,
    learner_code VARCHAR(100) NOT NULL UNIQUE,
    timezone VARCHAR(64) NOT NULL DEFAULT 'Asia/Seoul',
    onboarding_status VARCHAR(20) NOT NULL DEFAULT 'new',
    onboarding_started_at TIMESTAMP,
    onboarding_completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_learner_profiles_status CHECK (
        onboarding_status IN ('new', 'collecting', 'ready_for_training', 'active')
    )
);

CREATE TABLE IF NOT EXISTS collection_batches (
    batch_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    learner_id BIGINT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    planned_start_date DATE NOT NULL,
    planned_end_date DATE NOT NULL,
    days_planned INT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'planned',
    notes VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_collection_batches_days CHECK (days_planned > 0),
    CONSTRAINT chk_collection_batches_dates CHECK (planned_end_date >= planned_start_date),
    CONSTRAINT chk_collection_batches_status CHECK (
        status IN ('planned', 'collecting', 'completed', 'failed', 'cancelled')
    )
);

CREATE TABLE IF NOT EXISTS collection_batch_sessions (
    batch_session_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id BIGINT NOT NULL REFERENCES collection_batches(batch_id) ON DELETE CASCADE,
    session_id BIGINT NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    collection_date DATE NOT NULL,
    session_order SMALLINT NOT NULL,
    template_subject_code VARCHAR(20) NOT NULL,
    template_session_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'planned',
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    event_count INT,
    error_message VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_collection_batch_day_order UNIQUE (batch_id, collection_date, session_order),
    CONSTRAINT chk_collection_batch_sessions_order CHECK (session_order > 0),
    CONSTRAINT chk_collection_batch_sessions_status CHECK (
        status IN ('planned', 'collecting', 'completed', 'failed', 'skipped')
    ),
    CONSTRAINT chk_collection_batch_sessions_event_count CHECK (
        event_count IS NULL OR event_count >= 0
    )
);

CREATE OR REPLACE VIEW v_collection_batch_progress AS
SELECT
    cb.batch_id,
    cb.learner_id,
    lp.learner_code,
    cb.status AS batch_status,
    cb.planned_start_date,
    cb.planned_end_date,
    cb.days_planned,
    COUNT(cbs.batch_session_id) AS total_sessions,
    COUNT(*) FILTER (WHERE cbs.status = 'completed') AS completed_sessions,
    COUNT(*) FILTER (WHERE cbs.status = 'failed') AS failed_sessions,
    COUNT(*) FILTER (WHERE cbs.status = 'collecting') AS collecting_sessions,
    COUNT(DISTINCT cbs.collection_date) FILTER (WHERE cbs.status = 'completed') AS completed_days,
    COALESCE(SUM(cbs.event_count), 0) AS total_events_collected
FROM collection_batches cb
JOIN learner_profiles lp ON lp.learner_id = cb.learner_id
LEFT JOIN collection_batch_sessions cbs ON cbs.batch_id = cb.batch_id
GROUP BY
    cb.batch_id, cb.learner_id, lp.learner_code, cb.status,
    cb.planned_start_date, cb.planned_end_date, cb.days_planned;

CREATE OR REPLACE VIEW v_personalization_ready_batches AS
SELECT
    p.*,
    (
        p.batch_status = 'completed'
        AND p.failed_sessions = 0
        AND p.completed_sessions = p.total_sessions
        AND p.completed_days >= p.days_planned
    ) AS ready_for_personal_training
FROM v_collection_batch_progress p;

CREATE INDEX IF NOT EXISTS idx_learner_profiles_code
    ON learner_profiles (learner_code);

CREATE INDEX IF NOT EXISTS idx_collection_batches_learner_status
    ON collection_batches (learner_id, status);

CREATE INDEX IF NOT EXISTS idx_collection_batch_sessions_batch_status
    ON collection_batch_sessions (batch_id, status);

CREATE INDEX IF NOT EXISTS idx_collection_batch_sessions_session
    ON collection_batch_sessions (session_id);

COMMIT;
