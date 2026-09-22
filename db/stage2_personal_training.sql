BEGIN;

CREATE TABLE IF NOT EXISTS personal_models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    learner_id BIGINT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    batch_id BIGINT NOT NULL REFERENCES collection_batches(batch_id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL,
    target_name VARCHAR(100) NOT NULL,
    feature_set VARCHAR(100) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    train_rows INT NOT NULL,
    test_rows INT NOT NULL,
    train_sessions INT NOT NULL,
    test_sessions INT NOT NULL,
    artifact_path VARCHAR(255) NOT NULL,
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'trained',
    notes VARCHAR(255),
    CONSTRAINT uq_personal_models_version UNIQUE (learner_id, model_version),
    CONSTRAINT chk_personal_models_counts CHECK (
        train_rows >= 0 AND test_rows >= 0 AND train_sessions >= 0 AND test_sessions >= 0
    ),
    CONSTRAINT chk_personal_models_status CHECK (
        status IN ('trained', 'active', 'retired', 'failed')
    )
);

CREATE TABLE IF NOT EXISTS personal_model_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES personal_models(model_id) ON DELETE CASCADE,
    split_name VARCHAR(20) NOT NULL,
    accuracy DOUBLE PRECISION,
    f1_macro DOUBLE PRECISION,
    auc_ovr_macro DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_personal_model_metrics_split CHECK (split_name IN ('validation', 'test'))
);

CREATE OR REPLACE VIEW v_latest_active_personal_model AS
SELECT DISTINCT ON (pm.learner_id)
    pm.model_id,
    pm.learner_id,
    lp.learner_code,
    pm.batch_id,
    pm.model_version,
    pm.target_name,
    pm.feature_set,
    pm.algorithm,
    pm.artifact_path,
    pm.trained_at,
    pm.status
FROM personal_models pm
JOIN learner_profiles lp ON lp.learner_id = pm.learner_id
WHERE pm.status IN ('trained', 'active')
ORDER BY pm.learner_id, pm.trained_at DESC, pm.model_id DESC;

CREATE INDEX IF NOT EXISTS idx_personal_models_learner_batch
    ON personal_models (learner_id, batch_id, trained_at DESC);

CREATE INDEX IF NOT EXISTS idx_personal_model_metrics_model
    ON personal_model_metrics (model_id, split_name);

COMMIT;
