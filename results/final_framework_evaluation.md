# Final Framework Performance Evaluation

- evaluated_at: 2026-05-27 21:14:19
- batch_id: 1
- learner_code: L2026_STAGE1_001

## Completion Check
- step1_onboarding_week_collection: True
- step2_personal_model_training: True
- step3_realtime_intervention_active: True

## Learner/Batch Status
- learner_profile: (11, 'L2026_STAGE1_001', 'ready_for_training')
- batch_progress: (1, 11, 'completed', 21, 21, 0, 7, 7, 65000)
- personal_model: (1, 11, 1, 'v20260527_205613', 'trained', 'outputs/personal_model_batch1_v20260527_205613.pkl')
- personal_model_metrics: [('test', 1.0, 1.0, 1.0)]

## System Metrics (batch sessions)
- n_sessions: 21
- avg_latency_ms: 0.0006261333333333332
- avg_throughput_req_per_sec: 4874.430721288485
- avg_data_loss_rate_pct: 0.0
- pass_latency_target(<100ms): 21/21
- pass_throughput_target(>1000req/s): 21/21
- pass_loss_target(<0.1%): 21/21

## Intervention Runtime Benchmark
- iterations_requested: 120
- iterations_processed: 120
- avg_inference_ms: 117.19995833312471
- p95_inference_ms: 189.27850003819913
- total_elapsed_ms: 14064.345400081947
- should_alert_count: 120
- suppressed_count: 120
- inference_count_batch: 123
- alert_count_batch: 1
