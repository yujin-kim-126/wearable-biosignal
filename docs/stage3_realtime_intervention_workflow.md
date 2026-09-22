# Stage3 Realtime Intervention Workflow

## Goal
- Use the personalized model after onboarding week.
- Analyze the latest 60s feature window in real time.
- Emit immediate intervention alerts (break/pacing/attention reminder) with cooldown control.

## Components
- SQL: `db/stage3_realtime_intervention.sql`
  - `intervention_inference_log`
  - `intervention_alerts`
  - `v_recent_intervention_status`
- Engine: `src/realtime_db/intervention.py`
- Runner: `scripts/stage3_run_realtime_intervention.py`
- API endpoint: `POST /intervention/analyze`

## One-shot execution example
```bash
python scripts/stage3_run_realtime_intervention.py --session-id 31 --apply-sql
```

## Repeated polling example
```bash
python scripts/stage3_run_realtime_intervention.py --session-id 31 --iterations 10 --sleep-seconds 2
```

## Output
- CSV: `outputs/stage3_realtime_intervention_session<session_id>.csv`
- DB logs:
  - all inference decisions in `intervention_inference_log`
  - actual alerts in `intervention_alerts`
