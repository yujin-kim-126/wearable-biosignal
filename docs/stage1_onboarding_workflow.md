# Stage1 Onboarding Workflow

## Goal
- Register a new learner.
- Generate a 1-week collection plan.
- Execute planned collection sessions and track progress.

## DB Objects
- SQL file: `db/stage1_onboarding.sql`
- Core tables:
  - `learner_profiles`
  - `collection_batches`
  - `collection_batch_sessions`
- Progress views:
  - `v_collection_batch_progress`
  - `v_personalization_ready_batches`

## Scripts
- Plan creation:
  - `python scripts/stage1_plan_weekly_collection.py --learner-code L2026_STAGE1_001 --days 7 --sessions-per-day 3 --template-subject S1 --template-sessions "Final,Midterm 1,Midterm 2" --apply-sql`
- Plan execution:
  - `python scripts/stage1_run_weekly_collection.py --batch-id 1 --speed 1000 --max-events 5000 --sync-method zoh`

## Typical Runtime Flow
1. Apply onboarding SQL (`--apply-sql`) once.
2. Create 1-week batch plan for one learner.
3. Run planned sessions in order.
4. Check progress from `v_collection_batch_progress`.
5. When all sessions complete and days condition is met, batch is marked ready for personal training.
