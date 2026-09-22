from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_db.config import load_config
from realtime_db.db import DbClient
from realtime_db.simulator import run_session_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="e.g., S1")
    parser.add_argument("--session", required=True, help="e.g., Final")
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--speed", type=float, default=30.0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--sync-method", choices=["zoh", "linear"], default="zoh")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    cfg = load_config()
    db_client = DbClient(cfg.database_url)

    result = await run_session_simulation(
        dataset_root=cfg.dataset_root,
        subject_code=args.subject,
        session_name=args.session,
        session_id=args.session_id,
        db_client=db_client,
        speed_multiplier=args.speed,
        max_events=args.max_events,
        dry_run=args.dry_run,
        sync_method=args.sync_method,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(_main())

