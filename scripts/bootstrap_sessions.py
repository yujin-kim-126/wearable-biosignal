from __future__ import annotations

from pathlib import Path
import sys

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_db.config import load_config


def main() -> None:
    cfg = load_config()
    data_root = cfg.dataset_root / "Data"
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    with psycopg.connect(cfg.database_url) as conn:
        with conn.cursor() as cur:
            for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
                subject_code = subject_dir.name
                cur.execute(
                    """
                    INSERT INTO learners (gender, age)
                    VALUES (%s, %s)
                    RETURNING learner_id
                    """,
                    ("unknown", 20),
                )
                learner_id = cur.fetchone()[0]

                for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                    session_name = session_dir.name
                    cur.execute(
                        """
                        INSERT INTO learning_sessions (learner_id, activity_type)
                        VALUES (%s, %s)
                        """,
                        (learner_id, session_name),
                    )
                print(f"Inserted subject {subject_code} with learner_id={learner_id}")
        conn.commit()


if __name__ == "__main__":
    main()
