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
    sql_files = [
        ROOT / "db" / "schema_from_pdf_v2.sql",
        ROOT / "db" / "pipeline_from_pdf_v2.sql",
    ]
    with psycopg.connect(cfg.database_url) as conn:
        with conn.cursor() as cur:
            for sql_path in sql_files:
                sql_text = sql_path.read_text(encoding="utf-8-sig")
                cur.execute(sql_text)
                print(f"Applied: {sql_path.name}")
        conn.commit()


if __name__ == "__main__":
    main()
