from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    dataset_root: Path
    database_url: str


def load_config() -> AppConfig:
    dataset_root = Path(
        os.getenv(
            "DATASET_ROOT",
            "a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0",
        )
    )
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set in the environment")
    return AppConfig(dataset_root=dataset_root, database_url=database_url)
