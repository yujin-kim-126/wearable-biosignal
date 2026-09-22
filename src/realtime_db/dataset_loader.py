from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


SENSOR_FILES = {
    "ACC": "ACC.csv",
    "BVP": "BVP.csv",
    "EDA": "EDA.csv",
    "HR": "HR.csv",
    "IBI": "IBI.csv",
    "TEMP": "TEMP.csv",
    "TAG": "tags.csv",
}


@dataclass(frozen=True)
class SensorEvent:
    sensor_type: str
    source_ts: datetime
    sample_rate_hz: float | None
    value_1: float | None
    value_2: float | None
    value_3: float | None
    emitted_seq: int
    payload_json: dict[str, Any]


def _unix_to_dt(unix_ts: float) -> datetime:
    return datetime.fromtimestamp(unix_ts, tz=UTC).replace(tzinfo=None)


def _read_text_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _first_float(cell_text: str) -> float:
    # Some files store repeated values in comma-separated form (e.g., ACC header rows).
    return float(cell_text.split(",")[0].strip())


def _load_regular_sensor(path: Path, sensor_type: str, row_limit: int | None = None) -> list[SensorEvent]:
    lines = _read_text_lines(path)
    if len(lines) < 3:
        return []

    start_ts = _unix_to_dt(_first_float(lines[0]))
    sample_rate_hz = _first_float(lines[1])
    frame = pd.read_csv(path, skiprows=2, header=None, nrows=row_limit)
    events: list[SensorEvent] = []

    for idx, row in frame.iterrows():
        source_ts = start_ts + timedelta(seconds=idx / sample_rate_hz)
        if sensor_type == "ACC":
            x = float(row.iloc[0])
            y = float(row.iloc[1])
            z = float(row.iloc[2])
            payload = {"acc_x": x, "acc_y": y, "acc_z": z}
            events.append(
                SensorEvent(
                    sensor_type=sensor_type,
                    source_ts=source_ts,
                    sample_rate_hz=sample_rate_hz,
                    value_1=x,
                    value_2=y,
                    value_3=z,
                    emitted_seq=idx,
                    payload_json=payload,
                )
            )
            continue

        v = float(row.iloc[0])
        events.append(
            SensorEvent(
                sensor_type=sensor_type,
                source_ts=source_ts,
                sample_rate_hz=sample_rate_hz,
                value_1=v,
                value_2=None,
                value_3=None,
                emitted_seq=idx,
                payload_json={"value": v},
            )
        )
    return events


def _load_ibi(path: Path, row_limit: int | None = None) -> list[SensorEvent]:
    lines = _read_text_lines(path)
    if not lines:
        return []

    start_ts = _unix_to_dt(_first_float(lines[0]))
    frame = pd.read_csv(path, skiprows=1, header=None, nrows=row_limit)
    events: list[SensorEvent] = []
    for idx, row in frame.iterrows():
        offset = float(row.iloc[0])
        ibi_value = float(row.iloc[1])
        source_ts = start_ts + timedelta(seconds=offset)
        events.append(
            SensorEvent(
                sensor_type="IBI",
                source_ts=source_ts,
                sample_rate_hz=None,
                value_1=ibi_value,
                value_2=None,
                value_3=None,
                emitted_seq=idx,
                payload_json={"offset_sec": offset, "ibi_sec": ibi_value},
            )
        )
    return events


def _load_tags(path: Path, row_limit: int | None = None) -> list[SensorEvent]:
    lines = _read_text_lines(path)
    events: list[SensorEvent] = []
    if row_limit is not None:
        lines = lines[:row_limit]
    for idx, line in enumerate(lines):
        source_ts = _unix_to_dt(float(line))
        events.append(
            SensorEvent(
                sensor_type="TAG",
                source_ts=source_ts,
                sample_rate_hz=None,
                value_1=None,
                value_2=None,
                value_3=None,
                emitted_seq=idx,
                payload_json={"tag_press": True},
            )
        )
    return events


def load_session_events(
    dataset_root: Path,
    subject_code: str,
    session_name: str,
    max_events: int | None = None,
) -> list[SensorEvent]:
    session_path = dataset_root / "Data" / subject_code / session_name
    if not session_path.exists():
        raise FileNotFoundError(f"Session path not found: {session_path}")

    all_events: list[SensorEvent] = []
    per_sensor_limit = max_events if max_events is not None else None
    for sensor_type, file_name in SENSOR_FILES.items():
        file_path = session_path / file_name
        if not file_path.exists():
            continue

        if sensor_type == "IBI":
            events = _load_ibi(file_path, row_limit=per_sensor_limit)
        elif sensor_type == "TAG":
            events = _load_tags(file_path, row_limit=per_sensor_limit)
        else:
            events = _load_regular_sensor(file_path, sensor_type=sensor_type, row_limit=per_sensor_limit)
        all_events.extend(events)

    all_events.sort(key=lambda ev: ev.source_ts)
    if max_events is not None:
        return all_events[:max_events]
    return all_events
