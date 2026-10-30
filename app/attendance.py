from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Union

from app.config import BASE


AttendanceLogPath = Union[str, Path, None]
ATTENDANCE_FIELDS = ["time", "user", "method", "attempts", "security_level"]


def attendance_log_path(log_path: AttendanceLogPath = None) -> Path:
    if log_path is not None:
        return Path(log_path)
    return BASE / "logs" / "attendance_log.csv"


def _format_time_value(value) -> str:
    if value is None:
        return "0.00 s"
    if isinstance(value, (int, float)):
        return f"{float(value):.2f} s"
    return str(value)


def _add_security_level_column(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or "security_level" in fieldnames:
            return
        rows = list(reader)

    migrated_fieldnames = [*fieldnames, "security_level"]
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=migrated_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def append_attendance_record(
    user: str,
    method: str = "face",
    time_value: float | str | None = None,
    attempts: int | None = None,
    log_path: AttendanceLogPath = None,
    security_level: str | None = None
) -> Path:
    path = attendance_log_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0
    fieldnames = ATTENDANCE_FIELDS
    if file_exists:
        _add_security_level_column(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or ATTENDANCE_FIELDS

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "time": _format_time_value(time_value),
            "user": user,
            "method": method,
            "attempts": attempts if attempts is not None else 0,
            "security_level": security_level if security_level is not None else ""
        })
    return path


def build_attendance_message(
    user: str,
    method: str = "face",
    time_value: float | str | None = None,
    attempts: int | None = None,
    security_level: str | None = None
) -> str:
    timestamp = "now"  # just a textual label, not a timestamp used in the attendance log
    attempts_part = f" | próby: {attempts}" if attempts is not None else ""
    security_level_part = f" | poziom bezpieczeństwa: {security_level}" if security_level is not None else ""
    if time_value is not None:
        return f"{timestamp} | Zalogowano się po {method} metodą: {user} | czas: {_format_time_value(time_value)}{attempts_part}{security_level_part}"
    return f"{timestamp} | Zalogowano się po {method} metodą: {user}{attempts_part}{security_level_part}"
