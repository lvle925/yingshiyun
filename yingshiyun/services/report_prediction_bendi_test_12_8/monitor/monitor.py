import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone, timedelta as td
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_logs(days_to_keep: int = 3) -> None:
    """Delete log files older than the given number of days."""
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    for log_file in LOG_DIR.glob("*.log*"):
        try:
            modified_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if modified_time < cutoff:
                log_file.unlink(missing_ok=True)
        except OSError:
            # Ignore permissions or concurrent deletion issues
            continue


class UTC8Formatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, timezone(td(hours=8)))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def _configure_monitor_logger() -> logging.Logger:
    logger = logging.getLogger("api_monitor")
    if logger.handlers:
        return logger

    handler = logging.FileHandler(LOG_DIR / "api_monitor.log", encoding="utf-8")
    formatter = UTC8Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


cleanup_old_logs()
monitor_logger = _configure_monitor_logger()


def generate_request_id() -> str:
    """Return a short uuid string to correlate logs."""
    return uuid.uuid4().hex[:8]


def _format_extra(extra_data: Optional[Dict[str, Any]]) -> str:
    if not extra_data:
        return ""
    try:
        return json.dumps(extra_data, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(extra_data)


def log_step(
    step_name: str,
    request_id: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    status: str = "成功",
) -> None:
    rid = request_id or generate_request_id()
    payload = {"request_id": rid, "step": step_name, "status": status}
    if extra_data:
        payload.update(extra_data)
    monitor_logger.info(_format_extra(payload))


class StepMonitor:
    """Context manager for logging step duration and status."""

    def __init__(
        self,
        step_name: str,
        request_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        self.step_name = step_name
        self.request_id = request_id or generate_request_id()
        self.extra_data = extra_data or {}
        self._start_time: Optional[float] = None

    def __enter__(self) -> "StepMonitor":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed_ms = None
        if self._start_time is not None:
            elapsed_ms = round((time.perf_counter() - self._start_time) * 1000, 2)
        status = "失败" if exc_type else "成功"
        data = dict(self.extra_data)
        if elapsed_ms is not None:
            data["elapsed_ms"] = elapsed_ms
        log_step(
            step_name=self.step_name,
            request_id=self.request_id,
            extra_data=data,
            status=status,
        )
        return False

