import json
import logging
import time
import uuid
from contextlib import ContextDecorator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "api_monitor.log"


def cleanup_old_logs(days_to_keep: int = 3) -> None:
    """删除超过 days_to_keep 天的旧日志文件，只保留近 3 天。"""
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    for file_path in LOG_DIR.glob("*.log"):
        try:
            if file_path.is_file() and datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                file_path.unlink()
        except OSError:
            # 避免删除失败影响主流程
            continue


cleanup_old_logs()


monitor_logger = logging.getLogger("intent_monitor")
if not monitor_logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | req=%(request_id)s | step=%(step_name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    monitor_logger.addHandler(handler)
    monitor_logger.setLevel(logging.INFO)
    monitor_logger.propagate = False


def _serialize_extra(extra_data: Optional[Dict[str, Any]]) -> str:
    if not extra_data:
        return ""
    try:
        return json.dumps(extra_data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(extra_data)


def _logger_extra(request_id: str, step_name: str) -> Dict[str, Any]:
    return {"request_id": request_id, "step_name": step_name}


class StepMonitor(ContextDecorator):
    """上下文监控器，自动记录步骤耗时与结果。"""

    def __init__(
        self,
        step_name: str,
        request_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        self.step_name = step_name
        self.request_id = request_id or uuid.uuid4().hex[:8]
        self.extra_data = extra_data or {}
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        monitor_logger.info(
            f"{self.step_name} 开始",
            extra=_logger_extra(self.request_id, self.step_name),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = 0.0 if self.start_time is None else time.perf_counter() - self.start_time
        status = "失败" if exc_type else "成功"
        message = f"{self.step_name} {status} | 耗时={duration:.3f}s"
        serialized_extra = _serialize_extra({**self.extra_data, "duration": f"{duration:.3f}s"})
        if serialized_extra:
            message = f"{message} | {serialized_extra}"

        if exc_type:
            monitor_logger.exception(
                message,
                extra=_logger_extra(self.request_id, self.step_name),
            )
        else:
            monitor_logger.info(
                message,
                extra=_logger_extra(self.request_id, self.step_name),
            )
        # 不吞异常
        return False


def log_step(
    step_name: str,
    request_id: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    status: str = "成功",
) -> None:
    """在需要时单独记录某个步骤的状态。"""
    rid = request_id or uuid.uuid4().hex[:8]
    serialized_extra = _serialize_extra(extra_data)
    message = f"{step_name} {status}"
    if serialized_extra:
        message = f"{message} | {serialized_extra}"

    monitor_logger.info(
        message,
        extra=_logger_extra(rid, step_name),
    )

