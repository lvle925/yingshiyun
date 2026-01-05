import json
import logging
import uuid
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "api_monitor.log"


def cleanup_old_logs(days_to_keep: int = 3) -> None:
    """Remove log files older than the retention window."""
    if not LOG_DIR.exists():
        return

    cutoff = datetime.now() - timedelta(days=days_to_keep)
    for file_path in LOG_DIR.glob("api_monitor.log*"):
        try:
            if not file_path.is_file():
                continue
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff:
                file_path.unlink(missing_ok=True)
        except Exception:
            # 清理异常不应影响主流程，忽略即可
            continue


cleanup_old_logs()


monitor_logger = logging.getLogger("services.monitor")
if not monitor_logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    monitor_logger.addHandler(handler)
    monitor_logger.setLevel(logging.INFO)
    monitor_logger.propagate = False


def _format_extra(extra: Optional[Dict[str, Any]]) -> str:
    if not extra:
        return ""
    try:
        return json.dumps(extra, ensure_ascii=False)
    except TypeError:
        # 极端场景下无法序列化，转为字符串
        safe_extra = {}
        for key, value in extra.items():
            try:
                json.dumps(value)
                safe_extra[key] = value
            except TypeError:
                safe_extra[key] = str(value)
        return json.dumps(safe_extra, ensure_ascii=False)


def generate_request_id() -> str:
    """生成 8 位请求 ID，便于跨模块追踪。"""
    return uuid.uuid4().hex[:8]


class StepMonitor(AbstractAsyncContextManager):
    """业务步骤监控上下文管理器。"""

    def __init__(
        self,
        step_name: str,
        request_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        self.step_name = step_name
        self.request_id = request_id or generate_request_id()
        self.extra_data = extra_data or {}
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = datetime.now().timestamp()
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "失败" if exc else "成功"
        duration = 0.0
        if self._start is not None:
            duration = max(datetime.now().timestamp() - self._start, 0)
        extra = dict(self.extra_data)
        extra["duration_sec"] = round(duration, 3)
        monitor_logger.info(
            f"{self.step_name} | {status} | request_id={self.request_id} | { _format_extra(extra) }"
        )
        # 不吞异常
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)


def log_step(
    step_name: str,
    request_id: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    status: str = "成功",
) -> None:
    """在特殊节点手动写入监控日志。"""
    rid = request_id or generate_request_id()
    monitor_logger.info(
        f"{step_name} | {status} | request_id={rid} | {_format_extra(extra_data)}"
    )

