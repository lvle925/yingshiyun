"""统一的流程监控日志模块。"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "api_monitor.log"


def cleanup_old_logs(days_to_keep: int = 3) -> None:
    """清理超过 `days_to_keep` 天的历史日志文件。"""
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    for log_file in LOG_DIR.glob("api_monitor.log*"):
        try:
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
                log_file.unlink(missing_ok=True)
        except OSError:
            # 忽略删除失败，避免影响主流程
            continue


cleanup_old_logs()


monitor_logger = logging.getLogger("api_monitor")
if not monitor_logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    monitor_logger.addHandler(handler)
monitor_logger.setLevel(logging.INFO)
monitor_logger.propagate = False


def generate_request_id() -> str:
    """生成 8 位 request_id，用于链路追踪。"""
    return uuid.uuid4().hex[:8]


def _serialize_extra(extra_data: Optional[Dict[str, Any]]) -> str:
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
    """在流程收尾或特殊节点手动写一次监控日志。"""
    rid = request_id or generate_request_id()
    monitor_logger.info(
        "req=%s | step=%s | status=%s | extra=%s",
        rid,
        step_name,
        status,
        _serialize_extra(extra_data),
    )


@dataclass
class StepMonitor:
    """用于包裹关键步骤的上下文管理器。"""

    step_name: str
    request_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    status: str = field(default="成功", init=False)
    _start: float = field(default=0.0, init=False)

    def __enter__(self) -> "StepMonitor":
        self._start = time.perf_counter()
        rid = self.request_id or generate_request_id()
        self.request_id = rid
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc_type:
            self.status = "失败"
            level = monitor_logger.error
            extra = {**self.extra_data, "error": str(exc_val), "duration_ms": duration_ms}
        else:
            level = monitor_logger.info
            extra = {**self.extra_data, "duration_ms": duration_ms}

        level(
            "req=%s | step=%s | status=%s | extra=%s",
            self.request_id,
            self.step_name,
            self.status,
            _serialize_extra(extra),
        )

    def update_extra(self, **kwargs: Any) -> None:
        """动态更新附加信息。"""
        self.extra_data.update(kwargs)

    def update_status(self, status: str) -> None:
        """修改最终状态文案。"""
        self.status = status


