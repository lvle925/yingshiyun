# app/monitor.py
"""
日志监控核心模块
提供细粒度的步骤监控和日志记录功能
"""
import logging
import uuid
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
# 确保无论工作目录为何，日志都写进容器 /app/logs
try:
    LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    # 监控日志文件路径
    MONITOR_LOG_FILE = LOG_DIR / "api_monitor.log"
except Exception as e:
    # 如果创建日志目录失败，使用临时目录
    import tempfile
    LOG_DIR = Path(tempfile.gettempdir()) / "app_logs"
    LOG_DIR.mkdir(exist_ok=True)
    MONITOR_LOG_FILE = LOG_DIR / "api_monitor.log"


def cleanup_old_logs(days_to_keep: int = 3):
    """
    启动时清理旧日志（只保留近 N 天），避免替换已打开的文件。
    
    Args:
        days_to_keep: 保留最近几天的日志文件，默认3天
    """
    if not LOG_DIR.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    for log_file in LOG_DIR.glob("*.log"):
        try:
            # 获取文件的修改时间
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff_date:
                log_file.unlink()
                print(f"已删除旧日志文件: {log_file}")
        except Exception as e:
            print(f"清理日志文件 {log_file} 时出错: {e}")


# 配置独立的监控日志记录器
monitor_logger = logging.getLogger("monitor")
monitor_logger.setLevel(logging.INFO)
monitor_logger.propagate = False  # 不传播到根日志记录器

# 如果已经有处理器，先清除（避免重复添加）
if monitor_logger.handlers:
    monitor_logger.handlers.clear()

# 创建文件处理器（使用 try-except 确保即使失败也能导入模块）
try:
    file_handler = logging.FileHandler(MONITOR_LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式：时间 | 级别 | 信息
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    monitor_logger.addHandler(file_handler)
    
    # 启动时清理旧日志
    cleanup_old_logs(days_to_keep=3)
except Exception as e:
    # 如果文件处理器创建失败，使用控制台处理器作为后备
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    monitor_logger.addHandler(console_handler)


class StepMonitor:
    """
    步骤监控上下文管理器
    
    作为上下文管理器使用，进入时记录开始时间，退出时自动记录成功/失败日志并附带耗时、额外信息。
    
    示例:
        with StepMonitor("调用紫薇API", request_id="abc123", extra_data={"api_url": "..."}):
            result = call_api()
    """
    
    def __init__(self, step_name: str, request_id: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None):
        """
        初始化步骤监控器
        
        Args:
            step_name: 步骤名称
            request_id: 请求ID，如果不提供则自动生成时间戳ID
            extra_data: 额外的上下文数据，会在日志中记录
        """
        self.step_name = step_name
        self.request_id = request_id or self._generate_request_id()
        self.extra_data = extra_data or {}
        self.start_time = None
        self.exception = None
    
    def _generate_request_id(self) -> str:
        """生成8位uuid作为request_id"""
        return str(uuid.uuid4())[:8]
    
    def __enter__(self):
        """进入上下文时记录开始时间"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时记录日志"""
        if self.start_time is None:
            return
        
        elapsed_time = time.time() - self.start_time
        
        # 判断状态
        if exc_type is None:
            status = "成功"
        else:
            status = "失败"
            self.exception = str(exc_val) if exc_val else str(exc_type)
        
        # 构建日志消息
        message_parts = [
            f"步骤: {self.step_name}",
            f"状态: {status}",
            f"耗时: {elapsed_time:.3f}秒",
            f"request_id: {self.request_id}"
        ]
        
        # 添加额外数据
        if self.extra_data:
            extra_str = ", ".join([f"{k}={v}" for k, v in self.extra_data.items()])
            message_parts.append(f"额外信息: {extra_str}")
        
        # 添加异常信息（如果有）
        if self.exception:
            message_parts.append(f"异常: {self.exception}")
        
        log_message = " | ".join(message_parts)
        
        # 根据状态选择日志级别
        if status == "成功":
            monitor_logger.info(log_message)
        else:
            monitor_logger.error(log_message)
        
        # 不抑制异常，让异常正常传播
        return False


def log_step(step_name: str, request_id: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None, status: str = "成功"):
    """
    手动记录一次监控日志
    
    用于在流程收尾或特殊节点手动写一次监控日志。
    
    Args:
        step_name: 步骤名称
        request_id: 请求ID，如果不提供则自动生成时间戳ID
        extra_data: 额外的上下文数据
        status: 状态，默认为"成功"
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    
    message_parts = [
        f"步骤: {step_name}",
        f"状态: {status}",
        f"request_id: {request_id}"
    ]
    
    if extra_data:
        extra_str = ", ".join([f"{k}={v}" for k, v in extra_data.items()])
        message_parts.append(f"额外信息: {extra_str}")
    
    log_message = " | ".join(message_parts)
    
    if status == "成功":
        monitor_logger.info(log_message)
    else:
        monitor_logger.error(log_message)

