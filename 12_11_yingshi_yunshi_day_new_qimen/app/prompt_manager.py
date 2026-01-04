import os
import yaml
import threading
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class PromptReloadHandler(FileSystemEventHandler):
    """文件变化处理器"""
    
    def __init__(self, prompt_manager):
        self.prompt_manager = prompt_manager
        self.last_modified = 0
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.yaml') or event.src_path.endswith('.yml'):
            # 避免重复触发（某些编辑器会触发多次）
            import time
            current_time = time.time()
            if current_time - self.last_modified < 1.0:
                return
            self.last_modified = current_time
            
            logger.info(f"检测到提示词文件变化: {event.src_path}")
            try:
                self.prompt_manager.reload_prompts()
                logger.info("提示词热更新成功")
            except Exception as e:
                logger.error(f"提示词热更新失败: {e}", exc_info=True)


class PromptManager:
    """提示词管理器，支持热更新"""
    
    def __init__(self, prompts_file_path: str = None):
        """
        初始化提示词管理器
        
        Args:
            prompts_file_path: 提示词文件路径，默认为 prompts/prompts.yaml
        """
        if prompts_file_path is None:
            # 默认路径：app目录下的 prompts.yaml
            app_dir = Path(__file__).parent
            prompts_file_path = app_dir / "prompts.yaml"
        
        self.prompts_file_path = Path(prompts_file_path)
        self._lock = threading.RLock()  # 可重入锁，确保线程安全
        self._prompts = {}
        self._observer = None
        
        # 确保文件存在
        if not self.prompts_file_path.exists():
            logger.warning(f"提示词文件不存在: {self.prompts_file_path}，将使用默认值")
            self._create_default_prompts_file()
        
        # 初始加载
        self.reload_prompts()
        
        # 启动文件监控（异步，不阻塞）
        try:
            self._start_file_watcher()
        except Exception as e:
            logger.warning(f"启动文件监控失败，将使用手动重新加载: {e}")
    
    def _create_default_prompts_file(self):
        """创建默认的提示词文件"""
        self.prompts_file_path.parent.mkdir(parents=True, exist_ok=True)
        # 这里可以写入默认内容，或者从代码中提取
        logger.info(f"已创建提示词文件目录: {self.prompts_file_path.parent}")
    
    def reload_prompts(self):
        """重新加载提示词文件"""
        with self._lock:
            try:
                if not self.prompts_file_path.exists():
                    logger.error(f"提示词文件不存在: {self.prompts_file_path}")
                    return
                
                with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                    prompts_data = yaml.safe_load(f)
                
                if prompts_data is None:
                    logger.error("提示词文件为空或格式错误")
                    return
                
                # 更新提示词
                self._prompts = prompts_data
                logger.info("提示词已重新加载")
                
            except yaml.YAMLError as e:
                logger.error(f"YAML解析错误: {e}")
            except Exception as e:
                logger.error(f"加载提示词文件失败: {e}", exc_info=True)
    
    def _start_file_watcher(self):
        """启动文件监控"""
        try:
            self._observer = Observer()
            event_handler = PromptReloadHandler(self)
            # 监控文件所在目录
            watch_dir = str(self.prompts_file_path.parent)
            self._observer.schedule(event_handler, watch_dir, recursive=False)
            self._observer.start()
            logger.info(f"已启动提示词文件监控: {watch_dir}")
        except Exception as e:
            logger.warning(f"启动文件监控失败: {e}，将使用手动重新加载")
    
    def get(self, key: str, default: str = None) -> str:
        """
        获取提示词
        
        Args:
            key: 提示词键名
            default: 默认值（如果不存在）
        
        Returns:
            提示词内容
        """
        with self._lock:
            return self._prompts.get(key, default)
    
    def format_prompt(self, key: str, **kwargs) -> str:
        """
        获取并格式化提示词（支持模板变量）
        
        Args:
            key: 提示词键名
            **kwargs: 模板变量
        
        Returns:
            格式化后的提示词
        """
        template = self.get(key)
        if template is None:
            logger.warning(f"提示词键 '{key}' 不存在")
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"提示词模板变量缺失: {e}")
            return template
    
    def stop(self):
        """停止文件监控"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("提示词文件监控已停止")


# 全局单例
_prompt_manager_instance = None
_prompt_manager_lock = threading.Lock()


def get_prompt_manager(prompts_file_path: str = None) -> PromptManager:
    """获取全局提示词管理器单例"""
    global _prompt_manager_instance
    
    if _prompt_manager_instance is None:
        with _prompt_manager_lock:
            if _prompt_manager_instance is None:
                _prompt_manager_instance = PromptManager(prompts_file_path)
    
    return _prompt_manager_instance