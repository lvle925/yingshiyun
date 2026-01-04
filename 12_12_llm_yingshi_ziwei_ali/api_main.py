# api_main.py

import uvicorn
import logging
import importlib
import sys
from pathlib import Path
from fastapi import FastAPI, Depends, Request, HTTPException
from starlette.responses import StreamingResponse, JSONResponse
from prometheus_client import make_asgi_app

# 模块化导入
from config import APP_SECRET  # 检查配置
from models import SignableAPIRequest
from security.verifier import signature_verifier
from monitoring import REQUESTS_RECEIVED
from database import  db_manager
from clients import external_api_client, vllm_client
from services import session_manager, chat_processor
from clients import shared_client
from fastapi.middleware.cors import CORSMiddleware
from services.monitor import StepMonitor, log_step, generate_request_id

# 【新增】热更新相关导入
import asyncio
import os
# --- 日志配置 ---
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


# --- 【新增】Prompts 目录热更新功能（轮询模式，兼容 Docker 挂载）---
class PromptsFileMonitor:
    """
    通过轮询文件修改时间来监控 prompts/ 目录下的 XML 文件变化
    
    这种方式比 watchdog 更可靠，特别是在 Docker 容器中使用文件挂载时。
    因为很多编辑器保存文件时会删除原文件再创建新文件，导致 inotify 事件丢失。
    """
    
    def __init__(self, prompts_dir: str = "prompts", check_interval: float = 2.0):
        self.prompts_dir = Path(prompts_dir)
        self.check_interval = check_interval
        self.file_mtimes = {}  # 存储每个提示词文件的修改时间（包括 .xml 和 .j2）
        self.running = False
        self.task = None
    
    async def start(self):
        """启动文件监控"""
        if not self.prompts_dir.exists():
            logger.warning(f"⚠️  {self.prompts_dir} 目录不存在，热更新功能未启用")
            return
        
        # 初始化所有提示词文件的修改时间（包括 .xml 和 .j2）
        for xml_file in self.prompts_dir.glob("*.xml"):
            self.file_mtimes[str(xml_file)] = os.path.getmtime(xml_file)
        for j2_file in self.prompts_dir.glob("*.j2"):
            self.file_mtimes[str(j2_file)] = os.path.getmtime(j2_file)
        
        self.running = True
        logger.info(f"✅ prompts/ 目录监控已启动（监控 {len(self.file_mtimes)} 个提示词文件，轮询间隔: {self.check_interval}秒）")
        
        self.task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """停止文件监控"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("✅ prompts/ 目录监控已停止")

    
    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                await asyncio.sleep(self.check_interval)
                
                if not self.prompts_dir.exists():
                    continue
                
                # 检查所有提示词文件是否有变化（包括 .xml 和 .j2）
                changed_files = []
                current_files = {}
                
                # 监控 XML 文件
                for xml_file in self.prompts_dir.glob("*.xml"):
                    file_path = str(xml_file)
                    current_mtime = os.path.getmtime(xml_file)
                    current_files[file_path] = current_mtime
                    
                    # 检查是否是新文件或被修改
                    if file_path not in self.file_mtimes or self.file_mtimes[file_path] != current_mtime:
                        changed_files.append(xml_file.name)
                
                # 监控 Jinja2 模板文件
                for j2_file in self.prompts_dir.glob("*.j2"):
                    file_path = str(j2_file)
                    current_mtime = os.path.getmtime(j2_file)
                    current_files[file_path] = current_mtime
                    
                    # 检查是否是新文件或被修改
                    if file_path not in self.file_mtimes or self.file_mtimes[file_path] != current_mtime:
                        changed_files.append(j2_file.name)
                
                # 如果有文件变化，执行热更新
                if changed_files:
                    logger.info(f"🔄 检测到 {len(changed_files)} 个提示词文件变化: {', '.join(changed_files)}")
                    self.file_mtimes = current_files
                    self._reload_prompts()
            
            except Exception as e:
                logger.error(f"❌ 监控文件时出错: {e}", exc_info=True)
    
    def _reload_prompts(self):
        """重新加载提示词模板及相关模块"""
        try:
            # 如果 prompt_logic 已经被导入，先重新加载提示词文件
            if "prompt_logic" not in sys.modules:
                logger.warning("⚠️  prompt_logic 模块尚未被导入，跳过重新加载")
                return
            
            # 1. 重新加载提示词文件到内存缓存（热更新时使用锁）
            import prompt_logic
            prompt_logic.reload_all_prompts(use_lock=True)
            
            # 2. 重新初始化模块级变量
            prompt_logic._init_prompt_templates()
            logger.info("✅ prompt_logic 模块级变量已重新初始化")
            
            # 3. 重新加载 prompt_logic 模块（确保其他模块能获取到最新值）
            importlib.reload(sys.modules["prompt_logic"])
            logger.info("✅ prompt_logic 模块已重新加载")
            
            # 4. 重新加载所有导入了 prompt_logic 的模块（保持正确的依赖顺序）
            modules_to_reload = [
                "clients.vllm_client",
                "services.chat_processor"
            ]
            
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    logger.info(f"✅ {module_name} 已重新加载")
            
            logger.info("🎉 提示词模板及相关模块热更新完成")
            
        except Exception as e:
            logger.error(f"❌ 重新加载提示词模板失败: {e}", exc_info=True)


# --- FastAPI 应用 ---
app = FastAPI(
    title="紫微斗数AI API (模块化高性能版)",
    description="一个使用aiohttp进行底层HTTP请求，实现高性能、高并发、模块化的AI接口。",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 或明确写上 "http://192.168.1.101:5500"
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 Prometheus 指标路由
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# 全局异常处理器 - 记录非200状态码
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """处理 HTTPException，记录非200状态码到监控日志"""
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    
    # 只记录非200状态码
    if exc.status_code != 200:
        log_step(
            "错误：HTTP异常",
            request_id=request_id,
            status="失败",
            extra_data={
                "reason": "HTTP异常",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": str(request.url.path),
                "method": request.method
            }
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理所有未捕获的异常"""
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    
    log_step(
        "错误：未捕获的异常",
        request_id=request_id,
        status="失败",
        extra_data={
            "reason": "未捕获的异常",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "status_code": 500,
            "path": str(request.url.path),
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误"}
    )


@app.on_event("startup")
async def startup_event():
    """应用启动时，初始化所有必要的模块。"""
    logger.info("应用启动，开始初始化所有服务...")
    if not APP_SECRET:
        logger.error("致命错误: 必须在 config.py 中设置 APP_SECRET。")
        # 在实际生产中，您可能希望这里直接退出程序

    shared_client.initialize_shared_client()

    await db_manager.init_db_pool()
    await session_manager.initialize_session_manager()
    
    # 【新增】初始化提示词模板（在启动时加载，而不是在模块导入时）
    try:
        import prompt_logic
        # 先预加载提示词文件到缓存（不使用锁，避免阻塞）
        prompt_logic.reload_all_prompts(use_lock=False)
        # 然后初始化模块级变量
        prompt_logic._init_prompt_templates()
        logger.info("✅ 提示词模板初始化完成")
    except Exception as e:
        logger.error(f"❌ 提示词模板初始化失败: {e}", exc_info=True)
        # 不阻止应用启动，即使提示词加载失败
    
    # 【新增】启动 prompts/ 目录监控（提示词热更新功能，使用轮询模式）
    prompts_monitor = PromptsFileMonitor(check_interval=2.0)
    await prompts_monitor.start()
    app.state.prompts_monitor = prompts_monitor
    
    logger.info("所有服务初始化完毕。")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时，清理所有资源。"""
    logger.info("应用关闭，开始清理所有服务...")
    
    # 【新增】停止 prompts/ 目录监控
    if hasattr(app.state, "prompts_monitor"):
        await app.state.prompts_monitor.stop()
    
    await shared_client.close_shared_client()

    await db_manager.close_db_pool()
    await session_manager.close_session_manager()
    logger.info("所有服务清理完毕。")


@app.post("/chat_yingshis_V12_25", summary="发送聊天消息 (流式 & 签名验证)")
async def chat(request: Request, validated_body: dict = Depends(signature_verifier)):

    """
    处理用户聊天请求的核心入口。
    接收经过签名验证的请求体，并将其传递给聊天处理器。
    """
    REQUESTS_RECEIVED.inc()
    request_id = generate_request_id()
    
    # 将 request_id 存储到 request.state，供异常处理器使用
    request.state.request_id = request_id

    with StepMonitor(
        "成功获取请求",
        request_id=request_id,
        extra_data={"endpoint": "/chat_yingshis_V10_23"},
    ):
        api_request = SignableAPIRequest.model_validate(validated_body)

    # 将所有业务逻辑委托给 chat_processor
    # process_chat_request 是一个异步生成器，可以直接用于 StreamingResponse
    return StreamingResponse(
        chat_processor.process_chat_request(
            api_request,
            monitor_request_id=request_id,
            #http_client=request.app.state.aiohttp_client
        ),
        media_type="text/plain; charset=utf-8"
    )


if __name__ == "__main__":
    print("\n--- 启动说明---")
    print("uvicorn api_main:app --host 0.0.0.0 --port 8044 --reload")

    uvicorn.run("api_main:app", host="0.0.0.0", port=8044, reload=True)