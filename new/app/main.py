import logging
import os
import aiomysql
from fastapi import FastAPI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入我们的模块
from app.routes import router as api_router
from app import lenormand_service

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'db': os.getenv('DB_NAME'),
    'autocommit': True
}

app = FastAPI(
    title="综合运势服务平台",
    description="包含紫微斗数年运评分与雷诺曼每日占卜服务。"
)

# 注册路由
app.include_router(api_router)

# --- 生命周期管理 ---
@app.on_event("startup")
async def startup_event():
    logger.info("系统启动中...")
    
    # 1. 初始化数据库连接池
    try:
        if DB_CONFIG['host']:
            app.state.db_pool = await aiomysql.create_pool(**DB_CONFIG)
            logger.info("数据库连接池已就绪。")
        else:
            logger.warning("未配置 DB HOST，跳过数据库连接。")
            app.state.db_pool = None
    except Exception as e:
        logger.critical(f"数据库连接失败: {e}")
        app.state.db_pool = None

    # 2. 初始化雷诺曼服务资源 (CSV, HTTP Client)
    await lenormand_service.init_resources()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("系统关闭中...")
    
    # 1. 关闭雷诺曼资源
    await lenormand_service.close_resources()
    
    # 2. 关闭数据库连接池
    if getattr(app.state, 'db_pool', None):
        app.state.db_pool.close()
        await app.state.db_pool.wait_closed()
        logger.info("数据库连接池已关闭。")

if __name__ == "__main__":
    import uvicorn
    # 允许热重载
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)