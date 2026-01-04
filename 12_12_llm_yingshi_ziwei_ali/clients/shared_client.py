# clients/shared_client.py

import logging
import aiohttp
from typing import Optional

from config import VLLM_REQUEST_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# 在这里定义和持有唯一的、共享的 aiohttp.ClientSession 实例
async_aiohttp_client: Optional[aiohttp.ClientSession] = None

def initialize_shared_client():
    """在应用启动时调用，创建并配置全局共享的 aiohttp 客户端。"""
    global async_aiohttp_client
    if async_aiohttp_client is None or async_aiohttp_client.closed:
        logger.info("正在创建新的共享 aiohttp.ClientSession 实例...")
        connector = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=1000,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=120
        )
        async_aiohttp_client = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                connect=50,  # 连接超时30秒足够
                sock_read=None
            )
        )
        logger.info("共享的 aiohttp.ClientSession 初始化完成。")
    else:
        logger.warning("共享的 aiohttp.ClientSession 已存在且未关闭，跳过重复初始化。")

async def close_shared_client():
    """在应用关闭时调用，清理共享客户端。"""
    global async_aiohttp_client
    if async_aiohttp_client and not async_aiohttp_client.closed:
        await async_aiohttp_client.close()
        logger.info("共享的 aiohttp.ClientSession 已关闭。")
    async_aiohttp_client = None
