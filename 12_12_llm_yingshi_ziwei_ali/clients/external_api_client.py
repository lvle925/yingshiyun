# clients/external_api_client.py

import asyncio
import random
import logging
import aiohttp
from typing import Optional, Dict, Any

#from .shared_client import async_aiohttp_client
from config import VLLM_REQUEST_TIMEOUT_SECONDS, MAX_API_CALL_RETRIES
from clients import shared_client

logger = logging.getLogger(__name__)

# 全局客户端实例，在主应用启动时初始化
#async_aiohttp_client: Optional[aiohttp.ClientSession] = None


def initialize_external_api_client():
    """在应用启动时调用，创建并配置全局aiohttp客户端。"""
    global async_aiohttp_client
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
            connect=VLLM_REQUEST_TIMEOUT_SECONDS,
            sock_read=None
        )
    )
    logger.info("外部API客户端 (aiohttp.ClientSession) 初始化完成。")


async def close_external_api_client():
    """在应用关闭时调用，清理客户端。"""
    if shared_client.async_aiohttp_client:
        await shared_client.async_aiohttp_client.close()
        logger.info("外部API客户端已关闭。")


async def robust_api_call_with_retry(
        #session: aiohttp.ClientSession, # <--- 新增参数
        url: str,
        payload: dict,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 16.0,
        timeout: int = 30
        # ...
) -> dict:
    if not shared_client.async_aiohttp_client: # 检查传入的 session
        raise RuntimeError("AIOHTTP session 未提供。")

    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.warning(f"API调用失败，将在 {delay:.2f} 秒后进行第 {attempt} 次重试...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay) + random.uniform(0, delay * 0.5)

            async with shared_client.async_aiohttp_client.post(url, json=payload,
                                                 timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientConnectionError, aiohttp.ClientOSError, asyncio.TimeoutError,
                aiohttp.ServerDisconnectedError) as e:
            logger.warning(f"API调用尝试 #{attempt + 1} 遇到可重试的错误: {type(e).__name__}: {e}")
            last_exception = e
            continue
        except aiohttp.ClientResponseError as e:
            if e.status >= 500:
                logger.warning(f"API调用尝试 #{attempt + 1} 遇到可重试的服务器错误: {e.status} {e.message}")
                last_exception = e
                continue
            else:
                logger.error(f"API调用遇到不可重试的客户端错误: {e.status} {e.message}")
                raise e

    logger.error(f"API调用在 {max_retries} 次重试后彻底失败。")
    raise last_exception