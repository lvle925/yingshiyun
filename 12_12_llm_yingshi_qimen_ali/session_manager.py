# session_manager.py - 奇门服务的会话管理

import json
import logging
from typing import Optional, Dict, Any, Union, List
from redis.asyncio.cluster import RedisCluster
from collections import namedtuple
from redis.asyncio import Redis as AsyncRedis, ConnectionError as RedisConnectionError
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from urllib.parse import urlparse
from config import REDIS_URL, MAX_HISTORY_MESSAGES, SESSION_TTL

logger = logging.getLogger(__name__)

user_data_redis_client: Optional[Union[AsyncRedis, RedisCluster]] = None
dev_cache: Dict[str, Dict] = {} # 内存缓存


async def initialize_session_manager():
    """初始化Redis客户端（自动检测单实例/集群）"""
    global user_data_redis_client
    if REDIS_URL:
        try:
            logger.info(f"[奇门-Session] 正在解析 REDIS_URL")
            parsed_url = urlparse(REDIS_URL)

            host_dicts = []
            # 修复：正确处理密码中的@符号
            netloc_without_auth = parsed_url.netloc
            if '@' in netloc_without_auth:
                netloc_without_auth = netloc_without_auth.rsplit('@', 1)[1]
            
            netlocs = netloc_without_auth.split(',')
            for loc in netlocs:
                if ':' in loc:
                    host, port_str = loc.split(':', 1)
                    if host and port_str.isdigit():
                        host_dicts.append({"host": host, "port": int(port_str)})
            
            if not host_dicts:
                raise ValueError("无法解析Redis地址")

            # 判断单实例还是集群
            if len(host_dicts) == 1:
                logger.info(f"[奇门-Session] 检测到单实例Redis: {host_dicts[0]}")
                user_data_redis_client = AsyncRedis(
                    host=host_dicts[0]['host'],
                    port=host_dicts[0]['port'],
                    password=parsed_url.password,
                    encoding="utf-8",
                    decode_responses=True,
                    db=int(parsed_url.path.lstrip('/')) if parsed_url.path else 0
                )
            else:
                logger.info(f"[奇门-Session] 检测到Redis集群: {host_dicts}")
                Host = namedtuple("Host", ["host", "port"])
                startup_nodes = [Host(d['host'], d['port']) for d in host_dicts]
                user_data_redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=parsed_url.password,
                    encoding="utf-8",
                    decode_responses=True
                )
            
            await user_data_redis_client.ping()
            logger.info("✅ [奇门-Session] Redis连接成功")

        except Exception as e:
            logger.error(f"[奇门-Session] Redis连接失败: {e}，使用内存模式")
            user_data_redis_client = None
    else:
        logger.warning("[奇门-Session] 未配置REDIS_URL，使用内存模式")


async def close_session_manager():
    """关闭Redis连接。"""
    if user_data_redis_client:
        await user_data_redis_client.close()
        logger.info("[奇门-Session] Redis连接已关闭")


class AsyncClusterRedisChatMessageHistory(BaseChatMessageHistory):
    """Redis聊天历史管理器"""
    def __init__(self, session_id: str, client: Union[AsyncRedis, RedisCluster]):
        self.client = client
        self.session_id = session_id
        self.key = f"message_store:{self.session_id}"

    @property
    async def messages(self) -> List[BaseMessage]:
        """获取历史消息"""
        try:
            _items = await self.client.lrange(self.key, -MAX_HISTORY_MESSAGES, -1)
            items = [json.loads(m) for m in _items]
            messages = messages_from_dict(items)
            return messages
        except Exception as e:
            logger.error(f"[奇门-Session] 获取历史失败: {e}")
            return []

    async def add_messages(self, messages: List[BaseMessage]) -> None:
        """添加消息"""
        try:
            await self.client.rpush(self.key, *[json.dumps(m) for m in messages_to_dict(messages)])
            await self.client.ltrim(self.key, -MAX_HISTORY_MESSAGES, -1)
            await self.client.expire(self.key, SESSION_TTL)
        except Exception as e:
            logger.error(f"[奇门-Session] 添加历史失败: {e}")

    async def clear(self) -> None:
        """清除历史"""
        try:
            await self.client.delete(self.key)
        except Exception as e:
            logger.error(f"[奇门-Session] 清除历史失败: {e}")


async def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取会话历史"""
    if user_data_redis_client:
        return AsyncClusterRedisChatMessageHistory(session_id=session_id, client=user_data_redis_client)
    
    if session_id not in dev_cache:
        dev_cache[session_id] = {"history": []}
        
    class InMemoryHistory(BaseChatMessageHistory):
        def __init__(self, store: List[BaseMessage]): self._store = store
        @property
        async def messages(self) -> List[BaseMessage]: return self._store
        async def add_messages(self, messages: List[BaseMessage]) -> None: self._store.extend(messages)
        async def clear(self) -> None: self._store.clear()
            
    return InMemoryHistory(dev_cache[session_id]["history"])

