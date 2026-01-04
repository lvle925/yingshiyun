# services/session_manager.py

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
from config import REDIS_URL

logger = logging.getLogger(__name__)

user_data_redis_client: Optional[Union[AsyncRedis, RedisCluster]] = None
dev_cache: Dict[str, Dict] = {} # 内存缓存


async def initialize_session_manager():
    """初始化Redis客户端（自动检测单实例/集群）"""
    global user_data_redis_client
    if REDIS_URL:
        try:
            logger.info(f"[智慧卡-Session] 正在解析 REDIS_URL")
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
                logger.info(f"[智慧卡-Session] 检测到单实例Redis: {host_dicts[0]}")
                user_data_redis_client = AsyncRedis(
                    host=host_dicts[0]['host'],
                    port=host_dicts[0]['port'],
                    password=parsed_url.password,
                    encoding="utf-8",
                    decode_responses=True,
                    db=int(parsed_url.path.lstrip('/')) if parsed_url.path else 0
                )
            else:
                logger.info(f"[智慧卡-Session] 检测到Redis集群: {host_dicts}")
                Host = namedtuple("Host", ["host", "port"])
                startup_nodes = [Host(d['host'], d['port']) for d in host_dicts]
                user_data_redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=parsed_url.password,
                    encoding="utf-8",
                    decode_responses=True
                )
            
            await user_data_redis_client.ping()
            logger.info("✅ [智慧卡-Session] Redis连接成功")

        except Exception as e:
            logger.error(f"[智慧卡-Session] Redis连接失败: {e}，使用内存模式")
            user_data_redis_client = None
    else:
        logger.warning("[智慧卡-Session] 未配置REDIS_URL，使用内存模式")




async def close_session_manager():
    """关闭Redis连接。"""
    if user_data_redis_client:
        await user_data_redis_client.close()
        logger.info("会话管理器Redis连接已关闭。")

class AsyncClusterRedisChatMessageHistory(BaseChatMessageHistory):
    """一个兼容异步和集群的 Redis 聊天历史记录管理器。"""
    def __init__(self, session_id: str, client: Union[AsyncRedis, RedisCluster]):
        self.client = client
        self.session_id = session_id
        self.key = f"message_store:{self.session_id}"

    @property
    async def messages(self) -> List[BaseMessage]:
        """从 Redis 异步检索消息。"""
        try:
            _items = await self.client.lrange(self.key, 0, -1)
            items = [json.loads(m) for m in _items]
            messages = messages_from_dict(items)
            return messages
        except Exception as e:
            logger.error(f"从 Redis (key={self.key}) 检索聊天历史时出错: {e}")
            return []

    async def add_messages(self, messages: List[BaseMessage]) -> None:
        """向 Redis 异步添加多条消息。"""
        try:
            await self.client.rpush(self.key, *[json.dumps(m) for m in messages_to_dict(messages)])
        except Exception as e:
            logger.error(f"向 Redis (key={self.key}) 添加聊天历史时出错: {e}")

    async def clear(self) -> None:
        """从 Redis 清除会话内存。"""
        try:
            await self.client.delete(self.key)
        except Exception as e:
            logger.error(f"从 Redis (key={self.key}) 清除聊天历史时出错: {e}")



async def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取会话历史。如果 Redis 可用，返回我们自定义的异步集群兼容版本。
    否则，返回一个功能对等的内存版本。
    """
    if user_data_redis_client:
        return AsyncClusterRedisChatMessageHistory(session_id=session_id, client=user_data_redis_client)
    
    # 为内存模式也创建一个兼容的类，确保接口一致
    if session_id not in dev_cache:
        dev_cache[session_id] = {"history": []}
        
    class InMemoryHistory(BaseChatMessageHistory):
        def __init__(self, store: List[BaseMessage]): self._store = store
        @property
        async def messages(self) -> List[BaseMessage]: return self._store
        async def add_messages(self, messages: List[BaseMessage]) -> None: self._store.extend(messages)
        async def clear(self) -> None: self._store.clear()
            
    return InMemoryHistory(dev_cache[session_id]["history"])




async def get_user_analysis_data(session_id: str) -> dict:
    """
    异步获取用户分析数据。从 Redis 或内存缓存中读取。
    【V4 - 融合版】: 通用反序列化 + 对 last_relevant_palaces 的特殊健壮性处理。
    """
    raw_data_from_source = {}

    if user_data_redis_client:
        try:
            # 从 Redis 获取的是一个 {key: string_value} 的字典
            raw_data_from_source = await user_data_redis_client.hgetall(f"user_analysis:{session_id}")
            if not raw_data_from_source:
                return {}

        except RedisConnectionError as e:
            logger.error(f"从 Redis 获取数据失败: {e}。回退到内存缓存。")
            raw_data_from_source = dev_cache.get(session_id, {})
        except Exception as e:
            logger.error(f"获取用户分析数据时发生未知错误: {e}", exc_info=True)
            raw_data_from_source = dev_cache.get(session_id, {})
    else:
        # 如果不使用Redis，直接从内存缓存获取
        raw_data_from_source = dev_cache.get(session_id, {})

    if not raw_data_from_source:
        return {}

    # ======================================================================
    # --- 【核心修改 1】: 通用反序列化循环 ---
    # --- 遍历所有键值对，对所有看起来像JSON的值进行反序列化 ---
    # ======================================================================
    data_to_return = {}
    for key, value in raw_data_from_source.items():
        if isinstance(value, str):
            try:
                # 尝试将值作为JSON来解析
                # 如果值是 "{\"year\": 1990}"，它会变成 {'year': 1990}
                # 如果值是 "[\"命宫\"]"，它会变成 ['命宫']
                data_to_return[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # 如果解析失败，说明它就是一个普通的字符串（如 "未知"），直接赋值
                data_to_return[key] = value
        else:
            # 如果不是字符串，直接赋值
            data_to_return[key] = value

    # ======================================================================
    # --- 【核心修改 2】: 保留并应用您已有的对 palaces 的特殊处理逻辑 ---
    # --- 这个逻辑现在作为一个额外的、更强的“安全网”来工作 ---
    # ======================================================================
    if 'last_relevant_palaces' in data_to_return:
        palaces = data_to_return['last_relevant_palaces']

        # 即使通用反序列化失败了，或者存入的就是一个错误的字符串，这里也能处理
        if isinstance(palaces, str):
            logger.warning(f"检测到 'last_relevant_palaces' 是字符串类型 ('{palaces}')，将尝试修正。")
            try:
                parsed_palaces = json.loads(palaces)
                if isinstance(parsed_palaces, list):
                    data_to_return['last_relevant_palaces'] = parsed_palaces
                else:
                    logger.error(f"字段 'last_relevant_palaces' 的内容 '{palaces}' 解析为JSON后不是列表，已重置。")
                    data_to_return['last_relevant_palaces'] = []
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"无法将 '{palaces}'作为JSON解析，尝试按逗号分割。")
                data_to_return['last_relevant_palaces'] = [p.strip() for p in palaces.split(',') if p.strip()]

        elif not isinstance(palaces, list):
            logger.error(f"字段 'last_relevant_palaces' 的类型是 {type(palaces)} 而不是预期的 list 或 str，已重置。")
            data_to_return['last_relevant_palaces'] = []

    return data_to_return
    #return dev_cache.get(session_id, {})

async def store_user_analysis_data(session_id: str, data: dict):
    """
    异步存储用户分析数据到 Redis 或内存缓存。
    """
    # 过滤掉值为None的键，避免存储错误
    serializable_data = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in
                         data.items() if v is not None}
    if not serializable_data:
        return

    if user_data_redis_client:
        try:
            await user_data_redis_client.hmset(f"user_analysis:{session_id}", serializable_data)
        except RedisConnectionError as e:
            logger.error(f"存储数据到 Redis 失败: {e}。回退到内存缓存。")
            if session_id not in dev_cache:
                dev_cache[session_id] = {}
            dev_cache[session_id].update(serializable_data)
        except Exception as e:
            logger.error(f"存储用户分析数据时发生未知错误: {e}", exc_info=True)
            if session_id not in dev_cache:
                dev_cache[session_id] = {}
            dev_cache[session_id].update(serializable_data)
    else:
        if session_id not in dev_cache:
            dev_cache[session_id] = {}
        dev_cache[session_id].update(serializable_data)  # 使用update而不是覆盖