import os
from typing import Optional
import aiomysql

DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "db": os.getenv("DB_NAME"),
    "autocommit": True,
}

_db_pool: Optional[aiomysql.Pool] = None


async def get_db_pool() -> Optional[aiomysql.Pool]:
    return _db_pool


async def init_db_pool():
    global _db_pool
    if _db_pool:
        return _db_pool
    if not DB_CONFIG.get("host"):
        return None
    _db_pool = await aiomysql.create_pool(**DB_CONFIG)
    return _db_pool


async def close_db_pool():
    global _db_pool
    if _db_pool:
        _db_pool.close()
        await _db_pool.wait_closed()
        _db_pool = None
