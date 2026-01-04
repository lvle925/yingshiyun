import aiomysql
from typing import Optional

_db_pool: Optional[aiomysql.Pool] = None


async def init_db_pool(**kwargs) -> Optional[aiomysql.Pool]:
    global _db_pool
    if _db_pool:
        return _db_pool
    _db_pool = await aiomysql.create_pool(**kwargs)
    return _db_pool


async def close_db_pool():
    global _db_pool
    if _db_pool:
        _db_pool.close()
        await _db_pool.wait_closed()
        _db_pool = None
