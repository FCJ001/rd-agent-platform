"""Redis 双连接池：业务池（decode_responses=True）+ Checkpointer 池（bytes 模式）。"""

import redis.asyncio as redis

from src.core.config import get_settings

settings = get_settings()

# ── 业务连接池：存 triage_state JSON ──
redis_pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=True,
    encoding="utf-8",
)

_redis_client = redis.Redis(connection_pool=redis_pool)


async def get_redis_client() -> redis.Redis:
    """FastAPI Depends 注入用。返回模块级别 Redis 客户端。"""
    return _redis_client


# ── Checkpointer 连接池：AsyncRedisSaver 要求 bytes 模式 ──
_checkpointer_pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=False,
)

_checkpointer_client = redis.Redis(connection_pool=_checkpointer_pool)


def get_checkpointer_redis() -> redis.Redis:
    """返回供 AsyncRedisSaver（LangGraph checkpointer）专用的 Redis 客户端。"""
    return _checkpointer_client
