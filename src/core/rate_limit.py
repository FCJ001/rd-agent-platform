# ============================================================
# LLM 端点限流（Redis 固定窗口，按用户 + 端点桶）
#
# 为什么需要：认证只能证明"是合法用户"，挡不住合法用户刷
# chat / triage 这类一次消耗一次 LLM 调用的端点（成本 DoS）。
# SSE 流式端点还会长时间占住连接，更需要配额。
#
# 设计：
#   - 固定窗口计数（INCR + 首次 EXPIRE），实现最简单、Redis 往返 2 次；
#     窗口边界的双倍突发对 LLM 端点无所谓 —— 目标是挡滥用不是精确计量
#   - key 按 user_id 隔离：多用户共享窗口会互相挤兑
#   - ★ Redis 故障时 fail-open：限流是成本保护，不能反过来
#     把 Redis 抖动放大成全站 429；可用性靠 /health 的 critical 探测兜底
# ============================================================

import time

from fastapi import Depends
from loguru import logger

from src.core.deps import UserContext, get_current_user
from src.core.exceptions import ERR_RATE_LIMITED, BizException
from src.infra.redis_cache import _redis_client


async def enforce_rate_limit(
    client, scope: str, user_id: str, limit: int, window_seconds: int,
) -> None:
    """限流核心逻辑（独立出来便于单测，不绑 FastAPI）。

    超限抛 BizException(ERR_RATE_LIMITED) → 全局处理器映射 HTTP 429。
    """
    window = int(time.time()) // window_seconds
    key = f"ratelimit:{scope}:{user_id}:{window}"
    try:
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, window_seconds)
    except Exception as e:
        logger.warning(f"[RATE-LIMIT] Redis 不可用，fail-open scope={scope} user={user_id}: {e}")
        return
    if count > limit:
        raise BizException("请求过于频繁，请稍后再试", ERR_RATE_LIMITED)


def rate_limit(scope: str, limit: int, window_seconds: int = 60):
    """FastAPI 依赖工厂：rate_limit("chat", limit=30) → Depends 用。

    get_current_user 在同请求内被 FastAPI 依赖缓存去重，不会查两次库。
    """
    async def _dep(user: UserContext = Depends(get_current_user)) -> None:
        await enforce_rate_limit(_redis_client, scope, user.user_id, limit, window_seconds)

    return _dep
