# ============================================================
# 分诊会话锁：Redis 分布式锁 + 进程内 asyncio 锁，两层叠加
#
# 为什么两层都要（少一层都有洞）：
#   - 只用进程内 asyncio.Lock → 多 worker 部署时每个进程一把锁，
#     Dockerfile 默认 UVICORN_WORKERS=2，等于没锁；
#   - 只用 Redis 锁 → 进程内两条并发请求变成「一个跑、一个报忙」，
#     比排队串行更差（同一用户连点两次不该看到错误）。
#
# 为什么是「抢不到就快速失败」而不是阻塞排队：
#   interrupt() 是抛异常，锁在暂停时已释放，真实持有时长 = 两次
#   interrupt 之间的 LLM 调用（秒~十几秒）。抢不到锁意味着另一个请求
#   正在跑同一会话 —— 让用户等一个不确定的时长，不如短重试后告知
#   「上一次还在进行中」。长阻塞还会占住 FastAPI worker，高并发下放大问题。
#
# ★ 释放用 Lua CAS（比对 token 再删），不用 redis-py 的 Lock 类：
#   自实现逻辑全在自己代码里，注入假 redis 即可单测（符合本项目
#   「纯函数单测进 CI」的习惯）；且避免 redis-py Lock 的
#   thread_local=True 默认值在协程里取错 token 导致锁泄漏。
# ============================================================

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from src.core.config import get_settings
from src.core.exceptions import ConversationBusyError
from src.core.logger import logger

settings = get_settings()

LOCK_PREFIX = "triage_lock"

# 释放锁：比对 token 再删，两步必须原子。
# 直接 DEL 会在「自己的 TTL 已过期、锁已被别人拿走」时删掉别人的锁，
# 于是第二个等待者以为锁空了进来 —— 竞态重新出现。
_LUA_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


async def try_acquire(thread_id: str, redis_client=None) -> str | None:
    """尝试抢占会话锁。成功返回 token，失败返回 None。

    ★ Redis 异常时 fail-closed（返回 None，调用方按「忙」处理）。
      分布式锁的职责是防并发写坏状态 —— 状态写坏是不可逆的数据损坏，
      而误报忙只是让用户重试一次。两者不对称，所以往安全一侧倒。
      （对比 rate_limit 是 fail-open：它只影响成本，不影响正确性。）
    """
    redis_client = redis_client or _default_client()
    token = uuid.uuid4().hex
    key = f"{LOCK_PREFIX}:{thread_id}"
    try:
        ok = await redis_client.set(
            key, token, nx=True, px=settings.TRIAGE_LOCK_TIMEOUT_SECONDS * 1000
        )
    except Exception as e:
        logger.warning(f"[TRIAGE-LOCK] Redis 不可用，拒绝并发处理 thread={thread_id}: {e}")
        return None
    return token if ok else None


async def acquire_with_retry(thread_id: str, redis_client=None) -> str | None:
    """抢占会话锁，抢不到则短暂重试。

    区分两种「抢不到」：
      - 同进程内的另一条并发消息（用户连点、多标签页）→ 对方很快跑完，
        稍等 0.5s 重试即可，用户不该看到错误；
      - 另一进程正在跑长诊断 → 重试几次后仍失败，快速返回「忙」。
    次数与间隔由 TRIAGE_LOCK_RETRY_* 配置，设为 0 次即退化为立即失败。
    """
    import asyncio

    for attempt in range(settings.TRIAGE_LOCK_RETRY_TIMES + 1):
        token = await try_acquire(thread_id, redis_client)
        if token is not None:
            if attempt:
                logger.info(f"[TRIAGE-LOCK] 重试 {attempt} 次后取得锁 thread={thread_id}")
            return token
        if attempt < settings.TRIAGE_LOCK_RETRY_TIMES:
            await asyncio.sleep(settings.TRIAGE_LOCK_RETRY_INTERVAL_SECONDS)
    return None


async def release(thread_id: str, token: str, redis_client=None) -> bool:
    """释放锁（Lua CAS：只删自己那把）。返回是否真的删掉了。

    返回 False 有两种含义，都不需要调用方处理：
      - 锁已因 TTL 到期被 Redis 回收（自己跑太久）；
      - 锁已被别人持有（自己超时后别人抢到了）。
    两种情况都不该删别人的锁。
    """
    redis_client = redis_client or _default_client()
    key = f"{LOCK_PREFIX}:{thread_id}"
    try:
        deleted = await redis_client.eval(_LUA_RELEASE, 1, key, token)
        if not deleted:
            logger.warning(f"[TRIAGE-LOCK] 释放时锁已易主或过期 thread={thread_id}")
        return bool(deleted)
    except Exception as e:
        # 释放失败不抛：锁会由 TTL 兜底回收，不该让整个诊断流程跟着失败
        logger.warning(f"[TRIAGE-LOCK] 释放失败（TTL 将兜底回收）thread={thread_id}: {e}")
        return False


def _default_client():
    from src.infra.redis_cache import _redis_client

    return _redis_client


@asynccontextmanager
async def session_lock(thread_id: str, redis_client=None):
    """会话级互斥：Redis 跨进程锁在外，进程内 asyncio 锁在内。

    Raises:
        ConversationBusyError: 重试后仍抢不到锁（另一进程正在处理同一会话）。
    """
    token = await acquire_with_retry(thread_id, redis_client)
    if token is None:
        raise ConversationBusyError(thread_id)

    try:
        # 内层锁防同一进程内的并发。跨进程的并发已被上面的 Redis 锁挡住，
        # 所以这里等锁不会撞上「另一个进程的请求」。
        async with await _thread_lock(thread_id):
            yield
    finally:
        await release(thread_id, token, redis_client)
        _release_thread_lock(thread_id)


# ── 进程内锁（asyncio，仅覆盖本 worker）──
_thread_locks: dict[str, object] = {}
_locks_guard = None
_LOCKS_MAX = 4096  # 防字典无界增长：超过上限整表清一次（锁内的会重建）


def _get_locks_guard():
    # asyncio.Lock 必须在事件循环里创建（模块导入期创建会绑到错误的 loop）
    global _locks_guard
    if _locks_guard is None:
        import asyncio

        _locks_guard = asyncio.Lock()
    return _locks_guard


async def _thread_lock(thread_id: str):
    import asyncio

    async with _get_locks_guard():
        if len(_thread_locks) >= _LOCKS_MAX:
            _thread_locks.clear()
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            _thread_locks[thread_id] = lock
        return lock


def _release_thread_lock(thread_id: str) -> None:
    """会话结束时归还锁，避免字典无限膨胀。"""
    lock = _thread_locks.get(thread_id)
    if lock is not None and not lock.locked():
        _thread_locks.pop(thread_id, None)
