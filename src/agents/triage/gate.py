# ============================================================
# 分诊全局并发闸门：跨进程信号量（Redis ZSET）+ 进程内有界排队
#
# 为什么会话锁不够：会话锁的粒度是 thread_id，只保证「同一会话不并发」，
# 不限总量 —— 1000 个用户同时发起诊断就是 1000 路并发的模型调用，
# 把 LLM 端点、DB 连接池和进程内存一起打挂。闸门在全系统层面限一个
# 「同时运行的分诊轮次」上限（chat 工具 / REST 接口 / webhook 自动分诊
# 三个入口共用一个额度），超出的排队（有界、限时）或拒绝。
#
# 为什么是 ZSET 而不是 SETNX 计数：信号量要处理「持有者崩溃不归还」。
# ZSET 的 member 是持有者 token，score 是取位时间戳：
#   - 正常释放 → ZREM 自己的 member（uuid 唯一，不会误删别人）；
#   - 进程被 SIGKILL → 没人 ZREM，下一次抢占时按 score 清掉过期租约，
#     空位自动归还。持有期间后台任务定期续租（改写自己的 score），
#     正常的长任务不会被判过期。
#   score 用各进程的墙钟（time.time）：租约默认 90s，远大于 NTP 同步后
#   的节点间偏移，偏斜只是让有效租约伸缩零点几秒，可接受。
#
# 为什么排队是「轮询 + 抖动」而不是严格 FIFO 队列：跨进程的公平队列
# 要在 Redis 里维护等待列表 + 唤醒 + 崩溃等待者清理，复杂度换来的只是
# 严格的先来先到。轮询是统计意义上的先来先得，对「系统忙请重试」的
# 场景足够；排队上限（本进程）+ 等待超时保证队列有界、用户不会被
# 无限吊着。
#
# fail 方向：Redis 异常时 fail-closed（拿不到位 → 按忙处理），与会话锁
# 一致 —— 分诊本来就硬依赖 Redis（会话锁/会话存储/checkpointer），
# 闸门不引入新的宕机面；而空位计数一旦失控多放进来，就是真金白银的
# 并发模型调用，正好打在本闸门要防的问题上。
# ============================================================

from __future__ import annotations

import asyncio
import random
import time
import uuid
from contextlib import asynccontextmanager

from src.core.config import get_settings
from src.core.exceptions import TriageSystemBusyError
from src.core.logger import logger

settings = get_settings()

GATE_KEY = "triage_gate"


class TriageGateUnavailableError(Exception):
    """Redis 不可用，空位状态无法判定。

    ★ 内部异常，不穿出闸门：triage_gate 捕获后立即转 TriageSystemBusyError。
      对外语义仍是「忙」（fail-closed，与会话锁一致），区别是不对着
      已宕机的 Redis 空转完整个排队超时 —— 60 秒里 120 次注定失败的
      eval，每个请求还白占一个 FastAPI worker。
    """

# 占空位：清过期租约 → 数存量 → 未满则记入自己。三步必须原子，
# 分开的 CHECK-THEN-ACT 会让两个进程同时看到 19/20 然后都进来。
# ARGV: 1=now_ms 2=limit 3=lease_ms 4=member
_LUA_ACQUIRE = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1] - ARGV[3])
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[2]) then
    redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
    redis.call('PEXPIRE', KEYS[1], ARGV[3] * 2)
    return 1
end
return 0
"""

# 续租：只改写自己的 score（判定存活的时间戳），并顺手续上整个 key 的
# TTL（acquire 时设一次不会自动延长，长任务跑得久时兜底它不过期）。
# ARGV: 1=now_ms 2=member 3=lease_ms
_LUA_RENEW = """
if redis.call('ZSCORE', KEYS[1], ARGV[2]) then
    redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])
    redis.call('PEXPIRE', KEYS[1], ARGV[3] * 2)
    return 1
end
return 0
"""


def _default_client():
    from src.infra.redis_cache import _redis_client

    return _redis_client


async def try_acquire(redis_client=None) -> str | None:
    """尝试占一个全局空位。成功返回租约成员 token，满员返回 None。

    Raises:
        TriageGateUnavailableError: Redis 异常。闸门据此立即按忙拒绝，
            而不是继续对着死掉的 Redis 轮询到超时。
    """
    redis_client = redis_client or _default_client()
    member = uuid.uuid4().hex
    now_ms = int(time.time() * 1000)
    lease_ms = int(settings.TRIAGE_GATE_LEASE_SECONDS * 1000)
    try:
        ok = await redis_client.eval(
            _LUA_ACQUIRE, 1, GATE_KEY,
            now_ms, settings.TRIAGE_GLOBAL_CONCURRENCY, lease_ms, member,
        )
    except Exception as e:
        logger.warning(f"[TRIAGE-GATE] Redis 不可用，立即按忙处理: {e}")
        raise TriageGateUnavailableError() from e
    return member if ok else None


async def release(member: str, redis_client=None) -> bool:
    """归还空位。member 是 uuid，ZREM 只会删自己的，不存在会话锁的易主问题。

    释放失败不抛：租约到期会兜底回收，不该让诊断流程跟着失败。
    """
    redis_client = redis_client or _default_client()
    try:
        return bool(await redis_client.zrem(GATE_KEY, member))
    except Exception as e:
        logger.warning(f"[TRIAGE-GATE] 释放失败（租约将到期兜底回收）: {e}")
        return False


async def renew(member: str, redis_client=None) -> bool:
    """续租。返回 False = 空位确认已不属于我们（租约过期被回收）；
    Redis 异常往上抛，由 _renew_loop 区分「网络抖动」和「空位没了」。"""
    redis_client = redis_client or _default_client()
    now_ms = int(time.time() * 1000)
    lease_ms = int(settings.TRIAGE_GATE_LEASE_SECONDS * 1000)
    ok = await redis_client.eval(_LUA_RENEW, 1, GATE_KEY, now_ms, member, lease_ms)
    return bool(ok)


async def _renew_loop(member: str, redis_client, stop: asyncio.Event) -> None:
    """持有期间按租约的 1/3 周期续租，直到 stop 或空位被回收。

    续租周期取租约的 1/3：连续错过两次续租（Redis 抖动）才会被判过期。
    ★ 不设周期下限 —— 周期一旦 ≥ 租约，续租就永远追不上过期，空位
    必然被回收；比例规则对任何租约取值都自洽。

    网络抖动（Redis 异常）不退出：下一周期重试即可，空位此时仍归我们。
    确认空位被回收（renew 返回 False）才停止 —— 继续续也续不回来，
    只能靠跑完后 release 的空操作收尾。
    """
    interval = settings.TRIAGE_GATE_LEASE_SECONDS / 3
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            alive = await renew(member, redis_client)
        except Exception as e:
            logger.warning(f"[TRIAGE-GATE] 续租请求失败（下一周期重试）: {e}")
            continue
        if not alive:
            logger.warning(
                f"[TRIAGE-GATE] 空位已被租约超时回收 member={member[:8]}，停止续租"
            )
            return


# 本进程正在排队等空位的请求数。单事件循环内 +1/-1 之间没有 await，
# int 自增天然原子，不需要锁。
_waiters = 0


@asynccontextmanager
async def triage_gate(redis_client=None):
    """全局并发闸门：进入即排队（有界、限时），拿到空位才放行，退出归还。

    拒绝的两档（都抛 TriageSystemBusyError）：
      - 本进程排队人数已达 TRIAGE_GATE_QUEUE_MAX → 立刻拒绝。
        等到位也轮不上，不如把「稍后重试」说在前头，还省一个挂起的协程；
      - 排上了但 TRIAGE_GATE_WAIT_TIMEOUT_SECONDS 秒内等不到空位 → 拒绝。

    TRIAGE_GLOBAL_CONCURRENCY <= 0 时直通（本地开发与单测关闭闸门）。
    """
    if settings.TRIAGE_GLOBAL_CONCURRENCY <= 0:
        yield
        return

    global _waiters
    if _waiters >= settings.TRIAGE_GATE_QUEUE_MAX:
        logger.warning(f"[TRIAGE-GATE] 本进程排队已满（{_waiters}），立即拒绝")
        raise TriageSystemBusyError()
    _waiters += 1

    member = None
    renew_task = None
    stop = asyncio.Event()
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.TRIAGE_GATE_WAIT_TIMEOUT_SECONDS
        poll = settings.TRIAGE_GATE_POLL_INTERVAL_SECONDS
        while member is None:
            try:
                member = await try_acquire(redis_client)
            except TriageGateUnavailableError:
                raise TriageSystemBusyError("诊断服务暂时不可用，请稍后再试")
            if member is not None:
                break
            if loop.time() >= deadline:
                logger.warning(
                    f"[TRIAGE-GATE] 排队超时（{settings.TRIAGE_GATE_WAIT_TIMEOUT_SECONDS}s），拒绝"
                )
                raise TriageSystemBusyError()
            # ±20% 抖动：空位释放的一瞬多个等待者同拍打 Redis 会惊群
            await asyncio.sleep(poll * random.uniform(0.8, 1.2))

        if _waiters > settings.TRIAGE_GLOBAL_CONCURRENCY:
            logger.info(
                f"[TRIAGE-GATE] 排队后取得空位 member={member[:8]} 排队中={_waiters}"
            )
        renew_task = asyncio.create_task(_renew_loop(member, redis_client, stop))
        yield
    finally:
        # ★ 排队计数必须最先归还：后面两个 await 在任务被取消（客户端断开）
        #   时可能不执行，把 decrement 放在它们之后会让本进程的排队额度
        #   永久少一。
        _waiters -= 1
        if renew_task is not None:
            renew_task.cancel()
            # return_exceptions：取消与续租脚本里的异常都不该盖住业务异常
            await asyncio.gather(renew_task, return_exceptions=True)
        if member is not None:
            await release(member, redis_client)
