# ============================================================
# 会话锁：Redis 分布式锁 + 进程内 asyncio 锁，两层叠加
#
# ── 两个命名空间（同一套实现，不同前缀 = 不同职责）──
#   triage_lock:{thread_id}  分诊工作数据的互斥，保护 triage_state:{thread_id}。
#      三个入口（chat 工具 / REST /api/v1/triage / 后台）共用，key 规范必须一致。
#   chat_lock:{thread_id}    Supervisor 回合的互斥，保护 checkpointer 里该
#      thread 的消息历史。★ 必须是两把独立的锁，不能合并成一把：chat 回合内
#      会调用 call_triage_agent，工具自己会去拿 triage_lock —— 合并后就是
#      同一请求对自己加锁，而 asyncio.Lock 和 SET NX 都不可重入，直接自锁。
#      两者覆盖的资源本来就不同（控制面 vs 数据面），分开是语义正确的。
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
#
# ★ 续租（keep_alive）：TTL 只决定「持有者崩溃后多久自动解锁」，
#   不限制正常持有时长。没有续租时 TTL 就是硬墙 —— 一个 chat 回合
#   （Supervisor 决策 + 工具内整轮分诊）累加的模型耗时可以超过任何写死的
#   TTL，超了锁被 Redis 静默回收，互斥失效而调用方毫无察觉。持有期间按
#   TTL 的 1/3 周期续租（与 gate.py 同一套比例规则），于是 TTL 可以取小：
#   崩溃恢复快，正常慢任务不受影响。
#   代价要说清：续租之后「进程假死但仍存活」的持有者不再被 TTL 兜底，
#   该会话会一直忙到进程真正退出。两者相权取续租 —— 互斥失效写坏的是
#   数据（不可逆），假死只是暂时进不去（可逆），且日志有迹可循。
# ★ 锁易主必须中止本回合（fencing 的最小实现）：确认锁已不属于自己时，
#   取消持有它的那个任务，而不是「停止续租、临界区继续跑」——后者等于
#   互斥已破、双方同时写状态且毫无感知，正是这把锁要防的事故本身。
#   中止的代价是本回合作废（用户重发一次），比写坏 checkpoint 便宜得多。
# ============================================================

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

from src.core.config import get_settings
from src.core.exceptions import ConversationBusyError
from src.core.logger import logger

settings = get_settings()

# key 前缀 = 锁的职责。改这里等于改所有入口的 key 规范，必须同步改
TRIAGE_LOCK_PREFIX = "triage_lock"
CHAT_LOCK_PREFIX = "chat_lock"

# 释放锁：比对 token 再删，两步必须原子。
# 直接 DEL 会在「自己的 TTL 已过期、锁已被别人拿走」时删掉别人的锁，
# 于是第二个等待者以为锁空了进来 —— 竞态重新出现。
_LUA_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""

# 续租：同样是「先比对 token 再延期」。
# ★ 顺序不能反：先 PEXPIRE 再比对，会把已经属于别人的锁续上，
#   等于替别人延长持有时间（比不续租更糟）。
# ARGV: 1=token 2=ttl_ms
_LUA_RENEW = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""


def lock_key(prefix: str, thread_id: str) -> str:
    """锁 key 的唯一规范出处：{前缀}:{thread_id}。"""
    return f"{prefix}:{thread_id}"


def _ttl(ttl_seconds: float | None) -> float:
    """TTL 取值：显式传入优先，否则用分诊锁的配置默认值。"""
    return settings.TRIAGE_LOCK_TIMEOUT_SECONDS if ttl_seconds is None else ttl_seconds


async def try_acquire(
    thread_id: str,
    redis_client=None,
    *,
    prefix: str = TRIAGE_LOCK_PREFIX,
    ttl_seconds: float | None = None,
) -> str | None:
    """尝试抢占会话锁。成功返回 token，失败返回 None。

    ★ Redis 异常时 fail-closed（返回 None，调用方按「忙」处理）。
      分布式锁的职责是防并发写坏状态 —— 状态写坏是不可逆的数据损坏，
      而误报忙只是让用户重试一次。两者不对称，所以往安全一侧倒。
      （对比 rate_limit 是 fail-open：它只影响成本，不影响正确性。）
    """
    redis_client = redis_client or _default_client()
    token = uuid.uuid4().hex
    key = lock_key(prefix, thread_id)
    try:
        ok = await redis_client.set(
            key, token, nx=True, px=int(_ttl(ttl_seconds) * 1000)
        )
    except Exception as e:
        logger.warning(f"[TRIAGE-LOCK] Redis 不可用，拒绝并发处理 thread={thread_id}: {e}")
        return None
    return token if ok else None


async def acquire_with_retry(
    thread_id: str,
    redis_client=None,
    *,
    prefix: str = TRIAGE_LOCK_PREFIX,
    ttl_seconds: float | None = None,
) -> str | None:
    """抢占会话锁，抢不到则短暂重试。

    区分两种「抢不到」：
      - 同进程内的另一条并发消息（用户连点、多标签页）→ 对方很快跑完，
        稍等 0.5s 重试即可，用户不该看到错误；
      - 另一进程正在跑长诊断 → 重试几次后仍失败，快速返回「忙」。
    次数与间隔由 TRIAGE_LOCK_RETRY_* 配置，设为 0 次即退化为立即失败。
    """
    for attempt in range(settings.TRIAGE_LOCK_RETRY_TIMES + 1):
        token = await try_acquire(
            thread_id, redis_client, prefix=prefix, ttl_seconds=ttl_seconds
        )
        if token is not None:
            if attempt:
                logger.info(f"[TRIAGE-LOCK] 重试 {attempt} 次后取得锁 thread={thread_id}")
            return token
        if attempt < settings.TRIAGE_LOCK_RETRY_TIMES:
            await asyncio.sleep(settings.TRIAGE_LOCK_RETRY_INTERVAL_SECONDS)
    return None


async def release(
    thread_id: str,
    token: str,
    redis_client=None,
    *,
    prefix: str = TRIAGE_LOCK_PREFIX,
) -> bool:
    """释放锁（Lua CAS：只删自己那把）。返回是否真的删掉了。

    返回 False 有两种含义，都不需要调用方处理：
      - 锁已因 TTL 到期被 Redis 回收（自己跑太久且未续租）；
      - 锁已被别人持有（自己超时后别人抢到了）。
    两种情况都不该删别人的锁。
    """
    redis_client = redis_client or _default_client()
    key = lock_key(prefix, thread_id)
    try:
        deleted = await redis_client.eval(_LUA_RELEASE, 1, key, token)
        if not deleted:
            logger.warning(f"[TRIAGE-LOCK] 释放时锁已易主或过期 thread={thread_id}")
        return bool(deleted)
    except Exception as e:
        # 释放失败不抛：锁会由 TTL 兜底回收，不该让整个诊断流程跟着失败
        logger.warning(f"[TRIAGE-LOCK] 释放失败（TTL 将兜底回收）thread={thread_id}: {e}")
        return False


async def renew(
    thread_id: str,
    token: str,
    redis_client=None,
    *,
    prefix: str = TRIAGE_LOCK_PREFIX,
    ttl_seconds: float | None = None,
) -> bool:
    """续租：锁仍属于自己则延长 TTL。返回 False = 锁已易主或过期。

    Redis 异常往上抛，由 _renew_loop 区分「网络抖动」和「锁没了」：
    前者下一周期重试即可（锁此时仍归我们），后者继续续也续不回来。
    """
    redis_client = redis_client or _default_client()
    key = lock_key(prefix, thread_id)
    return bool(
        await redis_client.eval(_LUA_RENEW, 1, key, token, int(_ttl(ttl_seconds) * 1000))
    )


async def _renew_loop(
    thread_id: str,
    token: str,
    redis_client,
    prefix: str,
    ttl_seconds: float | None,
    stop: asyncio.Event,
    holder: asyncio.Task | None = None,
) -> None:
    """持有期间按 TTL 的 1/3 周期续租，直到 stop 或锁被回收。

    周期取 TTL 的 1/3：连续错过两次续租（Redis 抖动）才会被判过期。
    ★ 不设周期下限 —— 周期一旦 ≥ TTL，续租就永远追不上过期。

    网络抖动（Redis 异常）不退出：下一周期重试即可，空位此时仍归我们。
    确认锁被回收（renew 返回 False）则不再只是停止续租——同时取消持有
    本锁的任务（fencing）：锁已易主，继续跑临界区 = 双方并发写同一份
    会话状态，正是这把锁要防的事故。宁可本回合作废，不可双写。
    """
    interval = _ttl(ttl_seconds) / 3
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            alive = await renew(
                thread_id, token, redis_client, prefix=prefix, ttl_seconds=ttl_seconds
            )
        except Exception as e:
            logger.warning(f"[TRIAGE-LOCK] 续租请求失败（下一周期重试）thread={thread_id}: {e}")
            continue
        if not alive:
            logger.warning(f"[TRIAGE-LOCK] 锁已易主或过期，停止续租 thread={thread_id}")
            if holder is not None and not holder.done():
                logger.warning(
                    f"[TRIAGE-LOCK] fencing：中止持有已丢失锁的临界区 thread={thread_id}"
                )
                holder.cancel()
            return


def _default_client():
    from src.infra.redis_cache import _redis_client

    return _redis_client


@asynccontextmanager
async def session_lock(
    thread_id: str,
    redis_client=None,
    *,
    prefix: str = TRIAGE_LOCK_PREFIX,
    ttl_seconds: float | None = None,
    keep_alive: bool = True,
):
    """会话级互斥：Redis 跨进程锁在外，进程内 asyncio 锁在内。

    Args:
        prefix: 锁的命名空间（TRIAGE_LOCK_PREFIX / CHAT_LOCK_PREFIX）。
            同一 thread_id 上不同前缀是两把互不相干的锁，进程内锁字典
            也按前缀分开 —— 合在一起会让「外层持 chat 锁、内层再取
            triage 锁」变成自己等自己。
        ttl_seconds: 锁 TTL，None 取 TRIAGE_LOCK_TIMEOUT_SECONDS。
        keep_alive: 是否续租。True（默认）时 TTL 只决定崩溃后的回收时长；
            False 时 TTL 是硬上限（调试/token 不可续租的场景）。

    Raises:
        ConversationBusyError: 重试后仍抢不到锁（另一请求正在处理同一会话）。
    """
    token = await acquire_with_retry(
        thread_id, redis_client, prefix=prefix, ttl_seconds=ttl_seconds
    )
    if token is None:
        raise ConversationBusyError(thread_id)

    key = lock_key(prefix, thread_id)
    stop = asyncio.Event()
    renew_task = None
    if keep_alive:
        renew_task = asyncio.create_task(
            _renew_loop(
                thread_id, token, redis_client, prefix, ttl_seconds, stop,
                holder=asyncio.current_task(),
            )
        )

    try:
        # 内层锁防同一进程内的并发。跨进程的并发已被上面的 Redis 锁挡住，
        # 所以这里等锁不会撞上「另一个进程的请求」。
        async with await _thread_lock(key):
            yield
    finally:
        # 顺序：先同步取消续租，再还锁，最后等续租任务收尾。
        # ★ 不在 cancel 与 release 之间 await：本协程若在取消路径上执行到这里，
        #   中间插一个 await 就有机会被再次取消，release 被跳过 → 锁挂到 TTL
        #   才回收。cancel() 是同步的，取消已生效后才去 await。
        # 先取消再释放也不会漏续租：取消已生效，它不可能在 release 之后再跑一轮。
        if renew_task is not None:
            renew_task.cancel()
        await release(thread_id, token, redis_client, prefix=prefix)
        if renew_task is not None:
            # return_exceptions：取消与续租脚本里的异常都不该盖住业务异常
            await asyncio.gather(renew_task, return_exceptions=True)
        _release_thread_lock(key)


# ── 进程内锁（asyncio，仅覆盖本 worker）──
# key 是「前缀:thread_id」全串而不只是 thread_id：chat 回合锁与分诊数据锁
# 在同一个会话上必须落在不同的 asyncio.Lock 上（见模块头注释）。
_thread_locks: dict[str, object] = {}
_locks_guard = None
_LOCKS_MAX = 4096  # 防字典无界增长：超过上限整表清一次（锁内的会重建）


def _get_locks_guard():
    # asyncio.Lock 必须在事件循环里创建（模块导入期创建会绑到错误的 loop）
    global _locks_guard
    if _locks_guard is None:
        _locks_guard = asyncio.Lock()
    return _locks_guard


async def _thread_lock(key: str):
    async with _get_locks_guard():
        if len(_thread_locks) >= _LOCKS_MAX:
            _thread_locks.clear()
        lock = _thread_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _thread_locks[key] = lock
        return lock


def _release_thread_lock(key: str) -> None:
    """会话结束时归还锁，避免字典无限膨胀。"""
    lock = _thread_locks.get(key)
    if lock is not None and not lock.locked():
        _thread_locks.pop(key, None)
