# ============================================================
# 分诊全局并发闸门单元测试
#
# 覆盖：ZSET 信号量抢占/满员/释放、fail-closed、租约过期回收
#       （持有者崩溃场景）、续租保位、闸门直通开关、排队上限
#       立即拒绝、排队成功、排队超时拒绝、异常路径归还空位、
#       并发体上限（核心性质）。
#
# 用 FakeRedis 而非真 Redis：与 test_session_lock.py 同一约定 ——
# 「纯函数单测不进外部服务」。Lua 的原子性在假实现上按语义复刻，
# 真 Redis 只多验证一次 Lua 脚本本身的可执行性。
# ============================================================

import asyncio

import pytest

from src.agents.triage import gate as gt
from src.core.exceptions import ConversationBusyError, TriageSystemBusyError


class FakeGateRedis:
    """闸门用到的最小 Redis 子集：eval（两段 Lua 按语义复刻）+ zrem。

    两段脚本按特征串分发：抢占含 ZREMRANGEBYSCORE，续租含 ZSCORE。
    过期回收用真实时钟换算 —— score 是占用时的 now_ms，
    抢占时把 score <= now-lease 的成员清掉。
    """

    def __init__(self, *, raise_on_eval=False):
        self.zset: dict[str, float] = {}  # member -> score (now_ms)
        self.raise_on_eval = raise_on_eval

    async def eval(self, script, numkeys, key, *args):
        if self.raise_on_eval:
            raise ConnectionError("redis down")
        if "ZREMRANGEBYSCORE" in script:  # 抢占
            now, limit, lease, member = int(args[0]), int(args[1]), int(args[2]), args[3]
            cutoff = now - lease
            self.zset = {m: s for m, s in self.zset.items() if s > cutoff}
            if len(self.zset) < limit:
                self.zset[member] = float(now)
                return 1
            return 0
        # 续租：member 还在才改写 score
        now, member = int(args[0]), args[1]
        if member in self.zset:
            self.zset[member] = float(now)
            return 1
        return 0

    async def zrem(self, key, member):
        return 1 if self.zset.pop(member, None) is not None else 0


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """每个用例重置进程内排队计数，避免用例间串味。"""
    monkeypatch.setattr(gt, "_waiters", 0)


# ── 信号量抢占 / 释放 ──────────────────────────────────────────────────

async def test_acquire_under_limit_returns_member():
    fake = FakeGateRedis()
    member = await gt.try_acquire(fake)

    assert member is not None
    assert list(fake.zset) == [member]


async def test_acquire_rejected_at_limit(monkeypatch):
    """达到全局上限后必须拿不到空位 —— 这就是闸门本身。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 2)
    fake = FakeGateRedis()

    assert await gt.try_acquire(fake) is not None
    assert await gt.try_acquire(fake) is not None
    assert await gt.try_acquire(fake) is None


async def test_release_frees_slot(monkeypatch):
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    fake = FakeGateRedis()
    member = await gt.try_acquire(fake)

    assert await gt.release(member, fake) is True
    assert await gt.try_acquire(fake) is not None


async def test_release_twice_is_safe(monkeypatch):
    """重复释放不该报错：续租循环确认空位被回收后，release 是空操作。"""
    fake = FakeGateRedis()
    member = await gt.try_acquire(fake)

    assert await gt.release(member, fake) is True
    assert await gt.release(member, fake) is False


async def test_acquire_fail_closed_when_redis_down():
    """★ Redis 挂了必须 fail-closed，且以异常上报（而不是返回 None）。

    对外语义仍是「忙」—— 闸门把它转成 TriageSystemBusyError。分诊本来
    就硬依赖 Redis（会话锁/会话存储/checkpointer），闸门不引入新的宕机面；
    而空位计数失控多放进来，正是本闸门要防的事故本身。"""
    fake = FakeGateRedis(raise_on_eval=True)
    with pytest.raises(gt.TriageGateUnavailableError):
        await gt.try_acquire(fake)


async def test_gate_fails_fast_when_redis_down(monkeypatch):
    """★ Redis 宕机要立即拒绝，不许对着死掉的 Redis 空转完整个排队超时。

    回归值：WAIT_TIMEOUT 给 5 秒 —— 如果快败逻辑退化成轮询到超时，
    这个用例会跑满 5 秒（远超 1 秒红线）而不只是失败。"""
    import time

    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_WAIT_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_POLL_INTERVAL_SECONDS", 0.05)

    fake = FakeGateRedis(raise_on_eval=True)
    start = time.monotonic()
    with pytest.raises(TriageSystemBusyError):
        async with gt.triage_gate(fake):
            pytest.fail("不该进到闸门内")
    assert time.monotonic() - start < 1.0, "Redis 宕机未快速失败"
    assert gt._waiters == 0, "拒绝路径的排队计数必须归还"


# ── 租约：崩溃回收与续租 ────────────────────────────────────────────────

async def test_abandoned_slot_reclaimed_after_lease(monkeypatch):
    """★ 持有者崩溃（SIGKILL，没走 release）后，空位在租约过期后自动归还。

    如果回收逻辑写成「只看 ZCARD 不清过期成员」，这个用例会一直拿不到
    空位 —— 一次进程崩溃就把全局额度永久打掉一格。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_LEASE_SECONDS", 0.06)
    fake = FakeGateRedis()
    await gt.try_acquire(fake)

    await asyncio.sleep(0.12)  # 租约已过期，但没人来释放
    assert await gt.try_acquire(fake) is not None


async def test_renew_keeps_alive_slot_beyond_lease(monkeypatch):
    """续租改写存活时间戳：跑得久但活着的任务，空位不许被回收。

    时序（租约 500ms）：t≈200ms 续一次租 → t≈550ms 探测。此刻原始租约
    已过期 50ms+（不续租必被回收），距续租后的过期点（≈700ms）还有余量
    —— 两侧都留了余量，避免事件循环抖动造成偶发。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_LEASE_SECONDS", 0.5)
    fake = FakeGateRedis()
    member = await gt.try_acquire(fake)

    await asyncio.sleep(0.2)
    assert await gt.renew(member, fake) is True

    await asyncio.sleep(0.35)
    assert await gt.try_acquire(fake) is None, "续租过的空位被误回收"


async def test_renew_returns_false_for_unknown_member():
    """空位已被回收后续租确认失败 —— 续租循环据此停止。"""
    fake = FakeGateRedis()
    assert await gt.renew("nobody", fake) is False


# ── triage_gate 上下文 ─────────────────────────────────────────────────

async def test_gate_passes_through_when_disabled(monkeypatch):
    """TRIAGE_GLOBAL_CONCURRENCY <= 0 直通：本地开发/单测关闸门。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 0)
    fake = FakeGateRedis(raise_on_eval=True)  # 连 Redis 挂了都不该被问到

    ran = False
    async with gt.triage_gate(fake):
        ran = True
    assert ran


async def test_gate_rejects_immediately_when_queue_full(monkeypatch):
    """排队人数达上限 → 立刻拒绝，不进等待循环白占协程。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_QUEUE_MAX", 2)
    monkeypatch.setattr(gt, "_waiters", 2)  # 模拟已有 2 个在排队

    fake = FakeGateRedis()
    with pytest.raises(TriageSystemBusyError):
        async with gt.triage_gate(fake):
            pytest.fail("不该进到闸门内")
    assert fake.zset == {}, "被拒绝的请求不该占空位"
    assert gt._waiters == 2, "拒绝路径不该动排队计数"


async def test_gate_times_out_when_never_served(monkeypatch):
    """排上了但等不到空位 → 超时拒绝，不无限吊着用户。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_WAIT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_POLL_INTERVAL_SECONDS", 0.01)

    fake = FakeGateRedis()
    await gt.try_acquire(fake)  # 唯一的空位被别人占着

    with pytest.raises(TriageSystemBusyError):
        async with gt.triage_gate(fake):
            pytest.fail("不该进到闸门内")
    assert len(fake.zset) == 1, "超时拒绝后存量空位不变"


async def test_gate_queues_then_succeeds(monkeypatch):
    """满员时排队，空位释放后应能排到 —— 「排队」路径。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_POLL_INTERVAL_SECONDS", 0.01)

    fake = FakeGateRedis()
    holder = await gt.try_acquire(fake)

    async def _release_soon():
        await asyncio.sleep(0.03)
        await gt.release(holder, fake)

    releaser = asyncio.create_task(_release_soon())
    async with gt.triage_gate(fake):
        pass
    await releaser


async def test_gate_releases_slot_on_body_exception(monkeypatch):
    """★ 业务异常路径必须归还空位，否则一格额度锁死到租约过期。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    fake = FakeGateRedis()

    with pytest.raises(ValueError):
        async with gt.triage_gate(fake):
            raise ValueError("模拟诊断流程内异常")

    assert fake.zset == {}, "异常退出后空位必须已归还"
    assert await gt.try_acquire(fake) is not None


async def test_gate_caps_concurrent_bodies(monkeypatch):
    """★ 核心性质：闸门内同时跑的业务体 ≤ 全局上限，且没人被饿死。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 2)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_POLL_INTERVAL_SECONDS", 0.005)

    fake = FakeGateRedis()
    inside = 0
    max_inside = 0
    finished = 0

    async def worker():
        nonlocal inside, max_inside, finished
        async with gt.triage_gate(fake):
            inside += 1
            max_inside = max(max_inside, inside)
            await asyncio.sleep(0.02)
            inside -= 1
            finished += 1

    await asyncio.gather(*(worker() for _ in range(6)))

    assert max_inside <= 2, f"并发体越限：峰值 {max_inside}"
    assert finished == 6, "有请求被饿死没跑成"
    assert fake.zset == {}, "全部跑完后空位应清零"


async def test_gate_keeps_slot_alive_for_slow_body(monkeypatch):
    """闸门内跑长任务（超过单次租约时长）：续租循环保位，期间不许放人进来。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_LEASE_SECONDS", 0.06)

    fake = FakeGateRedis()
    state = {"intruded": False}

    async def prober():
        # 0.24s = 4 个租约周期，持有者的空位全靠续租撑住
        for _ in range(12):
            await asyncio.sleep(0.02)
            if await gt.try_acquire(fake) is not None:
                state["intruded"] = True

    async with gt.triage_gate(fake):
        await prober()

    assert state["intruded"] is False, "长任务持有期间空位被别人抢走（续租失效）"


async def test_waiter_count_restored_on_cancellation(monkeypatch):
    """客户端断开（任务被取消）时排队计数必须归还。

    decrement 放在 finally 的所有 await 之前 —— 取消风暴（uvicorn 会对
    断连任务反复 cancel）下 finally 里的 await 可能执行不到，计数一旦
    泄漏，本进程的排队额度就永久变小。"""
    monkeypatch.setattr(gt.settings, "TRIAGE_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_WAIT_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(gt.settings, "TRIAGE_GATE_POLL_INTERVAL_SECONDS", 0.01)

    fake = FakeGateRedis()
    await gt.try_acquire(fake)  # 占满唯一空位 → 后续请求卡在排队轮询

    async def stuck_waiter():
        async with gt.triage_gate(fake):
            pytest.fail("不该进到闸门内")

    task = asyncio.create_task(stuck_waiter())
    await asyncio.sleep(0.03)
    assert gt._waiters == 1, "应处于排队中"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gt._waiters == 0, "取消后排队计数泄漏"


# ── worker_tools 被拒分流（_triage_busy_fallback）──────────────────────

class _FakeStore:
    """TriageSessionStore 的最小替身：只看会话有没有分诊进度。"""

    def __init__(self, has_progress: bool):
        self.has_progress = has_progress

    async def load(self, thread_id):
        return object() if self.has_progress else None


async def test_busy_fallback_reraises_for_pending_triage():
    """★ 追问轮（store 有进度 = resume）被拒必须原样上抛。

    工具一旦 return（哪怕回的是一句「稍后再试」），langgraph 就把挂起的
    interrupt 当「节点完成」消费掉：checkpoint 越过追问点而 store 还停在
    原轮 —— 追问文本变空、轮次错位，状态机永久损坏。"""
    from src.agents.tools.worker_tools import _triage_busy_fallback

    with pytest.raises(TriageSystemBusyError):
        await _triage_busy_fallback(TriageSystemBusyError(), _FakeStore(True), "u1:s1")
    with pytest.raises(ConversationBusyError):
        await _triage_busy_fallback(ConversationBusyError("u1:s1"), _FakeStore(True), "u1:s1")


async def test_busy_fallback_returns_text_for_fresh_diagnosis():
    """全新诊断被拒：没有可错位的状态，返回人话由模型转述给用户。"""
    from src.agents.tools.worker_tools import _triage_busy_fallback

    out = await _triage_busy_fallback(TriageSystemBusyError(), _FakeStore(False), "u1:s1")
    assert "较多" in out and "稍后" in out

    out2 = await _triage_busy_fallback(ConversationBusyError("u1:s1"), _FakeStore(False), "u1:s1")
    assert "上一次" in out2


async def test_busy_fallback_reraises_when_progress_check_fails():
    """★ 进度读取失败（Redis 抖动）时判定不了是否追问轮 → 按状态安全一侧倒：
    原样上抛「忙」异常（可逆），绝不冒文本返回消费 interrupt 的险（不可逆）。
    上抛的必须是原异常而不是读库错误 —— 路由的分流处理依赖异常类型。"""
    from src.agents.tools.worker_tools import _triage_busy_fallback

    class _BrokenStore:
        async def load(self, thread_id):
            raise ConnectionError("redis down")

    with pytest.raises(TriageSystemBusyError):
        await _triage_busy_fallback(TriageSystemBusyError(), _BrokenStore(), "u1:s1")
    with pytest.raises(ConversationBusyError):
        await _triage_busy_fallback(ConversationBusyError("u1:s1"), _BrokenStore(), "u1:s1")
