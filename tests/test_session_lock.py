# ============================================================
# 会话锁单元测试
#
# 覆盖：跨进程互斥（Redis SET NX）、Lua CAS 释放、
#       fail-closed、重试后仍失败、进程内排队。
#
# 用 FakeRedis 而非真 Redis：本项目的 CI 约定是「纯函数单测不进外部服务」
# （见 .github/workflows/ci.yml）。锁的核心分支（抢到/没抢到/易主/异常）
# 在假实现上完全可测，真 Redis 只多验证一次 SET NX 的原子性。
# ============================================================

import asyncio

import pytest

from src.agents.triage import lock as lk


class FakeRedis:
    """最小可用的 Redis 子集：只实现锁用到的 set(nx)/eval/get/delete。"""

    def __init__(self, *, raise_on_set=False, raise_on_eval=False):
        self.store: dict[str, str] = {}
        self.raise_on_set = raise_on_set
        self.raise_on_eval = raise_on_eval
        self.set_calls: list[dict] = []

    async def set(self, key, value, nx=False, px=None):
        self.set_calls.append({"key": key, "value": value, "nx": nx, "px": px})
        if self.raise_on_set:
            raise ConnectionError("redis down")
        if nx and key in self.store:
            return None  # SET NX 未命中，与真 Redis 一致返回 None
        self.store[key] = value
        return True

    async def eval(self, script, numkeys, key, arg):
        if self.raise_on_eval:
            raise ConnectionError("redis down")
        # 复刻 Lua CAS 语义：token 对得上才删
        if self.store.get(key) == arg:
            del self.store[key]
            return 1
        return 0


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """每个用例重置模块级进程内锁字典，避免用例间串味。"""
    monkeypatch.setattr(lk, "_thread_locks", {})
    monkeypatch.setattr(lk, "_locks_guard", None)


# ── 抢锁 ────────────────────────────────────────────────────────────────

async def test_acquire_sets_nx_with_ttl():
    """占锁必须是 SET NX + TTL，两者缺一都出问题：
    没有 NX → 谁都能覆盖别人的锁；没有 TTL → 进程崩溃后锁永久泄漏。"""
    fake = FakeRedis()
    token = await lk.try_acquire("u1:s1", fake)

    assert token is not None
    call = fake.set_calls[0]
    assert call["key"] == "triage_lock:u1:s1"
    assert call["nx"] is True
    assert call["px"] == lk.settings.TRIAGE_LOCK_TIMEOUT_SECONDS * 1000


async def test_second_acquire_fails_while_held():
    """锁被持有时第二次抢占必须失败 —— 这就是跨进程互斥本身。"""
    fake = FakeRedis()
    assert await lk.try_acquire("u1:s1", fake) is not None
    assert await lk.try_acquire("u1:s1", fake) is None


async def test_different_sessions_do_not_block_each_other():
    """锁的粒度是会话，不是全局 —— 两个用户并发诊断不该互相挡。"""
    fake = FakeRedis()
    assert await lk.try_acquire("u1:s1", fake) is not None
    assert await lk.try_acquire("u2:s2", fake) is not None


async def test_acquire_fail_closed_when_redis_down():
    """★ Redis 挂了要 fail-closed（返回 None → 调用方按「忙」处理）。

    与 rate_limit 的 fail-open 方向相反，因为：限流只影响成本，
    而锁失效会让两个请求同时写坏会话状态 —— 数据损坏不可逆，
    误报「忙」只让用户重试一次。两者不对称。"""
    fake = FakeRedis(raise_on_set=True)
    assert await lk.try_acquire("u1:s1", fake) is None


# ── 释放 ────────────────────────────────────────────────────────────────

async def test_release_removes_own_lock():
    fake = FakeRedis()
    token = await lk.try_acquire("u1:s1", fake)
    assert await lk.release("u1:s1", token, fake) is True
    assert fake.store == {}


async def test_release_does_not_delete_other_holders_lock():
    """★ 最关键的一条：自己超时后锁被别人拿走，释放时绝不能删别人的。

    真实现靠 Lua 保证「比对 + 删除」原子。这里验证 token 不匹配时不删 ——
    如果释放写成裸 DEL，这个用例会在「锁已被 B 持有」时把它删掉，
    于是 C 又能进来，互斥彻底失效。"""
    fake = FakeRedis()
    token_a = await lk.try_acquire("u1:s1", fake)

    # A 的 TTL 到期，锁被 B 抢走
    fake.store["triage_lock:u1:s1"] = "token-b"

    assert await lk.release("u1:s1", token_a, fake) is False
    assert fake.store["triage_lock:u1:s1"] == "token-b"  # B 的锁没被误删


async def test_release_swallows_redis_error():
    """释放失败不该让整个诊断流程跟着失败 —— 锁由 TTL 兜底回收。"""
    fake = FakeRedis(raise_on_eval=True)
    assert await lk.release("u1:s1", "tok", fake) is False


# ── 重试 ────────────────────────────────────────────────────────────────

async def test_retry_succeeds_after_lock_freed(monkeypatch):
    """同进程内用户连点两次：第二次短重试后应成功，而不是报错。"""
    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_RETRY_TIMES", 5)
    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_RETRY_INTERVAL_SECONDS", 0.01)

    fake = FakeRedis()
    await lk.try_acquire("u1:s1", fake)          # 第一条消息持锁

    async def _release_soon():
        await asyncio.sleep(0.02)
        fake.store.clear()                        # 模拟第一条跑完

    task = asyncio.create_task(_release_soon())
    token = await lk.acquire_with_retry("u1:s1", fake)
    await task

    assert token is not None


async def test_retry_exhausted_returns_none(monkeypatch):
    """长任务一直持锁 → 重试耗尽后返回 None，调用方快速失败不无限等。"""
    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_RETRY_TIMES", 2)
    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_RETRY_INTERVAL_SECONDS", 0.01)

    fake = FakeRedis()
    await lk.try_acquire("u1:s1", fake)
    assert await lk.acquire_with_retry("u1:s1", fake) is None


# ── session_lock 上下文 ─────────────────────────────────────────────────

async def test_session_lock_raises_when_busy(monkeypatch):
    from src.core.exceptions import ConversationBusyError

    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_RETRY_TIMES", 0)
    fake = FakeRedis()
    await lk.try_acquire("u1:s1", fake)

    with pytest.raises(ConversationBusyError) as ei:
        async with lk.session_lock("u1:s1", fake):
            pytest.fail("不该进到锁内")
    assert ei.value.thread_id == "u1:s1"


async def test_session_lock_releases_on_exception():
    """★ 异常路径必须放锁，否则会话被锁死到 TTL 到期。

    分诊流程里 LLM 超时、DB 断开都会走这条路径；
    原来靠 `async with` 展开释放，现在靠 finally，两者都要覆盖到。"""
    fake = FakeRedis()

    with pytest.raises(ValueError):
        async with lk.session_lock("u1:s1", fake):
            raise ValueError("模拟流程内异常")

    assert fake.store == {}, "异常退出后锁必须已释放"


async def test_session_lock_serializes_within_process():
    """进程内两条并发必须串行，不能同时进临界区（后写覆盖先写的根因）。"""
    fake = FakeRedis()
    order: list[str] = []

    async def worker(tag: str):
        # 重试让第二条等第一条跑完，模拟真实的排队语义
        async with lk.session_lock("u1:s1", fake):
            order.append(f"{tag}-in")
            await asyncio.sleep(0.02)
            order.append(f"{tag}-out")

    await asyncio.gather(worker("a"), worker("b"))

    # 无论谁先，进出必须成对相邻 —— 交错（a-in, b-in）就是没锁住
    assert order in (["a-in", "a-out", "b-in", "b-out"],
                     ["b-in", "b-out", "a-in", "a-out"]), order
