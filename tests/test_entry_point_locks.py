# ============================================================
# 入口级并发防护测试
#
# 锁本身的行为在 test_session_lock.py 覆盖；这里测的是【入口有没有接上锁】。
# 这是一类单测测不出来的缺陷：机制全对、既有用例全绿，但某个入口忘了调用
# 它，并发照样写坏状态 —— 因为缺陷是「少调用了一个上下文管理器」，
# 不是某个函数行为错。所以每个用例都走真实的入口函数（chat 路由 /
# REST 路由 / 流式生成器），断言临界区不会被并发进入。
#
# 覆盖的不变量：
#   1. 同一 thread_id 上 chat 回合锁与分诊数据锁是两个独立命名空间
#      （合并 = chat 回合内调用分诊工具时自锁）
#   2. 锁在持有期间续租；易主后停止续租且不替新持有者续期
#   3. 同一会话的并发 chat（同步/流式）被挡住，图只进入一次
#   4. 同一会话的并发 REST 分诊被挡住，且返回 409 业务码而非 500
#   5. REST 与 chat 工具落在同一把 triage_lock 上（两者写同一份会话进度）
#
# 仍然用 FakeRedis：与 test_session_lock.py 同一约定（纯单测不进外部服务）。
# ============================================================

import asyncio
from types import SimpleNamespace

import pytest

from src.agents.triage import lock as lk
from src.api.routers import chat as chat_mod
from src.api.routers import triage as triage_mod
from src.core.deps import UserContext
from src.core.exceptions import ERR_CONVERSATION_BUSY, BizException


class FakeRedis:
    """最小 Redis 子集：set(nx, px) + 两个 Lua 脚本的语义。

    脚本按对象身份分发（不是按字符串匹配），这样「比对 token 再操作」的
    语义由 lk 里的常量定义，测试不会因为注释改动而失效。
    """

    def __init__(self):
        self.store: dict[str, str] = {}
        self.renew_calls: list[tuple[str, str]] = []

    async def set(self, key, value, nx=False, px=None):
        if nx and key in self.store:
            return None  # SET NX 未命中，与真 Redis 一致返回 None
        self.store[key] = value
        return True

    async def eval(self, script, numkeys, key, *args):
        token = args[0]
        if script is lk._LUA_RELEASE:
            if self.store.get(key) == token:
                del self.store[key]
                return 1
            return 0
        assert script is lk._LUA_RENEW
        self.renew_calls.append((key, token))
        if self.store.get(key) == token:
            return 1
        return 0


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """重置进程内锁字典 + 注入 Redis 替身 + 关掉重试与闸门。

    - 进程内锁字典必须逐个用例重置：锁对象绑事件循环，跨用例复用会炸。
    - 重试次数置 0：被拒的场景不该在测试里真的等 1.5 秒。
    - 闸门置 0（配置语义 <=0 直通）：本文件测的是互斥，不是容量。
    """
    monkeypatch.setattr(lk, "_thread_locks", {})
    monkeypatch.setattr(lk, "_locks_guard", None)
    fake = FakeRedis()
    monkeypatch.setattr(lk, "_default_client", lambda: fake)
    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_RETRY_TIMES", 0)
    monkeypatch.setattr(lk.settings, "TRIAGE_GLOBAL_CONCURRENCY", 0)
    return fake


USER = UserContext(user_id="7", session_id="s1", role="engineer", business_line="ia")
CHAT_REQ = chat_mod.ChatRequest(session_id="s1", message="车机黑屏了")
TRIAGE_REQ = triage_mod.TriageRequest(raw_input="车机偶尔黑屏", session_id="s1")


class FakeSupervisor:
    """Supervisor 替身：记录「同时进入图的条数」，互斥失效时它会是 2。"""

    def __init__(self):
        self.inflight = 0
        self.max_inflight = 0

    def _enter(self):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)

    async def aget_state(self, config):
        return SimpleNamespace(tasks=[])  # 无挂起追问

    async def ainvoke(self, payload, config=None, context=None):
        self._enter()
        try:
            await asyncio.sleep(0.05)
        finally:
            self.inflight -= 1
        return {"messages": [SimpleNamespace(content="收到")]}

    async def astream(self, payload, config=None, context=None, stream_mode=None):
        self._enter()
        try:
            await asyncio.sleep(0.05)
            yield (SimpleNamespace(content="收到"), {})
        finally:
            self.inflight -= 1


class _BlockingSupervisor(FakeSupervisor):
    """推一帧后挂住，直到测试主动关掉生成器（模拟客户端断连）。"""

    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()

    async def astream(self, payload, config=None, context=None, stream_mode=None):
        self._enter()
        try:
            yield (SimpleNamespace(content="第一帧"), {})
            await self.gate.wait()
        finally:
            self.inflight -= 1


class FakeTriageAgent:
    """分诊替身：记录并发进入条数。"""

    def __init__(self, status: str = "converged", follow_ups: list[str] | None = None):
        self.inflight = 0
        self.max_inflight = 0
        self.status = status
        self.follow_ups = follow_ups or []

    async def diagnose(self, raw_input, session_id=None, issue_id=None, existing_state=None, **kw):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(0.05)
        finally:
            self.inflight -= 1
        asking = self.status == "asking"
        return {
            "session_id": session_id or "generated-session",
            "status": self.status,
            "round": 1,
            "normalized_phenomena": [],
            "candidate_causes": [],
            "confidence": 0.9,
            "follow_up_questions": self.follow_ups,
            "diagnostic_summary": "" if asking else "结论",
            # asking 时才带 _state（路由据此落盘进度）
            "_state": {"round": 1} if asking else None,
        }


class FakeStore:
    """会话进度存储替身：本文件只关互斥，不关心进度内容。"""

    def __init__(self):
        self.saved: list[tuple[str, str]] = []
        self.cleared: list[str] = []

    async def load(self, thread_id):
        return None

    async def save(self, thread_id, state, reply):
        self.saved.append((thread_id, reply))

    async def clear(self, thread_id):
        self.cleared.append(thread_id)


# ── 1. 两个命名空间必须独立 ──────────────────────────────────────────────

async def test_chat_and_triage_locks_are_separate_namespaces(fake_redis):
    """★ 同一 thread_id 上 chat 回合锁与分诊数据锁必须是两把独立的锁。

    chat 回合内会调用 call_triage_agent，工具自己去拿 triage_lock。两者若共用
    同一 key，外层持锁后再取一次就是自己等自己（asyncio.Lock 与 SET NX 都
    不可重入）—— 表现是整个请求永久挂起。这条用例把它钉住。
    """

    async def _nested():
        async with lk.session_lock("7:s1", prefix=lk.CHAT_LOCK_PREFIX):
            assert "chat_lock:7:s1" in fake_redis.store
            # 工具那一层：默认前缀，必须能取到
            async with lk.session_lock("7:s1"):
                assert "triage_lock:7:s1" in fake_redis.store

    # 用超时兜底：真自锁时用例失败而不是把 CI 挂死
    await asyncio.wait_for(_nested(), timeout=2.0)
    assert fake_redis.store == {}, "退出后两把锁都必须释放"


# ── 2. 续租 ──────────────────────────────────────────────────────────────

async def test_lock_renewed_while_held(monkeypatch, fake_redis):
    """★ TTL 只决定「持有者崩溃后多久回收」，不是正常持有时长的硬墙。

    没有续租时，一轮耗时超过 TTL 的 chat 回合会被 Redis 静默摘掉锁，
    互斥失效而调用方毫无察觉 —— 这里把 TTL 压到 90ms 模拟那种场景。
    """
    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_TIMEOUT_SECONDS", 0.09)
    async with lk.session_lock("7:s1"):
        await asyncio.sleep(0.12)  # 超过 TTL，期间必须发生续租

    assert fake_redis.renew_calls, "持有超过 TTL 却从未续租"
    assert {k for k, _ in fake_redis.renew_calls} == {"triage_lock:7:s1"}


async def test_renew_stops_after_lock_taken_over(monkeypatch, fake_redis):
    """锁易主后必须停止续租，且绝不替新持有者续期。

    ★ fencing（lock._renew_loop 新语义）：确认易主后还会中止持有该锁的
      临界区（CancelledError 弹出），而不是「停止续租但继续跑」——
      继续跑 = 双方并发写同一会话状态，正是锁要防的事故。
    """
    monkeypatch.setattr(lk.settings, "TRIAGE_LOCK_TIMEOUT_SECONDS", 0.09)
    with pytest.raises(asyncio.CancelledError):
        async with lk.session_lock("7:s1"):
            fake_redis.store["triage_lock:7:s1"] = "new-holder"  # 模拟被抢走
            await asyncio.sleep(0.12)

    assert all(tok != "new-holder" for _, tok in fake_redis.renew_calls), "替别人续租了"
    assert len(fake_redis.renew_calls) <= 1, "发现自己被踢出后仍在续租"
    assert fake_redis.store["triage_lock:7:s1"] == "new-holder", "释放时误删了新持有者的锁"


# ── 3. chat 入口 ────────────────────────────────────────────────────────

async def test_chat_rejects_concurrent_request_same_session(monkeypatch, fake_redis):
    """★ 同一会话两条并发 chat 必须互斥，图只被进入一次。

    没有回合锁时两条请求各自读历史、各自写 checkpointer（后写覆盖先写 →
    丢消息），并各自判断「有没有挂起追问」形成 TOCTOU（可能同时发 resume）。
    max_inflight == 2 就是它。
    """
    agent = FakeSupervisor()

    async def _get_agent():
        return agent

    monkeypatch.setattr(chat_mod, "get_supervisor_agent", _get_agent)
    # 钉住编排图开关：.env 灰度翻开后测试也读 .env，不钉会绕过 fake agent
    monkeypatch.setattr(chat_mod.settings, "SUBGRAPH_TRIAGE_ENABLED", False)

    results = await asyncio.gather(
        chat_mod.chat(CHAT_REQ, USER, None),
        chat_mod.chat(CHAT_REQ, USER, None),
        return_exceptions=True,
    )

    assert agent.max_inflight == 1, "同一会话的图调用并发进入了"
    oks = [r for r in results if not isinstance(r, Exception)]
    errs = [r for r in results if isinstance(r, Exception)]
    assert len(oks) == 1 and len(errs) == 1
    # 被拒的那条走「忙」兜底：无挂起追问时是 409 业务码，不是 500
    assert isinstance(errs[0], BizException)
    assert errs[0].code == ERR_CONVERSATION_BUSY


async def test_chat_stream_rejects_concurrent_request(monkeypatch, fake_redis):
    """流式入口同样要持锁，且与同步入口共用同一命名空间。"""
    agent = FakeSupervisor()

    async def _get_agent():
        return agent

    monkeypatch.setattr(chat_mod, "get_supervisor_agent", _get_agent)
    # 钉住编排图开关：.env 灰度翻开后测试也读 .env，不钉会绕过 fake agent
    monkeypatch.setattr(chat_mod.settings, "SUBGRAPH_TRIAGE_ENABLED", False)

    async with lk.session_lock("7:s1", prefix=lk.CHAT_LOCK_PREFIX):
        resp = await chat_mod.chat_stream(CHAT_REQ, USER, None)
        frames = [f async for f in resp.body_iterator]

    assert any('"type": "error"' in f for f in frames), frames
    assert agent.max_inflight == 0, "锁没挡住，流式入口照样进了图"


async def test_chat_stream_releases_lock_on_client_disconnect(monkeypatch, fake_redis):
    """★ 客户端断连必须释放回合锁，而不是等 90 秒 TTL。

    Starlette 在客户端断开时取消流式任务，CancelledError 落在 yield 处，
    `async with` 随之展开释放。这里用 aclose() 复现同一条路径 ——
    锁没放掉的话，该会话在 TTL 到期前谁都进不来（用户表现为「发不出消息」）。
    """
    agent = _BlockingSupervisor()

    async def _get_agent():
        return agent

    monkeypatch.setattr(chat_mod, "get_supervisor_agent", _get_agent)
    # 钉住编排图开关：.env 灰度翻开后测试也读 .env，不钉会绕过 fake agent
    monkeypatch.setattr(chat_mod.settings, "SUBGRAPH_TRIAGE_ENABLED", False)

    resp = await chat_mod.chat_stream(CHAT_REQ, USER, None)
    agen = resp.body_iterator
    first = await agen.__anext__()
    assert "第一帧" in first
    assert fake_redis.store, "流式期间应当持有回合锁"

    await agen.aclose()  # ← 客户端断连
    assert fake_redis.store == {}, "断连后锁必须已释放"

    # 释放的直接后果：同一会话可以立刻再进，而不是被 TTL 挡住
    async with lk.session_lock("7:s1", prefix=lk.CHAT_LOCK_PREFIX):
        pass


# ── 4/5. REST 入口 ──────────────────────────────────────────────────────

def _patch_triage_deps(monkeypatch, agent, store=None):
    store = store or FakeStore()
    monkeypatch.setattr(triage_mod, "get_triage_agent", lambda: agent)
    monkeypatch.setattr(triage_mod, "TriageSessionStore", lambda _redis: store)
    return store


async def test_rest_triage_rejects_concurrent_same_session(monkeypatch, fake_redis):
    """★ 同一会话两条并发 REST 分诊必须互斥，且忙是 409 而不是 500。

    ConversationBusyError 故意不是 BizException 子类（它要在 chat 路由里被
    捕获以保住图内状态），REST 路由漏了转换就会落到全局处理器 → 500，
    用户以为系统挂了。这里同时锁住「互斥」和「错误码」两件事。
    """
    agent = FakeTriageAgent()
    _patch_triage_deps(monkeypatch, agent)

    results = await asyncio.gather(
        triage_mod.triage(TRIAGE_REQ, USER, None),
        triage_mod.triage(TRIAGE_REQ, USER, None),
        return_exceptions=True,
    )

    assert agent.max_inflight == 1, "同一会话的分诊并发进入了"
    errs = [r for r in results if isinstance(r, Exception)]
    assert len(errs) == 1
    assert isinstance(errs[0], BizException)
    assert errs[0].code == ERR_CONVERSATION_BUSY


async def test_rest_and_chat_triage_share_one_lock(monkeypatch, fake_redis):
    """★ REST 与 chat 工具写的是同一份 triage_state:{thread_id}，
    必须落在同一把锁上。

    这里用「工具那一层」的取锁方式（默认前缀）持锁，断言 REST 请求被挡住。
    若哪天有人给其中一条路径换掉前缀，两个入口就会各锁各的、同时改同一份
    进度 —— 那条改动的后果就是 round 错位，而这条用例会立刻变红。
    """
    agent = FakeTriageAgent()
    _patch_triage_deps(monkeypatch, agent)

    entered = asyncio.Event()
    release_ev = asyncio.Event()

    async def _chat_tool_side():
        # 等价于 call_triage_agent 里的 session_lock(thread_id)
        async with lk.session_lock("7:s1"):
            entered.set()
            await release_ev.wait()

    holder = asyncio.create_task(_chat_tool_side())
    await entered.wait()
    try:
        with pytest.raises(BizException) as ei:
            await triage_mod.triage(TRIAGE_REQ, USER, None)
        assert ei.value.code == ERR_CONVERSATION_BUSY
        assert agent.max_inflight == 0, "锁没挡住，分诊被并发执行了"
    finally:
        release_ev.set()
        await holder


# ── 6. REST 的另外两条分支 ──────────────────────────────────────────────

async def test_rest_triage_new_session_skips_lock(monkeypatch, fake_redis):
    """不带 session_id = 新建会话，不取锁（没有第二方能摸到它），
    这条分支走的是 nullcontext —— 钉住它真的能跑通而不是抛异常。"""
    agent = FakeTriageAgent()
    _patch_triage_deps(monkeypatch, agent)

    resp = await triage_mod.triage(
        triage_mod.TriageRequest(raw_input="车机偶尔黑屏"), USER, None
    )

    assert resp.data.session_id == "generated-session"
    assert fake_redis.store == {}, "新建会话不该占用任何会话锁"


async def test_rest_triage_persists_followup_while_locked(monkeypatch, fake_redis):
    """未收敛时进度必须在临界区内落盘：诊断与落盘共用一个临界区，
    否则并发请求会读到「诊断完了但进度还没写」的半成品状态。"""
    agent = FakeTriageAgent(status="asking", follow_ups=["故障出现时车速多少？"])
    store = _patch_triage_deps(monkeypatch, agent)

    resp = await triage_mod.triage(TRIAGE_REQ, USER, None)

    assert resp.data.status == "asking"
    assert store.saved == [("7:s1", "故障出现时车速多少？")]
    assert store.cleared == []


async def test_rest_triage_clears_progress_after_converge(monkeypatch, fake_redis):
    """收敛后必须清进度（与 chat 路径的 _finish_triage 一致）。

    不清的话，同一 session_id 的下一次诊断会读到上一段进度、按「第 N+1 轮」
    续聊，两个不相干的诊断串在一起。
    """
    agent = FakeTriageAgent()  # 默认 converged
    store = _patch_triage_deps(monkeypatch, agent)

    resp = await triage_mod.triage(TRIAGE_REQ, USER, None)

    assert resp.data.status == "converged"
    assert store.cleared == ["7:s1"], "收敛后应清掉进度与挂起标记"
    assert store.saved == []
