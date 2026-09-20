# ============================================================
# 分诊进度的 TTL 约定与「进度丢失」判定
#
# 锁住三件事：
#   1. 配置不变式：进度 TTL 必须严格大于 checkpointer TTL
#   2. 读时续期：load() 与 checkpointer 的 refresh_on_read 对齐
#   3. 进度丢失但仍有挂起追问 → 放弃对话，绝不按首轮重放
#
# 第 3 条不是假想问题：用 langgraph MemorySaver 按 _run_triage_turn 的结构
# 实测过，store 丢失时按首轮重跑会得到
#   [fresh-round, consumed A1, fresh-round, consumed A1, consumed A2]
# 即白跑一整轮，并把已经处理过的 A1 再当新输入处理一次（round 多涨一轮），
# 追问序列从此与用户看到的不一致。
# ============================================================

import pytest

from src.agents.tools import worker_tools
from src.agents.triage.session_store import TriageSessionStore
from src.agents.triage.state import TriagePhase, TriageState
from src.core.config import Settings
from tests.test_config_security import _prod_settings


class _StubRedis:
    """set/get/delete/expire 的最小替身（异步接口），记录 expire 调用。"""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.expires: list[tuple[str, int]] = []

    async def set(self, key, value, ex=None):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, *keys):
        for k in keys:
            self.data.pop(k, None)

    async def expire(self, key, ttl):
        self.expires.append((key, ttl))


class _FakeAgent:
    """替代 TriageAgent：放弃分支不该构造它，首轮分支只用到 _build_deps。"""

    def _build_deps(self):
        return None


# ── 1. 配置不变式 ───────────────────────────────────────────────────────

def test_state_ttl_outlives_checkpointer():
    """★ 进度必须比挂起的 interrupt 活得久 —— 这是不变式，不是偏好。

    取实际生效的配置（默认值 + 本机 .env），所以任何人调整这两个值、
    把顺序弄反，这里立刻红。
    """
    s = Settings(APP_ENV="dev")
    assert s.TRIAGE_STATE_TTL_SECONDS > s.CHECKPOINTER_TTL_MINUTES * 60, (
        "分诊进度 TTL 不得短于 checkpointer TTL：进度先过期会让 resume "
        "误判为首轮重跑，把历史回答再处理一遍"
    )


def test_prod_validation_rejects_inverted_ttl():
    """生产启动期就拒绝这个错配，而不是等线上出现错位会话。"""
    s = _prod_settings(TRIAGE_STATE_TTL_SECONDS=60, CHECKPOINTER_TTL_MINUTES=7 * 24 * 60)
    with pytest.raises(RuntimeError, match="TRIAGE_STATE_TTL_SECONDS"):
        s.validate_production()


# ── 2. 存取与读时续期 ───────────────────────────────────────────────────

async def test_save_marks_pending_and_clear_removes_both():
    """save 必须同时立起挂起标记，clear 必须同时清掉两者。"""
    redis = _StubRedis()
    store = TriageSessionStore(redis)

    await store.save("7:s1", TriageState(round=1), "故障出现时车速多少？")
    assert "triage_state:7:s1" in redis.data
    assert "triage_pending:7:s1" in redis.data
    assert await store.is_pending("7:s1") is True

    await store.clear("7:s1")
    assert redis.data == {}
    assert await store.is_pending("7:s1") is False


async def test_load_refreshes_ttl():
    """读时续期（对齐 checkpointer 的 refresh_on_read）。

    没有它，挂起期间进度的寿命会短于 interrupt —— 用户隔几天回来回答时
    走的就是那条错位路径。
    """
    redis = _StubRedis()
    store = TriageSessionStore(redis)
    await store.save("7:s1", TriageState(round=1), "Q1")

    redis.expires.clear()
    assert await store.load("7:s1") is not None
    assert redis.expires == [
        ("triage_state:7:s1", Settings(APP_ENV="dev").TRIAGE_STATE_TTL_SECONDS)
    ]


async def test_load_missing_does_not_refresh():
    """读不到就没什么可续的，也不该去写一个不存在的 key。"""
    redis = _StubRedis()
    assert await TriageSessionStore(redis).load("7:none") is None
    assert redis.expires == []


# ── 3. 进度丢失时的判定 ─────────────────────────────────────────────────

async def test_expired_progress_with_pending_marker_abandons(monkeypatch):
    """★ 进度没了、但追问还挂着 → 放弃这段对话，且绝不重跑诊断。"""
    redis = _StubRedis()
    monkeypatch.setattr(worker_tools, "get_checkpointer_redis", lambda: redis)
    # 模拟：进度因 TTL/淘汰已经消失，只剩挂起标记
    await TriageSessionStore(redis).mark_pending("7:s1")

    called = []

    async def _fail_run_triage(**kw):
        called.append(kw)
        raise AssertionError("过期会话不该跑诊断 —— 重放会把历史回答再处理一遍")

    monkeypatch.setattr(worker_tools, "run_triage", _fail_run_triage)

    reply = await worker_tools._run_triage_turn("我答的是 A1", "7", "s1", "engineer", "7:s1")

    assert "过期" in reply
    assert called == []
    # 标记必须清掉，否则用户下一次描述又会被判成「过期会话」
    assert redis.data == {}


async def test_missing_progress_without_marker_starts_fresh(monkeypatch):
    """全新会话（无标记）照旧跑首轮 —— 放弃分支不能把正常首轮也吞掉。"""
    redis = _StubRedis()
    monkeypatch.setattr(worker_tools, "get_checkpointer_redis", lambda: redis)
    monkeypatch.setattr(worker_tools, "TriageAgent", _FakeAgent)

    calls = []

    async def _fake_run_triage(**kw):
        calls.append(kw)
        return "首轮结论", TriageState(phase=TriagePhase.CONCLUDE)

    monkeypatch.setattr(worker_tools, "run_triage", _fake_run_triage)

    reply = await worker_tools._run_triage_turn("车机黑屏", "7", "s1", "engineer", "7:s1")

    assert reply == "首轮结论"
    assert len(calls) == 1
    assert calls[0]["existing_state"] is None  # 确实是按首轮跑的
