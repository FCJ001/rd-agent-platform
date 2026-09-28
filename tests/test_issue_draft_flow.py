# ============================================================
# 建单草稿两段式（HITL）测试
#
# 锁住的契约：
#   1. call_create_issue 只落草稿（不建单），参数校验先于落库
#   2. call_confirm_issue：属主校验（IDOR）/ 幂等 / 惰性过期 / 诚实失败
#   3. 作用域来自用户上下文，LLM 传值只是兜底
#
# ★ 不碰数据库：_save_draft/_load_draft/_mark_draft 被 monkeypatch。
# ============================================================

import types
from datetime import datetime, timedelta, timezone

import pytest

import src.agents.tools.platform_tools as pt
from src.core.deps import UserContext


class _FakeRuntime:
    def __init__(self, ctx: UserContext):
        self.context = ctx


def _ctx(uid: str = "7", line: str | None = "ev") -> UserContext:
    return UserContext(user_id=uid, session_id="s1", role="engineer", business_line=line)


def _draft(
    draft_id: int = 101, user_id: int = 7, status: str = "pending",
    age_seconds: float = 60.0,
):
    return types.SimpleNamespace(
        id=draft_id, user_id=user_id, session_id="s1",
        title="中控屏黑屏", description="高速上出现两次", severity="normal",
        business_line="ev", source="engineer", owner_domain_id=None,
        status=status, issue_no=None,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )


@pytest.fixture()
def recorder(monkeypatch):
    """记录草稿落库/状态迁移的假 DB 层。"""
    rec = {"saved": [], "marked": [], "draft": _draft()}

    async def fake_save(payload):
        rec["saved"].append(payload)
        return 101

    async def fake_load(draft_id):
        return rec["draft"] if draft_id == 101 else None

    async def fake_mark(draft_id, status, issue_no=None):
        rec["marked"].append((draft_id, status, issue_no))

    monkeypatch.setattr(pt, "_save_draft", fake_save)
    monkeypatch.setattr(pt, "_load_draft", fake_load)
    monkeypatch.setattr(pt, "_mark_draft", fake_mark)
    return rec


async def _call(tool, **kwargs):
    return await tool.coroutine(**kwargs)


# ── 1. 草稿生成 ──────────────────────────────────────────────────────────

async def test_create_only_saves_draft(recorder):
    """call_create_issue 只落草稿、不建单，回复引导确认。"""
    reply = await _call(
        pt.call_create_issue,
        title="中控屏黑屏", description="高速上出现两次",
        severity="normal", business_line="ia",   # 故意传错线：上下文 ev 优先
        runtime=_FakeRuntime(_ctx()),
    )
    assert len(recorder["saved"]) == 1
    payload = recorder["saved"][0]
    assert payload["business_line"] == "ev"      # 作用域来自用户上下文
    assert payload["user_id"] == 7
    assert "编号 101" in reply
    assert "call_confirm_issue" in reply         # 引导下一步（确认）
    assert "确认" in reply


async def test_create_validates_before_save(recorder):
    """非法 severity / business_line 在落库前被拒。"""
    r1 = await _call(
        pt.call_create_issue, title="t", description="d",
        severity="超高", business_line="ev", runtime=_FakeRuntime(_ctx()),
    )
    assert "severity 取值不合法" in r1
    assert recorder["saved"] == []               # 没有落库

    r2 = await _call(
        pt.call_create_issue, title="t", description="d",
        severity="normal", business_line="xx",
        runtime=_FakeRuntime(_ctx(uid="7", line=None)),  # 上下文无线 → 用 LLM 传的非法值
    )
    assert "business_line 取值不合法" in r2
    assert recorder["saved"] == []


# ── 2. 确认提交 ──────────────────────────────────────────────────────────

async def test_confirm_happy_path(recorder, monkeypatch):
    """本人 + pending + 未过期 + 平台已接 → 提交并关草稿。"""
    monkeypatch.setattr(pt, "_platform_ready", lambda: True)
    reply = await _call(pt.call_confirm_issue, draft_id=101, runtime=_FakeRuntime(_ctx()))
    assert "已确认并提交" in reply
    assert recorder["marked"] == [(101, "submitted", None)]


async def test_confirm_idempotent(recorder):
    """已提交的草稿再次确认 → 拒绝且不改状态。"""
    recorder["draft"] = _draft(status="submitted", age_seconds=60)
    reply = await _call(pt.call_confirm_issue, draft_id=101, runtime=_FakeRuntime(_ctx()))
    assert "已提交过" in reply
    assert recorder["marked"] == []


async def test_confirm_rejects_other_owner(recorder):
    """IDOR：别人的草稿 → 无权操作，不迁移状态。"""
    recorder["draft"] = _draft(user_id=99)
    reply = await _call(pt.call_confirm_issue, draft_id=101, runtime=_FakeRuntime(_ctx(uid="7")))
    assert "无权操作" in reply
    assert recorder["marked"] == []


async def test_confirm_expired_lazily(recorder):
    """超时未确认 → 惰性置 expired（无需定时任务）。"""
    recorder["draft"] = _draft(age_seconds=pt.settings.ISSUE_DRAFT_TTL_SECONDS + 60)
    reply = await _call(pt.call_confirm_issue, draft_id=101, runtime=_FakeRuntime(_ctx()))
    assert "已过期" in reply
    assert recorder["marked"] == [(101, "expired", None)]


async def test_confirm_already_expired(recorder):
    recorder["draft"] = _draft(status="expired")
    reply = await _call(pt.call_confirm_issue, draft_id=101, runtime=_FakeRuntime(_ctx()))
    assert "已过期" in reply
    assert recorder["marked"] == []


async def test_confirm_nonexistent(recorder):
    reply = await _call(pt.call_confirm_issue, draft_id=999, runtime=_FakeRuntime(_ctx()))
    assert "不存在" in reply


async def test_confirm_dev_placeholder_honest_failure(recorder, monkeypatch):
    """平台未接入：诚实告知没建单 + 关闭草稿防重复确认。"""
    monkeypatch.setattr(pt, "_platform_ready", lambda: False)
    reply = await _call(pt.call_confirm_issue, draft_id=101, runtime=_FakeRuntime(_ctx()))
    assert "没有**真正**创建问题单" in reply or "**没有**真正创建问题单" in reply
    assert recorder["marked"] == [(101, "submitted", None)]  # 草稿关闭防重放


def test_platform_tools_registry_updated():
    """确认 call_confirm_issue 已注册进平台工具表（Supervisor 可见）。"""
    names = [t.name for t in pt.PLATFORM_TOOLS]
    assert "call_create_issue" in names
    assert "call_confirm_issue" in names
