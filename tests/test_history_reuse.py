# ============================================================
# 历史诊断复用的单元测试
#
# 锁住三类契约：
#   1. 业务线推导：从根因编码前缀取归属，且不误吞年份型编号
#   2. 角色门禁：项目维度（跨用户）对客户不开放，降级为个人维度且必须留痕
#   3. 两级可信度：已采纳优先；降级到未复核时，返回文本必须带未复核标记
#
# ★ 不碰数据库：_query_sync 被替换成假实现，只验证「发出去的条件」与
#   「回来的东西怎么被解释」。
# ============================================================

import pytest

from src.agents.triage import history as hist
from src.utils.business_line import line_from_code, resolve_business_line


# ── 1. 业务线推导 ──────────────────────────────────────────────

@pytest.mark.parametrize("code,expected", [
    ("RC-EV-0012", "ev"),
    ("RC-IA-0001", "ia"),
    ("PH-EV-001", "ev"),
    ("REQ-EV-0042", "ev"),
    ("RC-EV-0012 ", "ev"),      # 容忍前后空白（编码可能从表单/日志带进来）
    ("ISS-2025-00001", ""),     # 第二段是年份 → 不推导
    ("CR-2025-0088", ""),
    ("P0A7F", ""),              # DTC 码没有线前缀
    ("", ""),
    (None, ""),
])
def test_line_from_code(code, expected):
    assert line_from_code(code) == expected


def test_resolve_prefers_explicit_value():
    assert resolve_business_line("ia", "RC-EV-0001") == "ia"


def test_resolve_falls_back_to_code_prefix():
    """customer 会话的常态：没有所属线，靠编码前缀兜底。"""
    assert resolve_business_line("", "RC-IA-0001") == "ia"
    assert resolve_business_line(None, "RC-IA-0001") == "ia"


def test_resolve_rejects_unknown_line():
    """推导出的值必须在配置白名单里，否则宁可无归属。"""
    assert resolve_business_line(None, "RC-XX-0001") == ""


# ── 2. 关键词切分 ──────────────────────────────────────────────

def test_keywords_split_and_dedupe():
    kws = hist._keywords("黑屏，无法唤醒、黑屏 掉电")
    assert kws == ["黑屏", "无法唤醒", "掉电"]


def test_keywords_drop_short_tokens():
    assert hist._keywords("a, 黑屏") == ["黑屏"]


def test_keywords_cap():
    raw = ",".join(f"现象{i}" for i in range(20))
    assert len(hist._keywords(raw)) == hist.MAX_KEYWORDS


# ── 3. 门禁与两级回退 ──────────────────────────────────────────

class _Recorder:
    """假 _query_sync：记录调用条件，按 level 返回预设结果。"""

    def __init__(self, by_level=None):
        self.calls = []
        self.by_level = by_level or {}

    def __call__(self, keywords, user_id, business_line, scope, level, limit):
        self.calls.append({
            "keywords": keywords, "user_id": user_id,
            "business_line": business_line, "scope": scope, "level": level,
        })
        return self.by_level.get(level, [])


_ADOPTED_ROW = {
    "cause_code": "RC-EV-0012", "confidence": 0.82,
    "confirmed_phenomena": ["续航异常衰减"], "total_rounds": 2,
    "created_at": "2026-09-01", "adopted": True,
    "issue_no": "ISS-2025-00007", "issue_title": "静置掉电", "issue_status": "closed",
}


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(hist, "_query_sync", rec)
    return rec


@pytest.mark.asyncio
async def test_customer_cannot_query_project_scope(recorder):
    """客户查项目维度 → 降级为个人维度，并留下降级说明。"""
    recorder.by_level[hist._LEVEL_ADOPTED] = [_ADOPTED_ROW]
    result = await hist.query_past_diagnoses(
        "掉电", role="customer", user_id="13", business_line="", scope=hist.SCOPE_LINE,
    )
    assert result["scope"] == hist.SCOPE_MINE
    assert "内部角色" in result["note"]
    assert all(c["scope"] == hist.SCOPE_MINE for c in recorder.calls)


@pytest.mark.asyncio
async def test_line_scope_without_business_line_downgrades(recorder):
    recorder.by_level[hist._LEVEL_ADOPTED] = [_ADOPTED_ROW]
    result = await hist.query_past_diagnoses(
        "掉电", role="engineer", user_id="1", business_line="", scope=hist.SCOPE_LINE,
    )
    assert result["scope"] == hist.SCOPE_MINE
    assert "业务线" in result["note"]


@pytest.mark.asyncio
async def test_engineer_can_query_project_scope(recorder):
    recorder.by_level[hist._LEVEL_ADOPTED] = [_ADOPTED_ROW]
    result = await hist.query_past_diagnoses(
        "掉电", role="engineer", user_id="1", business_line="ev", scope=hist.SCOPE_LINE,
    )
    assert result["scope"] == hist.SCOPE_LINE
    assert result["note"] == ""
    assert recorder.calls[0]["business_line"] == "ev"
    assert recorder.calls[0]["scope"] == hist.SCOPE_LINE


@pytest.mark.asyncio
async def test_mine_scope_requires_user_id(recorder):
    """没有身份就不查个人维度（自动分诊场景）—— fail-closed。"""
    result = await hist.query_past_diagnoses("掉电", role="customer", user_id="")
    assert result["items"] == []
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_adopted_first_no_fallback_when_hit(recorder):
    recorder.by_level[hist._LEVEL_ADOPTED] = [_ADOPTED_ROW]
    result = await hist.query_past_diagnoses("掉电", role="customer", user_id="13")
    assert result["level"] == hist._LEVEL_ADOPTED
    assert [c["level"] for c in recorder.calls] == [hist._LEVEL_ADOPTED]


@pytest.mark.asyncio
async def test_falls_back_to_unreviewed_when_no_adopted(recorder):
    """反馈闭环没人点采纳时，不能让功能空转 —— 降级但必须标明未复核。"""
    recorder.by_level[hist._LEVEL_UNREVIEWED] = [dict(_ADOPTED_ROW, adopted=None)]
    result = await hist.query_past_diagnoses("掉电", role="customer", user_id="13")
    assert result["level"] == hist._LEVEL_UNREVIEWED
    assert [c["level"] for c in recorder.calls] == [
        hist._LEVEL_ADOPTED, hist._LEVEL_UNREVIEWED
    ]
    text = hist.format_past_diagnoses(result)
    assert "尚未经人工复核" in text


@pytest.mark.asyncio
async def test_no_keywords_returns_note(recorder):
    result = await hist.query_past_diagnoses(" ， ", role="customer", user_id="13")
    assert result["items"] == []
    assert "关键词" in result["note"]
    assert recorder.calls == []


# ── 4. 文本呈现 ────────────────────────────────────────────────

def test_format_marks_scope_and_cause():
    text = hist.format_past_diagnoses({
        "items": [_ADOPTED_ROW], "scope": hist.SCOPE_LINE,
        "level": hist._LEVEL_ADOPTED, "note": "",
    })
    assert "本业务线" in text
    assert "RC-EV-0012" in text
    assert "ISS-2025-00007" in text
    # 已采纳的记录不该出现未复核警告
    assert "尚未经人工复核" not in text


def test_format_empty():
    assert "未找到" in hist.format_past_diagnoses({"items": []})
