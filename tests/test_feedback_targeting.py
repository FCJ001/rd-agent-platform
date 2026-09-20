# ============================================================
# 反馈回写必须只作用于一行
#
# 锁住：同一 session_id 下会有多行结论（工具每调一次写一行，同一会话里
# 重复诊断是正常行为）。按 session_id 整批更新时，点一次「采纳」会把该会话
# 所有历史结论一起标记，准确率统计失真；分诊那条还会顺带把图谱回流用的
# 现象/根因取自任一行（原来的 fetchone 没有排序，取哪行不确定）。
#
# 用假 DB 会话记录 SQL 与参数：本文件测的是「定位到哪一行」，不是 SQL 执行
# 本身（那是集成测试的活），所以不进外部服务。
# ============================================================

import pytest

from src.api.routers import feedback as feedback_mod
from src.api.routers import triage as triage_mod
from src.core.deps import UserContext

USER = UserContext(user_id="7", session_id="s1", role="engineer")


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def rowcount(self):
        return 1 if self._row else 0


class _FakeDB:
    """记录每条 SQL 与参数，按预设返回同一行（够用于定位逻辑）。"""

    def __init__(self, row):
        self.row = row
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, stmt, params=None):
        self.calls.append((" ".join(str(stmt).split()), params or {}))
        return _FakeResult(self.row)

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def sql_like(self, needle: str) -> bool:
        return any(needle in sql for sql, _ in self.calls)

    def update_sql(self) -> str:
        """返回唯一的 UPDATE 语句（没有则空串），用于检查回写条件。"""
        for sql, _ in self.calls:
            if sql.startswith("UPDATE"):
                return sql
        return ""

    def params_of(self, needle: str) -> dict:
        for sql, params in self.calls:
            if needle in sql:
                return params
        return {}


def _patch_db(monkeypatch, db: _FakeDB):
    # 两个模块都是在函数内 `from src.infra.db import AsyncSessionLocal`，
    # 所以打在源模块上即可生效
    monkeypatch.setattr("src.infra.db.AsyncSessionLocal", lambda: db)


# ── 影响分析 / 报告解读 ─────────────────────────────────────────────────

async def test_impact_feedback_targets_newest_row_by_default(monkeypatch):
    """不传 record_id 时取最新一行，而不是把整个会话刷一遍。"""
    db = _FakeDB(row=(17,))
    _patch_db(monkeypatch, db)

    resp = await feedback_mod.submit_impact_feedback(
        feedback_mod.FeedbackRequest(session_id="s1", adopted=True), USER
    )

    assert resp.data.updated is True
    assert resp.data.record_id == 17
    # 定位语句必须限定单行 + 带归属（session_id + user_id 都在这里）
    assert db.sql_like("ORDER BY id DESC LIMIT 1")
    assert db.params_of("ORDER BY id DESC LIMIT 1") == {"session_id": "s1", "user_id": 7}
    # 回写只按主键 —— 不能再出现「按 session_id 批量更新」的写法
    update_sql = db.update_sql()
    assert "WHERE id = :target_id" in update_sql
    assert "session_id" not in update_sql


async def test_impact_feedback_uses_explicit_record_id(monkeypatch):
    """传了 record_id 就精确回写那一行。"""
    db = _FakeDB(row=(42,))
    _patch_db(monkeypatch, db)

    resp = await feedback_mod.submit_impact_feedback(
        feedback_mod.FeedbackRequest(session_id="s1", adopted=False, record_id=42), USER
    )

    assert resp.data.record_id == 42
    assert db.params_of("id = :record_id")["record_id"] == 42


async def test_impact_feedback_rejects_when_no_row(monkeypatch):
    """无匹配行（他人会话 / 存量 NULL 行）→ 不更新，也不去写库。"""
    db = _FakeDB(row=None)
    _patch_db(monkeypatch, db)

    resp = await feedback_mod.submit_impact_feedback(
        feedback_mod.FeedbackRequest(session_id="s1", adopted=True), USER
    )

    assert resp.data.updated is False
    assert resp.data.record_id is None
    assert not db.sql_like("UPDATE")


# ── 分诊反馈 ────────────────────────────────────────────────────────────

async def test_triage_feedback_single_row_and_matching_graph_signal(monkeypatch):
    """★ 更新与图谱回流必须用同一行、且只有一行。

    原来 SELECT（无排序的 fetchone）与 UPDATE（按 session_id）各取各的：
    更新会刷整个会话，图谱回流的现象/根因可能取自另一条结论。
    """
    db = _FakeDB(row=(88, '["车机黑屏"]', "RC-EV-0012"))
    _patch_db(monkeypatch, db)

    seen = {}

    async def _fake_reinforce(confirmed_phenomena, primary_cause_code, session_id=""):
        seen["phenomena"] = confirmed_phenomena
        seen["cause"] = primary_cause_code
        return True

    # 避免真的去连 Neo4j
    monkeypatch.setattr(
        "src.agents.triage.feedback_loop.reinforce_graph_on_adopted", _fake_reinforce
    )

    resp = await triage_mod.submit_feedback(
        triage_mod.FeedbackRequest(session_id="7:s1", adopted=True), USER
    )

    assert resp.data.updated is True
    assert resp.data.record_id == 88
    assert db.params_of("ORDER BY id DESC LIMIT 1")["session_id"] == "7:s1"
    update_params = db.params_of("UPDATE ai_triage_results")
    assert update_params["target_id"] == 88          # 只按主键更新
    assert update_params["session_id"] == "7:s1"     # 且带了会话限定
    assert "WHERE id = :target_id AND session_id = :session_id" in db.update_sql()
    # 图谱回流拿到的是同一行的信号
    assert seen == {"phenomena": ["车机黑屏"], "cause": "RC-EV-0012"}


async def test_triage_feedback_uses_explicit_record_id(monkeypatch):
    db = _FakeDB(row=(91, "[]", "RC-1"))
    _patch_db(monkeypatch, db)

    async def _noop(*a, **kw):
        return True

    monkeypatch.setattr(
        "src.agents.triage.feedback_loop.reinforce_graph_on_adopted", _noop
    )

    resp = await triage_mod.submit_feedback(
        triage_mod.FeedbackRequest(session_id="7:s1", adopted=True, record_id=91), USER
    )

    assert resp.data.record_id == 91
    params = db.params_of("id = :record_id")
    assert params["record_id"] == 91
    assert params["session_id"] == "7:s1"  # 拿别人的 id 来试也落不到那行


async def test_triage_feedback_rejects_other_users_session():
    """所有权校验仍然生效（本用例不碰 DB：校验在查库之前）。"""
    from src.core.exceptions import ERR_PERMISSION_DENIED, BizException

    with pytest.raises(BizException) as ei:
        await triage_mod.submit_feedback(
            triage_mod.FeedbackRequest(session_id="9:s1", adopted=True), USER
        )
    assert ei.value.code == ERR_PERMISSION_DENIED
