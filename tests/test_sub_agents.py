# ============================================================
# 子智能体注册表测试（sub_agents.py）
#
# 锁住的契约：triage 是第一个按「子图 + handoff + done」形态接入的
# 子智能体；以后接入新智能体在 SUB_AGENTS 登记一行，chat 层的退出
# 判定 / 旁路工具摘除 / 闸门口径自动生效。未登记类型 fail-safe：
#   不当退出、不摘工具、默认过闸门（宁可保守）。
# ============================================================

from src.agents.sub_agents import (
    SUB_AGENTS, entry_tool_for, is_gated, is_task_exit, spec_for,
)
from src.agents.workers.side_assistant import _side_tools


# ── 注册表查询 ───────────────────────────────────────────────────────────

def test_registry_has_triage():
    spec = spec_for("triage_followup")
    assert spec is not None
    assert spec.entry_tool == "call_triage_agent"
    assert spec.gated is True


def test_unknown_type_fail_safe():
    """未登记类型：查不到规格、不是退出、不摘工具、默认过闸门。"""
    assert spec_for("nonexistent_followup") is None
    assert is_task_exit("nonexistent_followup", "退出诊断") is False
    assert entry_tool_for("nonexistent_followup") is None
    assert is_gated("nonexistent_followup") is True


# ── 退出判定按 type 路由 ────────────────────────────────────────────────

def test_exit_routes_by_interrupt_type():
    assert is_task_exit("triage_followup", "退出诊断") is True
    assert is_task_exit("triage_followup", "充电时跳闸") is False
    # 同一条消息对未登记类型不是退出（不能误杀未知流程的会话）
    assert is_task_exit("other_followup", "退出诊断") is False


# ── 旁路助手工具摘除 ────────────────────────────────────────────────────

def test_side_tools_excludes_given_entry_tool():
    """摘除指定入口工具；其余工具保留。"""
    tools = _side_tools(exclude_tool="call_triage_agent")
    names = [t.name for t in tools]
    assert "call_triage_agent" not in names
    assert "call_knowledge_agent" in names
    assert "search_past_diagnoses" in names


def test_side_tools_default_keeps_all():
    """无活跃任务类型时不摘（旁路仅在挂起时触发，这里是防御性默认）。"""
    names = [t.name for t in _side_tools()]
    assert "call_triage_agent" in names


# ── 注册表自身的一致性（接入新智能体时这条会提醒你配齐）──────────────

def test_registry_specs_wellformed():
    """每个登记项字段自洽（接入新智能体时这条提醒配齐必填项）。"""
    for itype, spec in SUB_AGENTS.items():
        assert spec.interrupt_type == itype
        assert callable(spec.exit_checker)
        assert spec.entry_tool
