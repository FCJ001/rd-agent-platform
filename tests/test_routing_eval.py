# ============================================================
# L0 路由评测 —— CI 静态门禁 + 判定语义单测
#
# 两层分工（与 eval 的 评分级/实况 双轨同构）：
#   静态门禁（本文件，CI 必跑）：
#     1. routing_cases.json 的 schema 与工具名有效性
#     2. supervisor prompt 必须描述到每一个注册工具（路由漂移护栏）
#     3. 切换词与退出词表不相交（交互协议无歧义）
#   实况判定（eval/run_routing_eval.py，发版前跑）：judge() 的语义
#     在这里用假数据锁死，runner 挂了先看这里。
# ============================================================

import json
from pathlib import Path

from src.agents.sub_agents import SWITCH_COMMANDS
from src.agents.supervisor_agent import SUPERVISOR_SYSTEM_PROMPT, get_supervisor_toolset
from src.agents.triage.session_store import TRIAGE_EXIT_COMMANDS
from eval.run_routing_eval import PREAMBLE_TOOLS, judge

CASES_FILE = Path(__file__).parent.parent / "eval" / "cases" / "routing_cases.json"


def _tool_names() -> set[str]:
    return {t.name for t in get_supervisor_toolset()}


def _cases() -> list[dict]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]


# ── 静态门禁 1：case 文件 schema ─────────────────────────────────────────

def test_routing_cases_schema_valid():
    cases = _cases()
    assert len(cases) >= 15, "L0 用例太少，路由覆盖不足"
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case id 重复"
    names = _tool_names()
    for c in cases:
        assert c.get("message"), c["id"]
        assert c.get("expect_tools") or c.get("expect_none_business"), \
            f"{c['id']} 缺断言（expect_tools 或 expect_none_business）"
        for t in c.get("expect_tools") or []:
            assert t in names, f"{c['id']} 引用了不存在的工具 {t}"
        for t in c.get("expect_not") or []:
            assert t in names, f"{c['id']} 引用了不存在的工具 {t}"


def test_routing_cases_cover_all_business_tools():
    """每个业务工具至少被一个正向用例锚定 —— 加新工具忘了配用例这里红。"""
    cases = _cases()
    covered = {t for c in cases for t in c.get("expect_tools") or []}
    uncovered = _tool_names() - covered - PREAMBLE_TOOLS
    assert not uncovered, f"未被任何 L0 用例覆盖的工具: {uncovered}"


# ── 静态门禁 2：prompt 与工具集一致（路由漂移护栏）──────────────────────

def test_prompt_documents_every_tool():
    """system prompt 必须提到每个注册工具名 —— 漏描述 = 模型永远调不到它。"""
    for name in _tool_names():
        assert name in SUPERVISOR_SYSTEM_PROMPT, f"工具 {name} 未在 prompt 中描述"


def test_switch_and_exit_commands_disjoint():
    """切换词与退出词不相交：同一个词不能既退出又切换（交互协议无歧义）。"""
    assert SWITCH_COMMANDS & TRIAGE_EXIT_COMMANDS == set()


# ── 判定语义（judge 纯函数）──────────────────────────────────────────────

def test_judge_pass_on_any_of_and_chain_allowed():
    """expect_tools 任一命中即过；合法链式调用（历史→判重→分诊）不算错。"""
    case = {"id": "x", "message": "黑屏", "expect_tools": ["call_triage_agent"]}
    assert judge(case, ["search_memory", "search_past_diagnoses",
                        "call_dedup_check", "call_triage_agent"]) == []


def test_judge_fail_when_key_tool_missing():
    case = {"id": "x", "message": "黑屏", "expect_tools": ["call_triage_agent"]}
    errors = judge(case, ["search_memory", "call_knowledge_agent"])
    assert errors and "期望命中" in errors[0]


def test_judge_fail_on_forbidden_tool():
    case = {"id": "x", "message": "统计", "expect_tools": ["call_operation_agent"],
            "expect_not": ["call_knowledge_agent"]}
    errors = judge(case, ["call_operation_agent", "call_knowledge_agent"])
    assert any("禁止调用" in e for e in errors)


def test_judge_memory_preamble_not_business():
    """原则 1 的记忆前奏不算业务调用（否则所有 case 都被 search_memory 污染）。"""
    case = {"id": "x", "message": "你好", "expect_none_business": True}
    assert judge(case, ["search_memory"]) == []


def test_judge_none_business_detects_real_call():
    case = {"id": "x", "message": "你好", "expect_none_business": True}
    assert judge(case, ["search_memory", "call_knowledge_agent"]) != []
