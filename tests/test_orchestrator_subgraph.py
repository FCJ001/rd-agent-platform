# ============================================================
# P1 单图化端到端测试：编排图（Supervisor ⇄ 分诊会话子图）
#
# 锁住的核心性质（相对旧 agent-as-tool 路径的收益）：
#   1. 多轮追问 interrupt 在子图内挂起/恢复，chat 层快照契约
#      （payload 形状、_pending_question）零改动
#   2. ★ 无重放：每轮工作恰好执行一次 —— resume 不重跑上一轮的
#      extract/ask 的 LLM 调用（旧机制靠外部 store + 快进空转实现，
#      新机制靠节点级 checkpoint 天然获得）
#   3. 退出指令 / 安全关键词终止路径
#   4. 新诊断状态隔离：第二次 handoff 的子图状态全新（round 归零）
#   5. 结论文本经共享消息历史回到 supervisor 收尾（结论不丢上下文）
#
# 全部纯内存（MemorySaver + 假模型 + monkeypatch 图模块的 IO 函数）。
# 机制前提见 scripts/probe_orchestrator_mechanics.py。
# ============================================================

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import src.agents.triage.graph as graph_mod
import src.agents.triage.session_graph as sg
from src.agents.orchestrator import build_orchestrator_agent
from src.agents.triage.graph import TriageDeps
from src.agents.triage.state import CandidateCause
from src.core.deps import UserContext
from tests.test_triage_hitl import _make_fake_model

QUESTION = "是否伴随死机？"
CONCLUSION = "诊断结论：中控主机软件异常，建议升级软件版本。"
WRAP_UP = "诊断已完成，以上是结论和操作选项。"


# ── 假依赖 ────────────────────────────────────────────────────────────────

class _ScriptedLLM:
    """按脚本顺序返回 content 的假 LLM，计次供无重放断言。"""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        i = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self.responses[i])


class _FakeDbCtx:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


def _fake_db_factory():
    return _FakeDbCtx()


def _candidate() -> CandidateCause:
    return CandidateCause(
        code="RC-IA-0001", name="中控主机软件异常", domain="智能座舱",
        business_line="ia", confidence=0.55, base_confidence=0.5,
        matched_phenomena=["中控屏黑屏"], all_phenomena=["中控屏黑屏", "伴随死机"],
    )


@pytest.fixture(autouse=True)
def bare_supervisor(monkeypatch):
    """裸装配 supervisor（去掉 Summarization/Repair 中间件）。

    本文件验证的是【图编排】（handoff/interrupt/无重放/状态隔离），
    不是中间件行为；SummarizationMiddleware 在 6 条消息后会把 fake 模型
    的脚本位吃掉（摘要调用也走同一个模型），干扰调用计数断言。
    """
    from langchain.agents import create_agent

    import src.agents.orchestrator as orch_mod
    from src.agents.supervisor_agent import SUPERVISOR_SYSTEM_PROMPT

    def simple_assemble(llm, tools, checkpointer=None, store=None):
        return create_agent(
            model=llm, tools=tools,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            context_schema=UserContext,
        )

    monkeypatch.setattr(orch_mod, "assemble_supervisor", simple_assemble)


@pytest.fixture()
def triage_io(monkeypatch):
    """monkeypatch 分诊节点的全部外部 IO（DB/Neo4j/词表/收敛判定）。

    返回计数器，供「无重放」断言。
    """
    counts = {"convergence": 0, "causes": 0}

    async def fake_match(db, names, line=""):
        return [{"name": n} for n in names]

    def fake_query(phenom_names, business_line="", top_k=10):
        counts["causes"] += 1
        return [_candidate()]

    async def fake_enrich(candidates):
        return candidates

    def fake_apply_weights(candidates, *a, **kw):
        return candidates

    def fake_convergence(candidates, round_no):
        # 按轮次判定（每场诊断各自生效）：round 0 追问，round 1+ 收敛
        counts["convergence"] += 1
        return (round_no >= 1, False)

    async def fake_save(db, payload):
        return 1

    async def fake_vocab(business_line=""):
        return "中控屏黑屏（别名：屏幕不亮）、伴随死机"

    monkeypatch.setattr(graph_mod, "match_phenomena_by_names", fake_match)
    monkeypatch.setattr(graph_mod, "query_causes_by_phenomena", fake_query)
    monkeypatch.setattr(graph_mod, "enrich_cause_details_async", fake_enrich)
    monkeypatch.setattr(graph_mod, "apply_context_weights", fake_apply_weights)
    monkeypatch.setattr(graph_mod, "check_convergence", fake_convergence)
    monkeypatch.setattr(graph_mod, "save_triage_result", fake_save)
    monkeypatch.setattr(sg, "load_phenomenon_vocabulary", fake_vocab)
    return counts


async def _build(triage_io, json_script=None, chat_script=None, supervisor_script=None):
    """构建注入假依赖的编排图。"""
    llm_json = _ScriptedLLM(json_script or [
        '{"phenomena": ["中控屏黑屏"], "dtc_codes": []}',           # r1 extract
        '{"confirmed": ["伴随死机"], "denied": [], "dtc_codes": []}',  # r2 parse
        '{"phenomena": [], "dtc_codes": []}',                       # r2 extract（增量）
    ])
    llm_chat = _ScriptedLLM(chat_script or [QUESTION, CONCLUSION])
    deps = TriageDeps(llm_json=llm_json, llm_chat=llm_chat, db_session_factory=_fake_db_factory)

    supervisor = _make_fake_model(supervisor_script or [
        AIMessage(content="", tool_calls=[
            {"name": "call_triage_agent", "args": {"message": "车机黑屏"}, "id": "c1"}
        ]),
        AIMessage(content=WRAP_UP),
    ])

    agent = await build_orchestrator_agent(
        llm=supervisor, triage_deps=deps, checkpointer=MemorySaver(),
    )
    return agent, llm_json, llm_chat


def _ctx() -> UserContext:
    return UserContext(user_id="u9", session_id="s9", role="engineer", business_line="ia")


def _texts(result) -> list[str]:
    return [m.content for m in result["messages"] if hasattr(m, "content")]


# ── 1. 多轮 e2e + 无重放 ─────────────────────────────────────────────────

async def test_multiturn_e2e_no_replay(triage_io):
    """症状 → handoff → 子图挂起追问 → resume 补答 → 收敛 → supervisor 收尾。

    ★ 无重放断言：llm_json 恰好 3 次（r1 extract + r2 parse + r2 extract）、
    llm_chat 恰好 2 次（追问 + 结论）—— resume 没有重跑上一轮的任何 LLM 调用。
    """
    agent, llm_json, llm_chat = await _build(triage_io)
    cfg = {"configurable": {"thread_id": "t1"}}

    # 首回合：supervisor 路由 → handoff → 子图挂起在追问
    r1 = await agent.ainvoke({"messages": [HumanMessage(content="车机黑屏")]}, config=cfg, context=_ctx())
    intr = r1.get("__interrupt__")
    assert intr, "应在追问处挂起"
    assert intr[0].value["type"] == "triage_followup"       # chat 层快照契约不变
    assert intr[0].value["question"] == QUESTION
    assert intr[0].value["round"] == 1

    # 快照路由判据（chat._pending_question 同款读取）
    snap = await agent.aget_state(cfg)
    assert any(t.interrupts for t in snap.tasks)

    # resume：补答 → 收敛 → save_record → triage_done → supervisor 收尾
    r2 = await agent.ainvoke(Command(resume="是的，每次黑屏后都会死机"), config=cfg, context=_ctx())
    assert not r2.get("__interrupt__")
    assert r2["messages"][-1].content == WRAP_UP             # supervisor 收尾
    # 结论在共享历史里（node_conclude 会在结论后追加操作选项块，用子串匹配）
    assert any(CONCLUSION in t for t in _texts(r2))
    assert not r2.get("active_task")                         # 粘性标记已清

    # ★ 无重放：每轮工作恰好一次
    assert llm_json.calls == 3, f"json LLM 应调用 3 次，实际 {llm_json.calls}"
    assert llm_chat.calls == 2, f"chat LLM 应调用 2 次，实际 {llm_chat.calls}"
    assert triage_io["causes"] == 2                          # 每轮检索一次


# ── 2. 退出指令 ──────────────────────────────────────────────────────────

async def test_exit_command_aborts_diagnosis(triage_io):
    """追问轮回复退出指令 → 诊断作废，控制权回 supervisor。"""
    agent, llm_json, _ = await _build(triage_io)
    cfg = {"configurable": {"thread_id": "t2"}}

    await agent.ainvoke({"messages": [HumanMessage(content="车机黑屏")]}, config=cfg, context=_ctx())
    r2 = await agent.ainvoke(Command(resume="退出诊断"), config=cfg, context=_ctx())

    assert not r2.get("__interrupt__")
    assert any("退出" in t and "作废" in t for t in _texts(r2))
    assert not r2.get("active_task")
    # 退出轮没有进入 parse/extract（json LLM 只剩首轮 extract 的 1 次）
    assert llm_json.calls == 1


# ── 3. 安全关键词终止 ────────────────────────────────────────────────────

async def test_safety_keyword_terminates(triage_io):
    """安全词命中 → 立即警示终止，不追问、不检索。"""
    agent, llm_json, _ = await _build(
        triage_io,
        json_script=['{"phenomena": ["制动失效"], "dtc_codes": []}'],
        chat_script=[CONCLUSION],
    )
    cfg = {"configurable": {"thread_id": "t3"}}

    r = await agent.ainvoke(
        {"messages": [HumanMessage(content="高速上制动失灵了")]}, config=cfg, context=_ctx(),
    )
    assert not r.get("__interrupt__")
    assert any("安全警示" in t for t in _texts(r))
    assert triage_io["causes"] == 0                          # 安全拦截先于检索
    assert llm_json.calls == 1                               # 只有首轮 extract


# ── 4. 新诊断状态隔离 ────────────────────────────────────────────────────

async def test_second_diagnosis_fresh_state(triage_io):
    """第一场收敛后再次 handoff：子图状态全新（追问 round 重新从 1 计）。"""
    agent, _, _ = await _build(triage_io, supervisor_script=[
        AIMessage(content="", tool_calls=[
            {"name": "call_triage_agent", "args": {"message": "车机黑屏"}, "id": "c1"}
        ]),
        AIMessage(content=WRAP_UP),
        AIMessage(content="", tool_calls=[
            {"name": "call_triage_agent", "args": {"message": "空调异响"}, "id": "c2"}
        ]),
        AIMessage(content="第二个诊断也完成了。"),
    ])
    cfg = {"configurable": {"thread_id": "t4"}}

    # 第一场：挂起 → 补答 → 收敛
    r1 = await agent.ainvoke({"messages": [HumanMessage(content="车机黑屏")]}, config=cfg, context=_ctx())
    first_round = r1["__interrupt__"][0].value["round"]
    await agent.ainvoke(Command(resume="是的会死机"), config=cfg, context=_ctx())

    # 第二场：新症状 → supervisor 再次 handoff → 子图全新
    r3 = await agent.ainvoke({"messages": [HumanMessage(content="空调异响")]}, config=cfg, context=_ctx())
    intr = r3.get("__interrupt__")
    assert intr, "第二场诊断应在追问处挂起"
    # ★ round 归零重来（上一场的轮次不残留 —— 探针 P4 的行为级断言）
    assert intr[0].value["round"] == 1
    assert first_round == 1


# ── 5. 挂起期间快照兼容（chat 层零改动的依据）────────────────────────────

async def test_pending_question_readable_via_snapshot(triage_io):
    """挂起时 chat._pending_question 能从快照取到追问文本（协议兼容）。"""
    from src.api.routers.chat import _pending_question

    agent, _, _ = await _build(triage_io)
    cfg = {"configurable": {"thread_id": "t5"}}
    await agent.ainvoke({"messages": [HumanMessage(content="车机黑屏")]}, config=cfg, context=_ctx())

    assert await _pending_question(agent, cfg) == QUESTION


# ── 6. 回归锁：business_line 字段（单图化测试炸出的隐性 bug）─────────────

def test_triage_state_carries_business_line():
    """★ 曾缺失的字段：构造入参曾被 pydantic 静默丢弃，节点③读 state.business_line
    直接 AttributeError —— tests/ 此前无用例真正跑到节点③（live eval 是
    integration），直到单图化端到端测试首次全链路执行引擎才暴露。"""
    from src.agents.triage.state import TriageState

    state = TriageState(session_id="t", business_line="ia")
    assert state.business_line == "ia"
    state.business_line = "ev"          # run_triage 多轮路径有属性赋值
    assert state.business_line == "ev"


# ── 7. 开关接线：SUBGRAPH_TRIAGE_ENABLED 选择编排图 ─────────────────────

async def test_chat_agent_switch_selects_orchestrator(monkeypatch):
    """flag=true → chat 入口取编排图；flag=false → 旧 supervisor。"""
    from contextlib import asynccontextmanager

    import src.api.routers.chat as chat_mod

    calls = {"orch": 0, "sup": 0}

    async def fake_orch():
        calls["orch"] += 1
        return object()

    import src.agents.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "get_orchestrator_agent", fake_orch)
    monkeypatch.setattr(chat_mod, "get_supervisor_agent", fake_orch.__wrapped__ if hasattr(fake_orch, "__wrapped__") else fake_orch)

    async def fake_sup():
        calls["sup"] += 1
        return object()

    monkeypatch.setattr(chat_mod, "get_supervisor_agent", fake_sup)

    monkeypatch.setattr(chat_mod.settings, "SUBGRAPH_TRIAGE_ENABLED", True)
    await chat_mod._get_chat_agent()
    assert calls == {"orch": 1, "sup": 0}

    monkeypatch.setattr(chat_mod.settings, "SUBGRAPH_TRIAGE_ENABLED", False)
    await chat_mod._get_chat_agent()
    assert calls == {"orch": 1, "sup": 1}


# ── 8. enrich 并行化（五个独立查询 gather，合并语义不变）────────────────

async def test_enrich_parallel_parts_and_merge(monkeypatch):
    """并行版：五个部分都被调、结果按语义合并回候选。"""
    import asyncio

    import src.agents.triage.graph_queries as gq
    from src.agents.triage.state import CandidateCause

    called = []

    def fake_pd(cands):
        called.append("pd")
        return {
            "phenom": {c.code: [{"name": "中控屏黑屏", "is_core": True, "weight": 0.8}] for c in cands},
            "domain": {c.code: "智能座舱" for c in cands},
        }

    def fake_dtc(cands):
        called.append("dtc")
        return {c.code: ["U0100"] for c in cands}

    def fake_verify(cands):
        called.append("verify")
        return {c.code: "检查线束" for c in cands}

    def fake_located(codes):
        called.append("located")
        return {codes[0]: [{"name": "主机", "ci_no": "CI-1", "module": "座舱", "supplier": "A"}]}

    def fake_co(codes):
        called.append("co")
        return {}

    monkeypatch.setattr(gq, "_enrich_part_phenom_domain", fake_pd)
    monkeypatch.setattr(gq, "_enrich_part_dtc", fake_dtc)
    monkeypatch.setattr(gq, "_enrich_part_verify_items", fake_verify)
    monkeypatch.setattr(gq, "_enrich_part_located_in", fake_located)
    monkeypatch.setattr(gq, "_enrich_part_co_occurs", fake_co)

    cands = [CandidateCause(code="RC-1", name="软件异常", matched_phenomena=["中控屏黑屏"])]
    out = await gq.enrich_cause_details_async(cands)

    assert set(called) == {"pd", "dtc", "verify", "located", "co"}   # 五路全跑
    c = out[0]
    assert c.all_phenomena == ["中控屏黑屏"]
    assert c.phenomena_weight == {"中控屏黑屏": 0.8}
    assert c.is_core_match is True
    assert c.domain == "智能座舱" and c.dtc_matched == ["U0100"]
    assert c.verify_items == "检查线束"
    assert c.related_config_items[0]["ci_no"] == "CI-1"
