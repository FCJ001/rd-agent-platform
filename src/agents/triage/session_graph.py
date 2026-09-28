# ============================================================
# 多轮分诊会话子图（P1 单图化）
# 设计：docs/multi-agent-architecture-design.md §6.3
#
# 与旧机制（worker_tools.call_triage_agent 的工具内循环）的本质区别：
#   旧：interrupt 在工具里，工具是原子单元 —— resume 重放整个工具，
#       进度只能存外部 TriageSessionStore + 快进空转对齐。
#   新：分诊作为父图的【子图节点】，每个节点独立 checkpoint ——
#       追问等待是图内 interrupt，恢复从断点节点续跑，
#       **没有外部 store、没有快进、没有重放**。
#
# 节点复用 graph.py 的现有实现（extract/safety/query/ask/parse/conclude/
# save_record 全部原样），只重排等待结构：
#
#   intake（首轮：scope 初始化 + 词表加载）
#     → load_issue → extract → check_safety → query
#         ├─ 收敛 → conclude → save_record → END
#         └─ 未收敛 → ask_details → wait_answer(interrupt 等回答)
#                        ├─ 退出指令 → END（诊断作废）
#                        └─ 回答 → parse_answer → extract（下一轮循环）
#
# ★ resume 重放语义对齐：interrupt() 是 wait_answer 的第一条语句，
#   重放零 LLM 成本；ask_details 的 LLM 工作在上一节点已完成并落盘，
#   永不重跑 —— 这就是快进结构被消灭的机制根源。
#
# 机制依据：scripts/probe_orchestrator_mechanics.py 四项探针
# （agent 作父图节点 / 工具 Command 跳转 / 子图 interrupt+父图 resume /
#  子图二次进入状态不残留）在 langgraph 1.1.6 上全部验证通过。
# ============================================================

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.triage.graph import (
    TriageDeps,
    node_ask_details,
    node_check_safety,
    node_conclude,
    node_extract_phenomena,
    node_load_issue,
    node_parse_answer,
    node_query_candidates,
    node_save_record,
    load_phenomenon_vocabulary,
)
from src.agents.triage.session_store import is_triage_exit
from src.agents.triage.state import TriagePhase, TriageState
from src.core.logger import logger


class TriageSessionState(TriageState):
    """会话子图状态 = 分诊引擎状态 + 与父图共享的编排键。

    active_task / task_scope / messages 与父图（OrchestratorState）
    同名同步：handoff 工具写入 task_scope，intake 读取初始化引擎字段；
    分诊内部字段（confirmed/denied/round…）不进父图 —— 探针 P4 验证
    子图节点二次进入时状态全新，上一场诊断不残留。
    """

    active_task: str | None = None
    task_scope: dict = {}


def _interrupt_payload(round_no: int, question: str) -> dict:
    """挂起载荷 —— 与旧 worker_tools 的形状保持一致：
    chat 层的快照路由（_pending_question）、SSE 补推、前端渲染都依赖
    {"type": "triage_followup", "round", "question"} 这个契约，不能改。"""
    return {"type": "triage_followup", "round": round_no, "question": question[:500]}


def build_triage_session_graph(deps: TriageDeps):
    """构建多轮分诊会话子图（编译产物，作为父图的 triage 节点）。"""

    async def _intake(state: TriageSessionState) -> dict:
        """首轮入口：从父图 task_scope 初始化引擎字段 + 加载词表。

        每次诊断（子图每次被进入）都全新跑一遍 —— P4 保证无残留。
        """
        scope = state.task_scope or {}
        vocabulary = await load_phenomenon_vocabulary(scope.get("business_line", ""))
        logger.info(
            f"[TRIAGE-SG] intake session={scope.get('session_id')} "
            f"role={scope.get('viewer_role')} line={scope.get('business_line')}"
        )
        return {
            "round": 0,
            "phase": TriagePhase.EXTRACT,
            "viewer_role": scope.get("viewer_role") or "customer",
            "business_line": scope.get("business_line") or "",
            "user_id": scope.get("user_id") or "",
            "session_id": scope.get("session_id") or "",
            "issue_id": scope.get("issue_id"),
            "phenomenon_vocabulary": vocabulary,
            # 上一场诊断的残留清洗（防御：正常路径 P4 已保证全新）
            "confirmed_phenomena": [],
            "denied_phenomena": [],
            "dtc_codes": [],
            "candidate_causes": [],
            "follow_up_questions": [],
            "force_conclude": False,
        }

    async def _wait_answer(state: TriageSessionState) -> dict:
        """等待用户回答挂起的追问。

        ★ interrupt() 必须是第一条语句：resume 重放本节点时零 LLM 成本，
        且拒绝/异常若发生在 interrupt 之前（[E] 语义），值不提交、重试用新值。
        """
        question = state.follow_up_questions[-1] if state.follow_up_questions else "请继续描述故障细节。"
        answer = interrupt(_interrupt_payload(state.round + 1, question))

        if is_triage_exit(answer):
            logger.info(f"[TRIAGE-SG] 用户退出诊断 round={state.round}")
            return {
                "phase": TriagePhase.END,
                "messages": [AIMessage(
                    content="分诊已按你的要求退出，本轮诊断作废。如需重新诊断，请直接描述故障现象。"
                )],
            }

        # 回答入历史 + 轮次推进（与旧 run_triage 的逐轮语义一致：
        # 新一轮从 parse_answer 开始消化这条回答）
        return {
            "round": state.round + 1,
            "messages": [HumanMessage(content=answer)],
        }

    def _route_after_wait(state: TriageSessionState) -> str:
        if state.phase == TriagePhase.END:
            return "end"
        return "parse_answer"

    async def _load_issue(state):       return await node_load_issue(state, deps)
    async def _extract(state):          return await node_extract_phenomena(state, deps)
    async def _check_safety(state):     return await node_check_safety(state, deps)
    async def _query(state):            return await node_query_candidates(state, deps)
    async def _ask(state):              return await node_ask_details(state, deps)
    async def _parse(state):            return await node_parse_answer(state, deps)
    async def _conclude(state):         return await node_conclude(state, deps)
    async def _save_record(state):      return await node_save_record(state, deps)

    g = StateGraph(TriageSessionState)
    g.add_node("intake", _intake)
    g.add_node("load_issue", _load_issue)
    g.add_node("extract_phenomena", _extract)
    g.add_node("check_safety", _check_safety)
    g.add_node("query_candidates", _query)
    g.add_node("ask_details", _ask)
    g.add_node("wait_answer", _wait_answer)
    g.add_node("parse_answer", _parse)
    g.add_node("conclude", _conclude)
    g.add_node("save_record", _save_record)

    g.add_edge(START, "intake")
    g.add_edge("intake", "load_issue")
    g.add_edge("load_issue", "extract_phenomena")
    g.add_edge("extract_phenomena", "check_safety")
    g.add_edge("parse_answer", "extract_phenomena")

    def _route_after_safety(state: TriageSessionState) -> str:
        if state.phase == TriagePhase.END:
            return "end"
        if state.phase == TriagePhase.QUERY:
            return "query_candidates"
        return "ask_details"

    def _route_after_query(state: TriageSessionState) -> str:
        if state.phase == TriagePhase.CONCLUDE:
            return "conclude"
        return "ask_details"

    def _route_after_conclude(state: TriageSessionState) -> str:
        if state.candidate_causes:
            return "save_record"
        return "end"

    g.add_conditional_edges("check_safety", _route_after_safety, {
        "query_candidates": "query_candidates",
        "ask_details": "ask_details",
        "end": END,
    })
    g.add_conditional_edges("query_candidates", _route_after_query, {
        "conclude": "conclude",
        "ask_details": "ask_details",
    })
    # 追问完进等待节点（旧图是 end 本轮、工具层 interrupt；新图等待在图内）
    g.add_edge("ask_details", "wait_answer")
    g.add_conditional_edges("wait_answer", _route_after_wait, {
        "parse_answer": "parse_answer",
        "end": END,
    })
    g.add_conditional_edges("conclude", _route_after_conclude, {
        "save_record": "save_record",
        "end": END,
    })
    g.add_edge("save_record", END)

    return g.compile()
