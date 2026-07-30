"""分诊 StateGraph。参考天宫医疗版 inquiry/graph.py（10 节点 → 7 节点精简）。"""

import json
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.triage.state import TriageState, TriagePhase, CandidateCause
from src.agents.triage.prompts import (
    EXTRACT_PHENOMENA_PROMPT, ASK_DETAILS_PROMPT,
    PARSE_ANSWER_PROMPT, ROLE_PROMPTS,
)
from src.agents.triage.db_queries import (
    load_issue_context, match_phenomena_by_names,
    lookup_dtc_codes, save_triage_result,
)
from src.agents.triage.graph_queries import query_causes_by_phenomena, enrich_cause_details
from src.agents.triage.confidence import apply_context_weights, check_convergence, MAX_ROUNDS
from src.core.config import get_settings
from src.core.logger import logger


settings = get_settings()


@dataclass
class TriageDeps:
    llm_json: ChatOpenAI   # for structured JSON output (extract, parse)
    llm_chat: ChatOpenAI   # for natural language output (ask, conclude)
    db_session_factory: callable


def _get_llm_json() -> ChatOpenAI:
    """LLM with JSON mode — for extract_phenomena, parse_answer."""
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.3,
        timeout=30,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _get_llm_chat() -> ChatOpenAI:
    """LLM without JSON mode — for ask_details, conclude."""
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.5,
        timeout=30,
    )


def _parse_llm_json(response) -> dict:
    """从 LLM 响应中提取 JSON，处理 markdown 代码块包裹。"""
    content = response.content.strip()
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return json.loads(content)


def _get_last_human_message(messages: list) -> str:
    """获取最后一条用户消息，兼容 LangChain 对象和 Redis 反序列化的 dict。"""
    for msg in reversed(messages):
        if hasattr(msg, "content"):
            type_name = msg.__class__.__name__
            content = msg.content
        else:
            type_name = msg.get("type", "")
            content = msg.get("content", "")
        if type_name in ("HumanMessage", "human"):
            return content
    return ""


# ════════════════════════════════════════════════════════════════════════
# 节点函数
# ════════════════════════════════════════════════════════════════════════

async def node_load_issue(state: TriageState, deps: TriageDeps) -> dict:
    """节点①：加载问题单上下文。仅 round==0 且 issue_id 存在时执行。"""
    if state.round > 0 or not state.issue_id:
        return {}

    logger.info(f"[TRIAGE] 节点① load_issue id={state.issue_id} round={state.round}")
    async for db in deps.db_session_factory():
        try:
            ctx = await load_issue_context(db, state.issue_id)
            if ctx:
                result = {
                    "issue_title": ctx["issue_title"],
                    "issue_desc": ctx["issue_desc"],
                    "issue_dtc_snapshot": ctx["issue_dtc_snapshot"],
                }
                if not state.viewer_role or state.viewer_role == "customer":
                    result["viewer_role"] = ctx.get("source", "customer")
                logger.info(f"[TRIAGE] 节点① issue_loaded title={ctx['issue_title'][:50]} viewer_role={result.get('viewer_role')}")
                return result
            return {}
        finally:
            break


async def node_extract_phenomena(state: TriageState, deps: TriageDeps) -> dict:
    """节点②：L1 LLM 口语→现象名 + DTC 提取。"""
    last_msg = _get_last_human_message(state.messages)
    if not last_msg:
        return {"phase": TriagePhase.ASK}

    logger.info(f"[TRIAGE] 节点② extract_phenomena round={state.round} last_msg={last_msg[:60]}")
    # 构建用户输入（含问题单上下文）
    user_input = last_msg
    if state.issue_title and state.round == 0:
        user_input = f"【问题单标题】{state.issue_title}\n【问题单描述】{state.issue_desc}\n{user_input}"
    if state.issue_dtc_snapshot and state.round == 0:
        user_input += f"\n【DTC快照】{state.issue_dtc_snapshot}"

    prompt = EXTRACT_PHENOMENA_PROMPT.format(
        user_input=user_input,
        phenomenon_vocabulary=state.phenomenon_vocabulary,
    )
    response = await deps.llm_json.ainvoke([SystemMessage(content=prompt)])

    try:
        parsed = _parse_llm_json(response)
        new_phenomena = parsed.get("phenomena", [])
        new_dtc = parsed.get("dtc_codes", [])
    except Exception:
        logger.warning(f"[TRIAGE] 节点② LLM JSON 解析失败")
        new_phenomena = []
        new_dtc = []

    logger.info(f"[TRIAGE] 节点② extracted phenomena={new_phenomena} dtc={new_dtc}")

    # Merge with existing
    all_confirmed = list(set(state.confirmed_phenomena) | set(new_phenomena))
    all_dtc = list(set(state.dtc_codes) | set(new_dtc))

    if all_confirmed:
        return {
            "confirmed_phenomena": all_confirmed,
            "dtc_codes": all_dtc,
            "phase": TriagePhase.QUERY,
        }
    else:
        return {
            "dtc_codes": all_dtc,
            "phase": TriagePhase.ASK,
        }


async def node_query_candidates(state: TriageState, deps: TriageDeps) -> dict:
    """节点③：L2 DB 匹配现象名→id + L3 Neo4j 查询候选根因 + 置信度计算。"""
    logger.info(f"[TRIAGE] 节点③ query_candidates confirmed={state.confirmed_phenomena} dtc={state.dtc_codes}")
    # Step 1: DB match phenomenon names → IDs
    async for db in deps.db_session_factory():
        try:
            matched = await match_phenomena_by_names(db, state.confirmed_phenomena)
            break
        finally:
            break

    if not matched:
        logger.info(f"[TRIAGE] 节点③ 无匹配现象 → force_conclude")
        return {"phase": TriagePhase.CONCLUDE, "force_conclude": True}

    # Step 2: Neo4j query candidate root causes
    phenom_names = [m["name"] for m in matched]
    candidates = query_causes_by_phenomena(phenom_names)
    logger.info(f"[TRIAGE] 节点③ matched_phenomena={phenom_names} neo4j_candidates={len(candidates)}")

    if candidates:
        candidates = enrich_cause_details(candidates)

        if state.dtc_codes:
            for c in candidates:
                pass  # dtc matching is handled in graph_queries.enrich_cause_details

        candidates = apply_context_weights(
            candidates, state.dtc_codes, state.denied_phenomena
        )

    should_conclude, force_conclude = check_convergence(candidates, state.round)
    logger.info(f"[TRIAGE] 节点③ should_conclude={should_conclude} force={force_conclude} top_confidence={candidates[0].confidence if candidates else 0}")

    if should_conclude:
        return {
            "candidate_causes": candidates,
            "phase": TriagePhase.CONCLUDE,
            "force_conclude": force_conclude,
        }
    elif not candidates:
        return {
            "candidate_causes": [],
            "phase": TriagePhase.CONCLUDE,
            "force_conclude": True,
        }
    else:
        return {
            "candidate_causes": candidates,
            "confidence": candidates[0].confidence,
            "phase": TriagePhase.ASK,
        }


async def node_ask_details(state: TriageState, deps: TriageDeps) -> dict:
    """节点④：LLM 生成追问（问伴随现象、DTC、发生条件、频次）。"""
    logger.info(f"[TRIAGE] 节点④ ask_details round={state.round} candidates={len(state.candidate_causes)}")
    candidate_summary = "\n".join(
        f"- {c.name}（置信度 {c.confidence:.0%}，匹配现象: {', '.join(c.matched_phenomena)}"
        f"{'，未命中: ' + ', '.join([p for p in c.all_phenomena if p not in c.matched_phenomena]) if c.all_phenomena else ''}"
        f"）"
        for c in state.candidate_causes[:5]
    ) if state.candidate_causes else "暂无高置信度候选根因"

    # Build dialogue history from messages (handle both dict and object types)
    dialogue_lines = []
    for msg in state.messages[-6:]:
        # Handle both LangChain message objects and dict-serialized messages from Redis
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        type_name = msg.__class__.__name__ if hasattr(msg, "__class__") else msg.get("type", "")
        role = "用户" if (type_name == "HumanMessage" or type_name == "human") else "助手"
        dialogue_lines.append(f"{role}：{content[:200]}")
    dialogue_history = "\n".join(dialogue_lines) if dialogue_lines else "（无历史）"

    prompt = ASK_DETAILS_PROMPT.format(
        dialogue_history=dialogue_history,
        confirmed_phenomena="、".join(state.confirmed_phenomena) or "暂无",
        denied_phenomena="、".join(state.denied_phenomena) or "暂无",
        dtc_codes="、".join(state.dtc_codes) or "无",
        candidate_summary=candidate_summary,
    )
    response = await deps.llm_chat.ainvoke([SystemMessage(content=prompt)])

    return {
        "follow_up_questions": [response.content],
        "messages": [AIMessage(content=response.content)],
    }

async def node_parse_answer(state: TriageState, deps: TriageDeps) -> dict:
    """节点⑤：LLM 解析用户回复 → 更新 confirmed/denied/dtc。"""
    last_msg = _get_last_human_message(state.messages)
    if not last_msg:
        return {}

    logger.info(f"[TRIAGE] 节点⑤ parse_answer round={state.round} answer={last_msg[:80]}")
    asked = state.follow_up_questions[-1] if state.follow_up_questions else "请描述更多故障细节"

    prompt = PARSE_ANSWER_PROMPT.format(
        asked_questions=asked,
        user_answer=last_msg,
        phenomenon_vocabulary=state.phenomenon_vocabulary,
    )
    response = await deps.llm_json.ainvoke([SystemMessage(content=prompt)])

    try:
        parsed = _parse_llm_json(response)
        new_confirmed = list(
            set(state.confirmed_phenomena) | set(parsed.get("confirmed", []))
        )
        new_denied = list(
            set(state.denied_phenomena) | set(parsed.get("denied", []))
        )
        new_dtc = list(
            set(state.dtc_codes) | set(parsed.get("dtc_codes", []))
        )
    except Exception:
        new_confirmed = state.confirmed_phenomena
        new_denied = state.denied_phenomena
        new_dtc = state.dtc_codes

    return {
        "confirmed_phenomena": new_confirmed,
        "denied_phenomena": new_denied,
        "dtc_codes": new_dtc,
        "phase": TriagePhase.QUERY,
    }


async def node_conclude(state: TriageState, deps: TriageDeps) -> dict:
    """节点⑥：生成诊断结论 + 存入 ai_triage_results。"""
    logger.info(f"[TRIAGE] 节点⑥ conclude round={state.round} candidates={len(state.candidate_causes)} viewer_role={state.viewer_role}")
    candidates = state.candidate_causes
    if not candidates:
        no_result = "根据您描述的故障现象，暂未在知识库中找到高置信度匹配的根因。建议实车诊断确认。"
        return {
            "phase": TriagePhase.CONCLUDE,
            "diagnostic_summary": no_result,
            "messages": [AIMessage(content=no_result)],
        }

    top1 = candidates[0]
    suspected = [f"{c.name}({c.confidence:.0%})" for c in candidates[1:5]]
    confidence = top1.confidence

    prompt = ROLE_PROMPTS.get(state.viewer_role, ROLE_PROMPTS["customer"]).format(
        confirmed_phenomena="、".join(state.confirmed_phenomena) or "暂无",
        denied_phenomena="、".join(state.denied_phenomena) or "暂无",
        dtc_codes="、".join(state.dtc_codes) or "无",
        primary_cause=top1.name,
        confidence=confidence,
        suspected_causes="、".join(suspected) if suspected else "无",
        domain=top1.domain or "未知",
        force_conclude=state.force_conclude,
    )
    response = await deps.llm_chat.ainvoke([SystemMessage(content=prompt)])
    summary = response.content

    # 追加操作选项（Agent 主动询问用户下一步）
    if confidence >= 0.5:
        summary += (
            "\n\n---\n"
            "### 接下来需要我做什么？\n"
            "- 回复 **创建** — 在 ALM 平台正式记录此问题，分配工程师跟进\n"
            "- 回复 **跳转** — 前往 ALM 平台查看相关历史问题单\n"
            "- 回复 **结案** — 如果已确认根因和修复方案，提交结案建议\n"
            '\n直接回复「创建」、「跳转」或「结案」，或者描述具体需求即可。'
        )

    # Save to DB
    async for db in deps.db_session_factory():
        try:
            await save_triage_result(db, {
                "issue_id": state.issue_id,
                "session_id": state.session_id,
                "raw_input": state.messages[0].content if state.messages else "",
                "confirmed_phenomena": state.confirmed_phenomena,
                "denied_phenomena": state.denied_phenomena,
                "candidate_causes": candidates,
                "primary_cause_code": top1.code,
                "primary_confidence": confidence,
                "suggest_domain_id": None,
                "total_rounds": state.round + 1,
                "force_conclude": state.force_conclude,
            })
            break
        finally:
            break

    return {
        "phase": TriagePhase.CONCLUDE,
        "diagnostic_summary": summary,
        "confidence": confidence,
        "messages": [AIMessage(content=summary)],
    }


# ════════════════════════════════════════════════════════════════════════
# 路由函数
# ════════════════════════════════════════════════════════════════════════

def route_dispatcher(state: TriageState) -> str:
    """入口分发：首轮→load_issue，后续轮次→parse_answer。"""
    if state.round == 0:
        return "load_issue"
    return "parse_answer"


def route_after_extract(state: TriageState) -> str:
    """现象提取后：有现象→query_candidates，无现象→ask_details。"""
    if state.phase == TriagePhase.QUERY:
        return "query_candidates"
    return "ask_details"


def route_after_query(state: TriageState) -> str:
    """查询后：收敛→conclude，否则→ask_details。"""
    if state.phase == TriagePhase.CONCLUDE:
        return "conclude"
    return "ask_details"


# ════════════════════════════════════════════════════════════════════════
# 图组装
# ════════════════════════════════════════════════════════════════════════

def build_triage_graph(deps: TriageDeps):
    """构建并编译分诊 StateGraph。"""
    async def _load_issue(state):       return await node_load_issue(state, deps)
    async def _extract_phenomena(state): return await node_extract_phenomena(state, deps)
    async def _query_candidates(state):  return await node_query_candidates(state, deps)
    async def _ask_details(state):       return await node_ask_details(state, deps)
    async def _parse_answer(state):      return await node_parse_answer(state, deps)
    async def _conclude(state):          return await node_conclude(state, deps)

    graph = StateGraph(TriageState)

    graph.add_node("dispatcher", lambda state: {})
    graph.add_node("load_issue", _load_issue)
    graph.add_node("extract_phenomena", _extract_phenomena)
    graph.add_node("query_candidates", _query_candidates)
    graph.add_node("ask_details", _ask_details)
    graph.add_node("parse_answer", _parse_answer)
    graph.add_node("conclude", _conclude)

    graph.set_entry_point("dispatcher")

    # 固定边
    graph.add_edge("load_issue", "extract_phenomena")
    graph.add_edge("ask_details", END)
    graph.add_edge("conclude", END)

    # 条件边
    graph.add_conditional_edges("dispatcher", route_dispatcher, {
        "load_issue": "load_issue",
        "parse_answer": "parse_answer",
    })
    graph.add_conditional_edges("extract_phenomena", route_after_extract, {
        "query_candidates": "query_candidates",
        "ask_details": "ask_details",
    })
    graph.add_conditional_edges("query_candidates", route_after_query, {
        "conclude": "conclude",
        "ask_details": "ask_details",
    })
    graph.add_edge("parse_answer", "extract_phenomena")

    return graph.compile()


# ════════════════════════════════════════════════════════════════════════
# 对外接口：供 call_triage_agent 工具 和 chat router 调用
# ════════════════════════════════════════════════════════════════════════

async def run_triage(
    user_message: str,
    thread_id: str,
    deps: TriageDeps,
    existing_state: TriageState | None = None,
    viewer_role: str = "customer",
) -> tuple[str, TriageState]:
    """
    执行一轮分诊对话。

    Args:
        user_message: 用户本轮输入
        thread_id: 会话标识（用于关联 Redis 状态）
        deps: 依赖注入容器
        existing_state: 上一轮的状态（多轮对话时传入）
        viewer_role: 查看结论的人的角色（engineer/business/aftersales/customer），影响输出格式

    Returns:
        (assistant_reply, new_state)
    """
    vocabulary = await load_phenomenon_vocabulary()

    if existing_state is None:
        state = TriageState(
            session_id=thread_id,
            phenomenon_vocabulary=vocabulary,
            viewer_role=viewer_role,
        )
    else:
        state = existing_state.model_copy()
        state.round = existing_state.round + 1
        state.phase = TriagePhase.EXTRACT
        state.phenomenon_vocabulary = vocabulary
        state.viewer_role = viewer_role

    state.messages.append(HumanMessage(content=user_message))

    graph = build_triage_graph(deps)
    config = {"configurable": {"thread_id": thread_id}}
    result_dict = await graph.ainvoke(state, config=config)
    result = TriageState(**result_dict)

    # Extract last AI message as reply (compatible with dict and object types)
    reply = ""
    for msg in reversed(result.messages):
        if hasattr(msg, "content"):
            type_name = msg.__class__.__name__
            content = msg.content
        else:
            type_name = msg.get("type", "")
            content = msg.get("content", "")
        if type_name in ("AIMessage", "ai"):
            reply = content
            break

    return reply, result


async def load_phenomenon_vocabulary() -> str:
    """从 DB 加载现象码词汇表（供 LLM prompt 注入）。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from src.infra.db import AsyncSessionLocal

    lines = []
    async with AsyncSessionLocal() as db:
        from src.agents.triage.db_queries import get_all_phenomena
        phenomena = await get_all_phenomena(db)
        for p in phenomena:
            parts = [p["name"]]
            if p.get("colloquial"):
                parts.append(f"（别名：{p['colloquial']}）")
            lines.append("、".join(parts))
    return "\n".join(lines)
