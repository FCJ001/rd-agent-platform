"""Worker 工具：封装分诊 Agent 和去重匹配器为 Supervisor 可调用的 @tool。

B2 重构：call_triage_agent 用 langgraph interrupt() 驱动多轮追问 ——
工具在图视角下恢复"单轮"语义（跑一轮、暂停等用户、恢复继续），
回合控制权统一收敛到 Supervisor checkpointer，不再有路由层 Redis 标记。
"""

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt

from src.agents.triage.graph import run_triage
from src.agents.triage.memory import save_diagnosis_memory
from src.agents.triage.session_store import TriageSessionStore, is_triage_exit
from src.agents.triage.state import TriagePhase
from src.agents.workers.triage_agent import TriageAgent
from src.core.deps import UserContext
from src.core.logger import logger
from src.infra.redis_cache import get_checkpointer_redis


def _interrupt_payload(round_no: int, question: str) -> dict:
    """暂停时抛给调用方的上下文，chat 层可据此渲染追问卡片。"""
    return {"type": "triage_followup", "round": round_no, "question": question[:500]}


async def _finish_triage(store: TriageSessionStore, thread_id: str, state, reply: str) -> str:
    """收敛/终止：清工作状态 + 结论写长期记忆，结论文本回到 Supervisor 消息历史。"""
    await store.clear(thread_id)
    try:
        await save_diagnosis_memory(state, thread_id)
    except Exception as e:
        logger.warning(f"[TRIAGE] 长期记忆保存失败 thread={thread_id}: {e}")
    return reply


@tool
async def call_triage_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """启动故障诊断流程，系统自动多轮追问直至给出结论。
    适用场景：用户描述车辆故障现象（如"车机黑屏"、"动力不足"、"充电异常"等），
    需要系统化诊断、分析根因时。追问期间用户的回复由系统自动转交给诊断流程，
    无需再次调用本工具；工具返回时即为最终诊断结论（或用户主动退出的提示）。

    Args:
        message: 用户描述的故障现象（原文传递，不要改写）
    """
    user_id = runtime.context.user_id
    session_id = runtime.context.session_id
    role = runtime.context.role
    thread_id = f"{user_id}:{session_id}"

    agent = TriageAgent()
    deps = agent._build_deps()
    store = TriageSessionStore(get_checkpointer_redis())

    saved = await store.load(thread_id)
    if saved is None:
        # 首轮：立即跑第一轮诊断；不收敛则落盘进度，进入追问循环
        reply, state = await run_triage(
            user_message=message, thread_id=thread_id, deps=deps,
            existing_state=None, viewer_role=role,
        )
        if state.phase in (TriagePhase.CONCLUDE, TriagePhase.END):
            return await _finish_triage(store, thread_id, state, reply)
        await store.save(thread_id, state, reply)
    else:
        state, reply = saved.state, saved.reply
        # 快进空转：resume 会让本工具从头重放，而图 checkpoint 只在节点完成时
        # 落盘 —— 前面已完成的 state.round 轮的 interrupt() 带有记录值（历史回答），
        # 必须按序空转消耗掉对齐位置，当前用户的回答才会落在循环里真正的
        # interrupt() 上。不空转会索引错位，不快进则会重跑已完成轮次的 LLM 调用。
        for _ in range(state.round):
            interrupt(_interrupt_payload(0, ""))

    while True:
        answer = interrupt(_interrupt_payload(state.round + 1, reply))
        if is_triage_exit(answer):
            await store.clear(thread_id)
            return "分诊已按你的要求退出，本轮诊断作废。如需重新诊断，请直接描述故障现象。"

        reply, state = await run_triage(
            user_message=answer, thread_id=thread_id, deps=deps,
            existing_state=state, viewer_role=role,
        )
        if state.phase in (TriagePhase.CONCLUDE, TriagePhase.END):
            return await _finish_triage(store, thread_id, state, reply)
        await store.save(thread_id, state, reply)


@tool
async def call_impact_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """分析变更请求的影响范围和风险。
    适用场景：评估 CR/变更单的影响范围，检查基线冲突、依赖冲突、重复变更。
    在创建或评审变更请求时使用。

    Args:
        message: 变更描述或 CR 编号（原文传递）
    """
    from src.agents.workers.impact_agent import get_impact_agent
    agent = get_impact_agent()
    report = await agent.analyze(message)

    # 保存到影子表，供反馈回写
    risk = "medium"
    if "critical" in report.lower() or "严重" in report:
        risk = "critical"
    elif "high" in report.lower() or "高风险" in report:
        risk = "high"
    elif "low" in report.lower() or "低风险" in report:
        risk = "low"

    try:
        from sqlalchemy import text
        from src.infra.db import AsyncSessionLocal
        session_id = runtime.context.session_id if runtime else "unknown"
        async with AsyncSessionLocal() as db:
            await db.execute(
                text(
                    "INSERT INTO ai_impact_analysis (session_id, raw_input, report_md, risk_level) "
                    "VALUES (:sid, :input, :report, :risk)"
                ),
                {"sid": session_id, "input": message, "report": report, "risk": risk},
            )
            await db.commit()
    except Exception:
        import traceback
        from src.core.logger import logger
        logger.warning(f"[IMPACT] 保存分析结果失败: {traceback.format_exc()}")

    return report


@tool
async def call_report_agent(message: str, report_type: str = "DTC扫描", runtime: ToolRuntime[UserContext] = None) -> str:
    """解读车辆报告/日志（DTC扫描报告、台架测试报告、OTA回归测试报告等）。
    适用场景：工程师上传了检测报告需要解读、需要判断指标是否异常。

    Args:
        message: 报告内容或报告摘要
        report_type: 报告类型（DTC扫描/台架测试/OTA回归），可选
    """
    from src.agents.workers.report_agent import get_report_agent
    agent = get_report_agent()
    result = await agent.analyze(message, report_type)

    # 保存到影子表，供反馈回写
    try:
        from sqlalchemy import text
        from src.infra.db import AsyncSessionLocal
        session_id = runtime.context.session_id if runtime else "unknown"
        async with AsyncSessionLocal() as db:
            await db.execute(
                text(
                    "INSERT INTO ai_report_interpretations "
                    "(session_id, report_type, raw_text, interpretation) "
                    "VALUES (:sid, :rtype, :raw, :interp)"
                ),
                {"sid": session_id, "rtype": report_type, "raw": message, "interp": result},
            )
            await db.commit()
    except Exception:
        import traceback
        from src.core.logger import logger
        logger.warning(f"[REPORT] 保存解读结果失败: {traceback.format_exc()}")

    return result


@tool
async def call_dedup_check(message: str, runtime: ToolRuntime[UserContext] = None) -> str:
    """检查新问题是否与已有问题重复。
    适用场景：用户报告新故障后，检查是否存在类似的未关闭问题单。
    如果发现重复，直接返回已有诊断结论，避免重复劳动。

    Args:
        message: 用户的问题描述（原文传递）
    """
    from src.agents.dedup.matcher import DedupMatcher
    from src.core.logger import logger

    logger.info(f"[DEDUP] call_dedup_check 被调用, message={message[:80]}")

    # Extract DTC codes from message
    import re
    dtc_codes = ",".join(re.findall(r"[A-Z]\d{4,5}", message))

    matcher = DedupMatcher()
    result = await matcher.detect_by_text(message, dtc_codes)

    logger.info(f"[DEDUP] 检测完成, is_duplicate={result.is_duplicate}, matches={len(result.matches)}")

    if not result.is_duplicate:
        return "未发现重复的未关闭问题单，可以继续诊断。"

    lines = [f"发现 {len(result.matches)} 个可能重复的问题单：\n"]
    for i, m in enumerate(result.matches, 1):
        evidence_map = {
            "model_and_sw": "车型+软件版本一致",
            "dtc": "DTC故障码重叠",
            "model_and_sw+dtc": "车型+软件版本一致 且 DTC故障码重叠",
        }
        lines.append(
            f"### {i}. {m.issue_no}: {m.title}\n"
            f"- 向量相似度: {m.similarity:.2%}\n"
            f"- 匹配证据: {evidence_map.get(m.evidence, m.evidence)}\n"
        )
    lines.append("\n建议先查看以上问题单的已有诊断结论，确认是否确为重复。")
    return "\n".join(lines)


from src.agents.tools.remote_knowledge import REMOTE_TOOLS
from src.agents.tools.platform_tools import PLATFORM_TOOLS

WORKER_TOOLS = [call_triage_agent, call_impact_agent, call_report_agent, call_dedup_check] + REMOTE_TOOLS + PLATFORM_TOOLS
