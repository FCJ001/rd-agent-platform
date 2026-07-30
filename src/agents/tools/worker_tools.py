"""Worker 工具：封装分诊 Agent 和去重匹配器为 Supervisor 可调用的 @tool。"""

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agents.triage.graph import run_triage, TriageDeps, build_triage_graph
from src.agents.triage.state import TriagePhase
from src.agents.workers.triage_agent import TriageAgent
from src.core.deps import UserContext
from src.infra.redis_cache import get_checkpointer_redis


@tool
async def call_triage_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """启动故障诊断流程。
    适用场景：用户描述车辆故障现象（如"车机黑屏"、"动力不足"、"充电异常"等），
    需要系统化诊断、分析根因时。后续多轮追问由系统自动处理，无需再次调用。

    Args:
        message: 用户描述的故障现象（原文传递，不要改写）
    """
    session_id = runtime.context.session_id
    user_id = runtime.context.user_id
    role = runtime.context.role

    agent = TriageAgent()
    deps = agent._build_deps()

    # 首轮：初始化空状态，执行第一轮诊断
    reply, new_state = await run_triage(
        user_message=message,
        thread_id=f"{user_id}:{session_id}",
        deps=deps,
        existing_state=None,
        viewer_role=role,
    )

    # 首轮即收敛 → 直接返回结论
    if new_state.phase == TriagePhase.CONCLUDE:
        return reply

    # 未收敛：保存状态到 Redis，设置 active flag，等待用户回复
    redis = get_checkpointer_redis()
    state_key = f"triage_state:{user_id}:{session_id}"
    await redis.set(state_key, new_state.model_dump_json(), ex=3600)

    active_key = f"triage_active:{user_id}:{session_id}"
    await redis.set(active_key, "1", ex=3600)

    return reply


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
