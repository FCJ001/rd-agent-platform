"""Worker 工具：封装分诊 Agent 和去重匹配器为 Supervisor 可调用的 @tool。

B2 重构：call_triage_agent 用 langgraph interrupt() 驱动多轮追问 ——
工具在图视角下恢复"单轮"语义（跑一轮、暂停等用户、恢复继续），
回合控制权统一收敛到 Supervisor checkpointer，不再有路由层 Redis 标记。
"""

import asyncio
import re

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt

from src.agents.triage.graph import run_triage
from src.agents.triage.lock import ConversationBusyError, session_lock
from src.agents.triage.session_store import TriageSessionStore, is_triage_exit
from src.agents.triage.state import TriagePhase
from src.agents.workers.triage_agent import TriageAgent
from src.core.deps import UserContext
from src.core.logger import logger
from src.infra.redis_cache import get_checkpointer_redis
from src.utils.dtc import extract_dtc_codes


# ── 同一会话并发防护 ──
# 同一 thread_id 的两条消息并发进来会各自 load→跑→save，后写覆盖先写、
# 追问错位。互斥由 src/agents/triage/lock.py 提供：
#   Redis 分布式锁（跨进程/worker）+ 进程内 asyncio.Lock（同进程排队）。
# ★ 原来这里是纯进程内的 _triage_locks —— 多 worker 部署（Dockerfile 默认
#   UVICORN_WORKERS=2）下每个进程一把锁，等于没锁，已移除。
# interrupt 挂起时锁随异常展开释放，resume 重放时重新获取，不会死锁。


def _interrupt_payload(round_no: int, question: str) -> dict:
    """暂停时抛给调用方的上下文，chat 层可据此渲染追问卡片。"""
    return {"type": "triage_followup", "round": round_no, "question": question[:500]}


async def _finish_triage(store: TriageSessionStore, thread_id: str, reply: str) -> str:
    """收敛/终止：清工作状态，结论文本回到 Supervisor 消息历史。

    ★ 结论不再写长期记忆：记忆集合的 namespace 是 users/{uid}，写进去只有
      本人能召回，同一故障两个人各诊断一次互相看不到。结论已经落在
      ai_triage_results（带 user_id / business_line），由 search_past_diagnoses
      按「我本人的」和「本业务线的」两个维度复用。

    ★ 不在这里释放会话锁 —— 锁由 call_triage_agent 的 session_lock 上下文
      统一持有到本轮结束。原实现在这里调 _release_thread_lock，但那时外层
      async with 还没退出（锁仍被自己持有），实际是个空操作；换成分布式锁后
      那样写会真的去删 key，语义就错了。
    """
    await store.clear(thread_id)
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
    # 数据作用域：工程师/业务角色在 users.business_line 上挂所属线，其它角色
    # 可能为空 —— 为空时下游不做业务线过滤（会打警告），关联问题单后由单子的
    # 归属覆盖（见 node_load_issue）
    business_line = runtime.context.business_line or ""

    try:
        async with session_lock(thread_id):
            return await _run_triage_turn(
                message, user_id, session_id, role, thread_id, business_line
            )
    except ConversationBusyError:
        # ★ 必须捕获后返回文本，不能往上抛：ConversationBusyError 是裸
        #   Exception，交给全局处理器会变成 HTTP 500，用户以为系统挂了。
        #   这里返回一句人话，模型会把它展示给用户（同时日志留痕）。
        logger.info(f"[TRIAGE] 会话忙，拒绝并发处理 thread={thread_id}")
        return "上一次诊断还在处理中（可能是刚才那条消息）。请稍等几秒再发送。"


async def _run_triage_turn(
    message: str, user_id, session_id, role, thread_id: str, business_line: str = "",
) -> str:
    agent = TriageAgent()
    deps = agent._build_deps()
    store = TriageSessionStore(get_checkpointer_redis())

    saved = await store.load(thread_id)
    if saved is None:
        # 首轮：立即跑第一轮诊断；不收敛则落盘进度，进入追问循环
        reply, state = await run_triage(
            user_message=message, thread_id=thread_id, deps=deps,
            existing_state=None, viewer_role=role,
            business_line=business_line, user_id=user_id,
        )
        if state.phase in (TriagePhase.CONCLUDE, TriagePhase.END):
            return await _finish_triage(store, thread_id, reply)
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
            business_line=business_line, user_id=user_id,
        )
        if state.phase in (TriagePhase.CONCLUDE, TriagePhase.END):
            return await _finish_triage(store, thread_id, reply)
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
    # 数据作用域：透传用户所属业务线，避免影响范围跨线聚合
    report = await agent.analyze(message, (runtime.context.business_line or "") if runtime else "")

    # 保存到影子表，供反馈回写
    # 风险等级从报告里判定：英文词用词边界匹配（裸子串会把 "below" 认成 low）
    low = report.lower()
    if re.search(r"\bcritical\b", low) or "高危" in report or "风险极高" in report:
        risk = "critical"
    elif re.search(r"\bhigh\b", low) or "高风险" in report:
        risk = "high"
    elif re.search(r"\blow\b", low) or "低风险" in report or "风险较低" in report:
        risk = "low"
    else:
        risk = "medium"

    try:
        from sqlalchemy import text
        from src.infra.db import AsyncSessionLocal
        session_id = runtime.context.session_id if runtime else "unknown"
        # ★ 记录创建者：feedback 接口按 user_id 做所有权校验（IDOR 防护）
        user_id = int(runtime.context.user_id) if runtime else None
        async with AsyncSessionLocal() as db:
            await db.execute(
                text(
                    "INSERT INTO ai_impact_analysis (user_id, session_id, raw_input, report_md, risk_level) "
                    "VALUES (:uid, :sid, :input, :report, :risk)"
                ),
                {"uid": user_id, "sid": session_id, "input": message, "report": report, "risk": risk},
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
        user_id = int(runtime.context.user_id) if runtime else None
        async with AsyncSessionLocal() as db:
            await db.execute(
                text(
                    "INSERT INTO ai_report_interpretations "
                    "(user_id, session_id, report_type, raw_text, interpretation) "
                    "VALUES (:uid, :sid, :rtype, :raw, :interp)"
                ),
                {"uid": user_id, "sid": session_id, "rtype": report_type, "raw": message, "interp": result},
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

    # 提取 DTC：SAE 标准 5 位（字母+4位十六进制），旧正则漏 P0A7F 这类码
    dtc_codes = ",".join(extract_dtc_codes(message))
    # 数据作用域：透传用户的所属业务线（工程师/业务角色有值）。为空时不按
    # 业务线过滤 —— 之前这里漏传，函数默认值 "ia" 让去重恒在 ia 切片里搜
    business_line = (runtime.context.business_line or "") if runtime else ""

    matcher = DedupMatcher()
    result = await matcher.detect_by_text(message, dtc_codes, business_line)

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
