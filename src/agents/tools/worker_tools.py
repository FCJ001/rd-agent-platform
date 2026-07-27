"""Worker 工具：封装分诊 Agent 和去重匹配器为 Supervisor 可调用的 @tool。"""

from dataclasses import dataclass

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agents.triage.graph import run_triage, TriageDeps, build_triage_graph
from src.agents.triage.state import TriagePhase
from src.agents.workers.triage_agent import TriageAgent
from src.infra.redis_cache import get_checkpointer_redis


@dataclass
class UserContext:
    user_id: str
    session_id: str


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

    agent = TriageAgent()
    deps = agent._build_deps()

    # 首轮：初始化空状态，执行第一轮诊断
    reply, new_state = await run_triage(
        user_message=message,
        thread_id=f"{user_id}:{session_id}",
        deps=deps,
        existing_state=None,
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
    return await agent.analyze(message, report_type)


WORKER_TOOLS = [call_triage_agent, call_impact_agent, call_report_agent]
