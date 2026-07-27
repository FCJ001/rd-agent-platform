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
async def call_dedup_check(issue_id: int, runtime: ToolRuntime) -> str:
    """检查问题单是否与现有问题重复。
    适用场景：创建新问题单前、或工程师怀疑当前问题是已知问题时。

    Args:
        issue_id: 待检查的问题单ID
    """
    from src.agents.dedup.matcher import get_dedup_matcher
    matcher = get_dedup_matcher()
    result = await matcher.detect(issue_id)

    if not result.is_duplicate:
        return f"问题单 #{issue_id} 未发现重复。"

    lines = [f"问题单 #{issue_id} 可能与以下问题重复："]
    for m in result.matches[:5]:
        lines.append(f"- #{m.issue_id} {m.issue_no}（相似度 {m.combined_score:.0%}）: {m.title}")
    return "\n".join(lines)


WORKER_TOOLS = [call_triage_agent, call_dedup_check]
