# ============================================================
# 编排图（P1 单图化）：Supervisor ⇄ 分诊会话子图 同图协作
# 设计：docs/multi-agent-architecture-design.md §6.1/§6.3
#
#   ┌────────────────── Orchestrator Graph（一个 checkpointer）─────────────────┐
#   │ entry_router: active_task=="triage" ? triage : supervisor（粘性路由）    │
#   │                                                                           │
#   │ supervisor 节点（create_agent，无自己的 checkpointer）                    │
#   │   tools = 记忆×3 + 单轮专才 + call_triage_agent(handoff)                  │
#   │   handoff = Command(goto="triage", graph=Command.PARENT)                  │
#   │                                                                           │
#   │ triage 节点 = build_triage_session_graph()（多轮子图，interrupt 图内）    │
#   │ triage_done 节点 = 清 active_task → supervisor（结论回合由它收尾展示）    │
#   └───────────────────────────────────────────────────────────────────────────┘
#
# 与旧路径（agent-as-tool + TriageSessionStore + 快进空转）的取舍：
#   消失的复杂度 —— 外部进度存储、快进对齐、busy 分流的状态错位推演、
#                  「进度过期但 interrupt 还挂着」的对账分支；
#   保留的资产   —— 锁/闸门/eval/反馈闭环/全部节点实现（session_graph 复用）。
#
# 灰度：SUBGRAPH_TRIAGE_ENABLED（默认 false）。旧路径完整保留，REST/webhook
# 入口继续走旧引擎，chat 入口按开关切换；存量挂起会话走旧路径 TTL 自然消亡。
#
# 机制依据：scripts/probe_orchestrator_mechanics.py（四项探针全 PASS）。
# ============================================================

import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from src.agents.supervisor_agent import assemble_supervisor
from src.agents.tools.history_tools import search_past_diagnoses
from src.agents.tools.store_tools import save_memory, search_memory
from src.agents.tools.worker_tools import WORKER_TOOLS
from src.agents.triage.session_graph import build_triage_session_graph
from src.core.config import get_settings
from src.core.deps import UserContext
from src.core.logger import logger

settings = get_settings()


class OrchestratorState(TypedDict, total=False):
    """父图状态：刻意保持最小 —— 只放编排需要的键。

    分诊的证据字段（confirmed/denied/round…）留在子图自己的 checkpoint
    里，不进父图：探针 P4 验证子图节点二次进入时状态全新，上一场诊断
    不残留，也避免父图长成「上帝状态」。
    """

    messages: Annotated[list, add_messages]
    active_task: str | None      # 活跃子智能体任务名（粘性路由依据）
    task_scope: dict             # handoff 时写入：role/business_line/user_id/session_id
                                  # （中性命字：任何子智能体的作用域注入都走这里）


@tool
async def call_triage_agent(message: str, runtime: ToolRuntime[UserContext]) -> Command:
    """启动故障诊断流程，系统自动多轮追问直至给出结论。
    适用场景：用户描述车辆故障现象（如"车机黑屏"、"动力不足"、"充电异常"等），
    需要系统化诊断、分析根因时。追问期间用户的回复由系统自动转交给诊断流程，
    无需再次调用本工具；诊断结论出来后回到你手中，请把结论和操作选项完整展示给用户。

    Args:
        message: 用户描述的故障现象（原文传递，不要改写）
    """
    ctx = runtime.context
    # ★ 作用域来自服务端认证上下文（与旧工具一致），不由模型决定
    return Command(
        goto="triage",
        graph=Command.PARENT,
        update={
            "active_task": "triage",
            "task_scope": {
                "viewer_role": ctx.role,
                "business_line": ctx.business_line or "",
                "user_id": ctx.user_id,
                "session_id": ctx.session_id,
            },
            # 症状描述随 handoff 进入共享消息历史，子图 intake 之后的
            # extract_phenomena 从这里取「最后一条用户消息」
            "messages": [HumanMessage(content=message)],
        },
    )


async def build_orchestrator_agent(*, llm=None, triage_deps=None, checkpointer=None, store=None):
    """构建编排图。参数可注入（测试用假模型/假 deps/MemorySaver），缺省走生产装配。"""
    if llm is None:
        llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.BASE_URL_CHAT,
            temperature=0.7,
            timeout=60,
        )
    if triage_deps is None:
        from src.agents.workers.triage_agent import TriageAgent
        triage_deps = TriageAgent()._build_deps()
    if checkpointer is None:
        # 生产路径：一次配齐 checkpointer + Milvus store（测试注入 MemorySaver
        # 时跳过，绝不触碰真实后端）
        checkpointer, store = await _default_memory_backend()

    # supervisor：工具集 = 旧集合去掉工具版 call_triage_agent，换成同名 handoff
    # （同名 ⇒ SUPERVISOR_SYSTEM_PROMPT 一字不改即可复用）
    tools = (
        [save_memory, search_memory, search_past_diagnoses]
        + [t for t in WORKER_TOOLS if t.name != "call_triage_agent"]
        + [call_triage_agent]
    )
    supervisor = assemble_supervisor(llm, tools, checkpointer=None, store=None)

    async def _triage_done(state: OrchestratorState) -> dict:
        """诊断结束（收敛/退出/安全终止）→ 清粘性标记，控制权回 supervisor。"""
        logger.info("[ORCH] triage_done 结论回合交回 supervisor")
        return {"active_task": None}

    def _entry_router(state: OrchestratorState) -> str:
        # 粘性路由：有活跃诊断任务直达子图（追问轮不过 LLM 路由）。
        # 注意：interrupt 挂起期间的 resume 不走这里（任务未结束，langgraph
        # 从断点续跑）；这里兜的是「子图跑完一轮返回 END 后下一条新消息」
        # 之前被错误路由的窗口 —— active_task 只在诊断存活期为 "triage"。
        return "triage" if state.get("active_task") == "triage" else "supervisor"

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("triage", build_triage_session_graph(triage_deps))
    graph.add_node("triage_done", _triage_done)
    graph.set_conditional_entry_point(_entry_router, {
        "supervisor": "supervisor",
        "triage": "triage",
    })
    graph.add_edge("supervisor", END)
    graph.add_edge("triage", "triage_done")
    # 结论回合由 supervisor 收尾：结论已在共享消息历史里，
    # 它按工作原则 2 展示结论 + 三个操作选项（与旧行为一致）
    graph.add_edge("triage_done", "supervisor")

    return graph.compile(checkpointer=checkpointer, store=store)


async def _default_memory_backend():
    """生产装配：Redis checkpointer（TTL+读时续期）+ Milvus 长期记忆。"""
    from langgraph.checkpoint.redis import AsyncRedisSaver

    from src.infra.milvus_client import get_milvus_client_alias
    from src.infra.milvus_store import MilvusStore
    from src.infra.redis_cache import get_checkpointer_redis

    # ★ 前向兼容债：checkpoint 序列化对 TriagePhase/CandidateCause 打
    #   "unregistered type" 警告（未来版本会 block）。langgraph 升级前需给
    #   AsyncRedisSaver 配 allowed_msgpack_modules —— 与契约测试同批处理。
    redis_client = get_checkpointer_redis()
    checkpointer = AsyncRedisSaver(
        redis_client=redis_client,
        ttl={
            "default_ttl": settings.CHECKPOINTER_TTL_MINUTES,
            "refresh_on_read": True,
        },
    )
    await checkpointer.asetup()

    from langchain_community.embeddings import DashScopeEmbeddings
    milvus_alias = await asyncio.to_thread(get_milvus_client_alias)
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v3", dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    store = MilvusStore(alias=milvus_alias, embeddings=embedding_model, dims=1024)
    return checkpointer, store


_orchestrator_agent = None
_orchestrator_lock = asyncio.Lock()


async def get_orchestrator_agent():
    """编排图单例（双检锁，与 supervisor 单例同款）。"""
    global _orchestrator_agent
    if _orchestrator_agent is None:
        async with _orchestrator_lock:
            if _orchestrator_agent is None:
                _orchestrator_agent = await build_orchestrator_agent()
    return _orchestrator_agent
