# ============================================================
# 无状态旁路助手 —— 分诊追问挂起期间，离题消息的就地回答
# 设计：docs/design-side-assistant-bypass.md §3.3
#
# 核心性质：**纯读取，不碰图**。不带 checkpointer（单轮、无状态），
# interrupt 位置 / checkpointer / triage_state store 三者零改动。
#
# 两条红线（漏了就是事故）：
#   1. 必须摘掉 call_triage_agent —— 否则模型可能在旁路线程另起一次
#      分诊，与挂起中的目标诊断并跑（两个 thread 抢同一份 store 进度）。
#   2. 必须透传 context=UserContext —— BI/去重/知识库/历史结论全部从
#      runtime.context 取业务线作用域，漏传 = 旁路回答跨业务线取数
#      （本项目已有跨租户泄漏门禁，属红线）。
# ============================================================

import asyncio

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.agents.tools.history_tools import search_past_diagnoses
from src.agents.tools.worker_tools import WORKER_TOOLS
from src.core.config import get_settings
from src.core.deps import UserContext

SIDE_ASSISTANT_TIMEOUT_SECONDS = 45  # 含工具调用的整轮上限；超时由调用方兜底回放追问

SIDE_ASSISTANT_PROMPT = """你是汽车研发 ALM 平台的事务助手。用户正在一次车辆故障诊断中，诊断暂停在一个追问上，
现在发来了一条与诊断无关的消息，你只处理这一条。要求：

1. **不要给出诊断结论、不要重复诊断、不要调用故障诊断工具**（诊断稍后会自己继续）。
2. 回答简洁，通常三句话以内；用户问的是别的事，就只答那件事。
3. 涉及统计数据、报表、知识库、报告解读时，调用对应工具获取真实数据，不要编造。
4. 查不到就直说查不到，不要猜测。
"""


def _side_tools(exclude_tool: str | None = None):
    """WORKER_TOOLS 去掉【当前活跃子智能体的入口工具】+ 历史诊断检索。

    exclude_tool 由 chat 层按挂起 interrupt 的 type 从注册表
    （sub_agents.entry_tool_for）查得：分诊挂起时摘 call_triage_agent，
    以后其它子智能体挂起时摘各自的入口 —— 旁路线程不得另起同种流程。
    平台工具（建单/关联/结案）按 bypass 设计 §3.3 保留 —— 用户显式要求
    操作问题单时旁路也该能办；system prompt 已约束不主动建议写操作。
    """
    tools = [t for t in WORKER_TOOLS if t.name != exclude_tool] if exclude_tool else list(WORKER_TOOLS)
    return tools + [search_past_diagnoses]


async def create_side_assistant(exclude_tool: str | None = None):
    """构建旁路助手。★ 无 checkpointer / 无 store：单轮、无状态。"""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.SIDE_ASSISTANT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.3,
        timeout=30,
    )
    return create_agent(
        model=llm,
        tools=_side_tools(exclude_tool),
        system_prompt=SIDE_ASSISTANT_PROMPT,
        # ★ 与 Supervisor 同一个 context_schema：工具里的 runtime.context
        #   取业务线/角色作用域，跨租户红线靠它守住
        context_schema=UserContext,
    )


# 按摘除工具分别缓存的实例表（不同活跃任务摘不同入口工具）
_side_assistants: dict[str | None, object] = {}
_side_lock = asyncio.Lock()


async def get_side_assistant(exclude_tool: str | None = None):
    """按 exclude_tool 缓存的实例（双检锁）。无状态可全局复用。"""
    if exclude_tool not in _side_assistants:
        async with _side_lock:
            if exclude_tool not in _side_assistants:
                _side_assistants[exclude_tool] = await create_side_assistant(exclude_tool)
    return _side_assistants[exclude_tool]


async def answer_offtopic(
    message: str, question: str, ctx: UserContext, exclude_tool: str | None = None,
) -> str:
    """就地回答一条离题消息。失败（超时/模型异常）抛出，由 chat 层兜底回放追问。

    ★ 失败绝不回落去 resume：把离题文本喂进分诊解析会白烧一轮且污染现象集
    （bypass 设计 §3.4）。
    挂起追问原文随用户消息注入（system prompt 是构建期参数，单例不能按轮改），
    帮模型分清「哪些属于诊断、哪些是这条消息要办的事」。
    """
    agent = await get_side_assistant(exclude_tool)
    user_content = (
        f"（背景：用户正在诊断中，系统挂起的追问是「{question[:300]}」，不用管它）\n"
        f"用户这条消息：{message}"
    )
    result = await asyncio.wait_for(
        agent.ainvoke(
            {"messages": [{"role": "user", "content": user_content}]},
            # 无 checkpointer，config 仅作占位；context 必须传（红线 2）
            config={"configurable": {"thread_id": f"side:{ctx.user_id}:{ctx.session_id}"}},
            context=ctx,
        ),
        timeout=SIDE_ASSISTANT_TIMEOUT_SECONDS,
    )
    return result["messages"][-1].content or ""
