"""Supervisor Agent — 总调度 Agent，按用户意图路由到对应 Worker 工具。"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import AsyncRedisSaver

from src.agents.tools.store_tools import save_memory, search_memory
from src.agents.tools.worker_tools import WORKER_TOOLS
from src.agents.tools.tool_call_repair import ToolCallRepairMiddleware
from src.core.config import get_settings
from src.core.deps import UserContext
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.milvus_store import MilvusStore
from src.infra.redis_cache import get_checkpointer_redis

settings = get_settings()


SUPERVISOR_SYSTEM_PROMPT = """你是汽车研发领域的智能总助手，服务于整车研发 ALM（应用生命周期管理）平台。

## 职责

识别用户意图，调用对应的专项助手处理。你**不直接**回答技术问题，必须通过工具调用获取信息后再回复。

## 可调用的专项助手

- **call_triage_agent**：故障诊断
  适用：用户描述车辆故障现象时（如"车机黑屏"、"动力不足"、"空调不制冷"、"充电异常"、"异响"等）
  用法：用户描述故障后，原样传递用户消息给此工具。首轮可能返回追问问题，将追问直接展示给用户。

- **call_impact_agent**：变更影响分析
  适用：评估 CR/变更单的影响范围，检查基线冲突、配置项依赖冲突、是否有在途重复变更
  用法：传入变更描述或 CR 编号，返回影响评估报告（含风险等级和建议）

- **call_report_agent**：报告/日志解读
  适用：解读 DTC 扫描报告、台架测试报告、OTA 回归测试报告等
  用法：传入报告内容，返回指标判定结果和处置建议（自动识别异常指标）

- **call_dedup_check**：问题去重检测
  适用：用户报告新故障时，检查是否与已有未关闭问题重复
  用法：传入用户的问题描述，如发现重复则返回已有诊断结论，避免重复诊断

- **call_knowledge_agent**：研发知识库搜索
  适用：用户询问技术参数、规范标准、设计文档、竞品信息时
  用法：传入搜索问题，返回知识库答案和参考来源

- **call_operation_agent**：运营 BI 数据查询
  适用：用户询问统计数据、趋势分析、质量报表时
  用法：传入查询问题，返回统计结果和数据明细

## 记忆工具

- **search_memory**：检索该用户/车辆的历史故障记录和诊断历史
- **save_memory**：将重要的诊断结论、车辆信息保存到长期记忆

## 工作原则

1. **每次收到用户消息，必须先调用 search_memory** 检索历史记录，再决定下一步。
2. 用户描述新故障现象时 → 先 search_memory → 再 call_dedup_check 检查重复 → 再 call_triage_agent。
3. 用户提到"之前"、"上次"、"又出现"、"也出现过"、"再来一次"等回顾性表述 → search_memory 查历史诊断记录，如命中则告知用户之前的结论，并询问是否需要重新诊断。
4. 诊断完成后 → save_memory 保存关键结论（根因、现象、DTC码）。
5. 传递消息规则：message 参数原样传递用户输入，不要改写。
6. 语气专业但易懂，不过度使用专业术语。
7. 用户询问技术参数、规范标准、专业知识时 → 先 call_knowledge_agent 查知识库。
8. 用户询问统计数据、趋势报表时 → 调 call_operation_agent 查询 BI 数据。
9. 安全优先：涉及高压电、制动系统等安全相关故障时，提醒用户停车检查。
"""


async def create_supervisor_agent():
    """创建 Supervisor Agent，装配 Redis checkpointer + Milvus 长期记忆。"""

    # Redis checkpointer（短期记忆）
    redis_client = get_checkpointer_redis()
    checkpointer = AsyncRedisSaver(redis_client=redis_client)
    await checkpointer.asetup()

    # Milvus Store（长期记忆）
    from langchain_community.embeddings import DashScopeEmbeddings
    milvus_alias = get_milvus_client_alias()
    embedding_model = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=settings.DASHSCOPE_API_KEY)
    store = MilvusStore(alias=milvus_alias, embeddings=embedding_model, dims=1024)

    # LLM（使用 DashScope qwen-max，与分诊 LLM 一致）
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.7,
        timeout=60,
    )

    tools = [save_memory, search_memory] + WORKER_TOOLS

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        context_schema=UserContext,
        middleware=[
            SummarizationMiddleware(
                model=llm,
                trigger=[("tokens", 4000), ("messages", 6)],
                keep=("messages", 6),
            ),
            ToolCallRepairMiddleware(),
        ],
        checkpointer=checkpointer,
        store=store,
    )

    return agent


_supervisor_agent = None


async def get_supervisor_agent():
    """获取 Supervisor Agent 单例。"""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = await create_supervisor_agent()
    return _supervisor_agent
